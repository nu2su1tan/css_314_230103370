import os
import sys
import json
import time
import hashlib
import platform
import threading
import subprocess
import numpy as np
from numba import njit

try:
    import resource
except ImportError:
    resource = None

GIT_USERNAME = "nu2su1tan"
NUM_THREADS = 4
ITERATIONS_PER_THREAD = 60_000_000
STRIDES = {"A_contiguous": 1, "B_half_line": 4, "C_line_padded": 8, "D_prefetch_safe": 16}


def detect_cache_line_size() -> int:
    linux_path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
    if os.path.exists(linux_path):
        with open(linux_path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    if sys.platform == "darwin":
        out = subprocess.check_output(["sysctl", "-n", "hw.cachelinesize"], encoding="utf-8")
        return int(out.strip())
    return 64


def cpu_model_string() -> str:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    if sys.platform == "darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], encoding="utf-8").strip()
        except Exception:
            pass
    return platform.processor() or platform.machine()


def init_system_context(git_user: str) -> dict:
    assert git_user != "REPLACE_WITH_YOUR_GIT_USERNAME" and len(git_user.strip()) > 0, (
        "Set GIT_USERNAME to your real Git handle before running."
    )
    logical_cores = os.cpu_count() or 1
    if logical_cores < NUM_THREADS:
        raise RuntimeError(f"Requires at least {NUM_THREADS} logical cores; detected {logical_cores}.")
    line_size = detect_cache_line_size()
    cpu_model = cpu_model_string()
    cpu_sig = f"{platform.machine()}::{platform.processor()}::{logical_cores}"

    digest = hashlib.sha256(f"{git_user}|{cpu_sig}|{cpu_model}".encode()).hexdigest()
    instance_seed = int(digest[:16], 16)

    affinity_supported = hasattr(os, "sched_setaffinity")
    if affinity_supported:
        cores = sorted(os.sched_getaffinity(0))[:NUM_THREADS]
    else:
        cores = []

    return {
        "git_username": git_user,
        "logical_cores": logical_cores,
        "cache_line_size": line_size,
        "cpu_model": cpu_model,
        "cpu_sig": cpu_sig,
        "instance_seed": instance_seed,
        "affinity_supported": affinity_supported,
        "affinity_cores": cores,
    }


def aligned_zeros(n_elems: int, align: int = 64) -> np.ndarray:
    raw = np.zeros(n_elems * 8 + align, dtype=np.uint8)
    off = (-raw.ctypes.data) % align
    arr = raw[off:off + n_elems * 8].view(np.uint64)
    assert arr.ctypes.data % align == 0
    return arr


@njit(nogil=True)
def worker_strided_accumulate(buffer, slot, iters, seed, zero):
    for i in range(iters):
        idx = slot + (i & zero)
        buffer[idx] = buffer[idx] + np.uint64((seed + i) & 7)


@njit(nogil=True)
def worker_private_accumulate(buffer, slot, iters, seed):
    acc = np.uint64(0)
    for i in range(iters):
        acc += np.uint64((seed + i) & 7)
    buffer[slot] = acc


def expected_total(seed: int, iters: int) -> int:
    full, rem = divmod(iters, 8)
    total = full * 28
    start = seed + full * 8
    for j in range(rem):
        total += (start + j) & 7
    return total


def thread_seed(ctx: dict, tid: int) -> int:
    return ((ctx["instance_seed"] >> 8) & 0xFFFF) + tid


def _run_threads(kernel_call, ctx):
    cores = ctx["affinity_cores"]

    def target(tid):
        if ctx["affinity_supported"] and cores:
            try:
                os.sched_setaffinity(0, {cores[tid % len(cores)]})
            except OSError:
                pass
        kernel_call(tid)

    threads = [threading.Thread(target=target, args=(t,)) for t in range(NUM_THREADS)]
    ru0 = resource.getrusage(resource.RUSAGE_SELF) if resource else None
    t0 = time.perf_counter_ns()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter_ns() - t0
    ru1 = resource.getrusage(resource.RUSAGE_SELF) if resource else None
    vol = (ru1.ru_nvcsw - ru0.ru_nvcsw) if ru0 else None
    invol = (ru1.ru_nivcsw - ru0.ru_nivcsw) if ru0 else None
    return elapsed, vol, invol


def run_stride_benchmark(ctx: dict, iters: int = ITERATIONS_PER_THREAD) -> dict:
    results, finals = {}, {}
    for name, stride in STRIDES.items():
        buf = aligned_zeros(NUM_THREADS * stride + 16)

        def call(tid, buf=buf, stride=stride):
            worker_strided_accumulate(buf, tid * stride, iters, thread_seed(ctx, tid), 0)

        elapsed, vol, invol = _run_threads(call, ctx)
        results[name] = {
            "stride": stride, "elapsed_ns": elapsed,
            "ns_per_update": elapsed / iters,
            "voluntary_ctx_switches": vol, "involuntary_ctx_switches": invol,
        }
        finals[name] = [int(buf[t * stride]) for t in range(NUM_THREADS)]

    buf = aligned_zeros(NUM_THREADS * 8 + 16)

    def call_priv(tid):
        worker_private_accumulate(buf, tid * 8, iters, thread_seed(ctx, tid))

    elapsed, vol, invol = _run_threads(call_priv, ctx)
    results["private_baseline"] = {
        "stride": 8, "elapsed_ns": elapsed, "ns_per_update": elapsed / iters,
        "voluntary_ctx_switches": vol, "involuntary_ctx_switches": invol,
    }
    finals["private_baseline"] = [int(buf[t * 8]) for t in range(NUM_THREADS)]
    return {"timings": results, "finals": finals}


def audit_llvm_and_parity(ctx: dict, bench: dict, iters: int = ITERATIONS_PER_THREAD) -> dict:
    expected = [expected_total(thread_seed(ctx, t), iters) for t in range(NUM_THREADS)]
    parity = all(v == expected for v in bench["finals"].values())

    ir_map = worker_strided_accumulate.inspect_llvm()
    ir = "\n".join(ir_map.values())

    native_parts = []
    for b in ir.split("\ndefine ")[1:]:
        header = b.split("{")[0]
        if "worker_strided_accumulate" in header and "cpython" not in header:
            native_parts.append(b.split("\n}")[0])
    native_ir = "\n".join(native_parts) or ir

    has_i64_store = "store i64" in native_ir
    has_i64_load = "load i64" in native_ir
    no_pyobject = "PyObject" not in native_ir and "PyEval" not in native_ir

    with open("worker_strided_accumulate.ll", "w", encoding="utf-8") as f:
        f.write(ir)

    t = bench["timings"]
    manifest = {
        "git_username": ctx["git_username"],
        "instance_seed": ctx["instance_seed"],
        "cpu_model": ctx["cpu_model"],
        "logical_cores": ctx["logical_cores"],
        "cache_line_size": ctx["cache_line_size"],
        "threads": NUM_THREADS,
        "iterations_per_thread": iters,
        "timings": t,
        "slowdown_stride1_vs_stride8": t["A_contiguous"]["elapsed_ns"] / t["C_line_padded"]["elapsed_ns"],
        "expected_final_values": expected,
        "final_values": bench["finals"],
        "parity_ok": parity,
        "llvm": {"store_i64": has_i64_store, "load_i64": has_i64_load, "no_pyobject": no_pyobject},
    }
    manifest["verification_passed"] = bool(parity and has_i64_store and no_pyobject)
    with open("cache_lab_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main():
    ctx = init_system_context(GIT_USERNAME)
    print("Context:", {k: v for k, v in ctx.items() if k != "cpu_sig"})

    w = aligned_zeros(64)
    worker_strided_accumulate(w, 0, 10, 1, 0)
    worker_private_accumulate(w, 0, 10, 1)

    bench = run_stride_benchmark(ctx)
    for name, r in bench["timings"].items():
        print(f"{name:18s} {r['elapsed_ns']:>16,d} ns  {r['ns_per_update']:.3f} ns/update")
    m = audit_llvm_and_parity(ctx, bench)
    print("slowdown stride1/stride8: %.2fx" % m["slowdown_stride1_vs_stride8"])
    print("verification_passed:", m["verification_passed"])


if __name__ == "__main__":
    main()