
import argparse
import ctypes
import json
import math
import multiprocessing as mp
import os
import platform
import statistics
import subprocess
import sys
import time
from array import array
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Callable




@dataclass
class BenchStats:
    min_ms: float
    median_ms: float
    iqr_ms: float
    max_ms: float


def fnv1a_hash(s: str) -> int:
    h = 14695981039346656037
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def run_benchmark(fn: Callable[[], None], warmup_runs: int = 3, timed_runs: int = 9) -> BenchStats:
    for _ in range(warmup_runs):
        fn()
    samples = []
    for _ in range(timed_runs):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        samples.append((t1 - t0) / 1e6)
    samples.sort()
    mid = timed_runs // 2
    med = samples[mid] if timed_runs % 2 == 1 else 0.5 * (samples[mid - 1] + samples[mid])
    q25 = samples[timed_runs // 4]
    q75 = samples[(3 * timed_runs) // 4]
    return BenchStats(samples[0], med, q75 - q25, samples[-1])


def emit_verify_tag(task_id: str, student_seed: int, primary_metric: float, secondary_metric: float) -> str:
    hw = mp.cpu_count()
    payload = f"{task_id}|{student_seed}|{hw}|{primary_metric:.4f}|{secondary_metric:.4f}"
    sig = fnv1a_hash(payload)
    return f"VERIFY_HASH[{task_id}]: {sig:016X} (P_log={hw})"


LOG_LINES = []


def log(s=""):
    print(s, flush=True)
    LOG_LINES.append(str(s))


def md_table(headers, rows):
    log("| " + " | ".join(headers) + " |")
    log("|" + "|".join("---" for _ in headers) + "|")
    for r in rows:
        log("| " + " | ".join(str(c) for c in r) + " |")
    log()


def sh(cmd):
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return (out.stdout or "").strip()
    except Exception as e:  # noqa
        return f"<error: {e}>"


def safe_close(shm):
    try:
        shm.close()
    except BufferError:
        pass


def p_list(plog):
    res, p = [], 1
    while p < plog:
        res.append(p)
        p *= 2
    res.append(plog)
    return sorted(set(res))


def split_ranges(n, p):
    return [(n * t // p, n * (t + 1) // p) for t in range(p)]





def fs_worker(shm_name: str, slot_idx: int, iters: int, nslots: int) -> None:
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = (ctypes.c_uint64 * nslots).from_buffer(shm.buf)
    acc = arr[slot_idx]
    mask = 0xFFFFFFFFFFFFFFFF
    mult = 6364136223846793005
    for i in range(iters):
        acc = (acc * mult + (i | 1)) & mask
        arr[slot_idx] = acc
    del arr
    safe_close(shm)


def run_fs(p, stride, M, nslots):
    shm = shared_memory.SharedMemory(create=True, size=nslots * 8)
    try:
        procs = [mp.Process(target=fs_worker, args=(shm.name, t * stride, M // p, nslots)) for t in range(p)]
        for x in procs:
            x.start()
        for x in procs:
            x.join()
    finally:
        safe_close(shm)
        shm.unlink()


def karp_flatt(sp, p):
    return (1.0 / sp - 1.0 / p) / (1.0 - 1.0 / p)





def stage_a(n):
    acc = 0.0
    for i in range(int(0.05 * n)):
        x = i + 1.0
        for _ in range(6): 
            acc += math.sqrt(x + acc * 1e-9) + math.sin(x)
    return acc


def stage_b(rng):
    lo, hi = rng
    acc = 0.0
    sin = math.sin
    for i in range(lo, hi):
        x = i * 1e-6
        for _ in range(35):
            x = x * 1.0000001 + 0.5
            acc += sin(x)
    return acc


def bench_pipeline(p, n, warm, runs):
    ranges = split_ranges(n, p)
    a_times = []
    pool = mp.Pool(processes=p)  
    try:
        def fn():
            t0 = time.perf_counter_ns()
            stage_a(n)
            a_times.append((time.perf_counter_ns() - t0) / 1e6)
            pool.map(stage_b, ranges, chunksize=1)

        st = run_benchmark(fn, warm, runs)
    finally:
        pool.close()
        pool.join()
    return st.median_ms, statistics.median(a_times[-runs:])





def copy_worker(na, nb, start, end, barrier, out, idx):
    a = shared_memory.SharedMemory(name=na)
    b = shared_memory.SharedMemory(name=nb)
    va, vb = a.buf, b.buf
    barrier.wait()
    t0 = time.perf_counter_ns()
    va[start:end] = vb[start:end]
    out[idx] = (time.perf_counter_ns() - t0) / 1e6
    del va, vb
    safe_close(a)
    safe_close(b)


def fma_worker(iters, barrier, out, idx):
    a, b = 1.0000001, 1e-7
    x0 = x1 = x2 = x3 = x4 = x5 = x6 = x7 = 1.0
    x8 = x9 = x10 = x11 = x12 = x13 = x14 = x15 = 1.0
    barrier.wait()
    t0 = time.perf_counter_ns()
    for _ in range(iters):
        x0 = x0 * a + b
        x1 = x1 * a + b
        x2 = x2 * a + b
        x3 = x3 * a + b
        x4 = x4 * a + b
        x5 = x5 * a + b
        x6 = x6 * a + b
        x7 = x7 * a + b
        x8 = x8 * a + b
        x9 = x9 * a + b
        x10 = x10 * a + b
        x11 = x11 * a + b
        x12 = x12 * a + b
        x13 = x13 * a + b
        x14 = x14 * a + b
        x15 = x15 * a + b
    out[idx] = (time.perf_counter_ns() - t0) / 1e6
    
    if x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15 < 0:
        print("unreachable")


def measure_bandwidth(pphys, runs, size):
    a = shared_memory.SharedMemory(create=True, size=size)
    b = shared_memory.SharedMemory(create=True, size=size)
    try:
        chunk = 8 * 1024 * 1024
        blk = b"\x01" * chunk
        for off in range(0, size, chunk):
            end = min(off + chunk, size)
            a.buf[off:end] = blk[: end - off]
            b.buf[off:end] = blk[: end - off]
        ranges = split_ranges(size, pphys)
        times = []
        for r in range(runs + 1): 
            barrier = mp.Barrier(pphys)
            out = mp.Array("d", pphys)
            procs = [mp.Process(target=copy_worker, args=(a.name, b.name, s, e, barrier, out, t))
                     for t, (s, e) in enumerate(ranges)]
            for x in procs:
                x.start()
            for x in procs:
                x.join()
            if r > 0:
                times.append(max(out[:]))
        return statistics.median(times)
    finally:
        safe_close(a)
        safe_close(b)
        a.unlink()
        b.unlink()


def measure_compute(pphys, runs, flops_per_worker):
    iters = int(flops_per_worker / 32) 
    times = []
    for r in range(runs + 1):
        barrier = mp.Barrier(pphys)
        out = mp.Array("d", pphys)
        procs = [mp.Process(target=fma_worker, args=(iters, barrier, out, t)) for t in range(pphys)]
        for x in procs:
            x.start()
        for x in procs:
            x.join()
        if r > 0:
            times.append(max(out[:]))
    return statistics.median(times), iters * 32





def gemm_worker(variant, bs, na, nb, nc, N, r0, r1):
    sa = shared_memory.SharedMemory(name=na)
    sb = shared_memory.SharedMemory(name=nb)
    sc = shared_memory.SharedMemory(name=nc)
    nbytes = N * N * 8
    ba, bb, bc = sa.buf[:nbytes], sb.buf[:nbytes], sc.buf[:nbytes]
    A, B, C = ba.cast("d"), bb.cast("d"), bc.cast("d")
    C[r0 * N:r1 * N] = array("d", bytes(8 * (r1 - r0) * N))
    if variant == 1:
        for i in range(r0, r1):
            for j in range(N):
                dot = 0.0
                for k in range(N):
                    dot += A[i * N + k] * B[k * N + j]
                C[i * N + j] = dot
    elif variant == 2:
        for i in range(r0, r1):
            ci = i * N
            for k in range(N):
                a_ik = A[i * N + k]
                bk = k * N
                for j in range(N):
                    C[ci + j] += a_ik * B[bk + j]
    else:
        for ii in range(r0, r1, bs):
            i_end = min(ii + bs, r1)
            for kk in range(0, N, bs):
                k_end = min(kk + bs, N)
                for jj in range(0, N, bs):
                    j_end = min(jj + bs, N)
                    for i in range(ii, i_end):
                        ci = i * N
                        for k in range(kk, k_end):
                            a_ik = A[i * N + k]
                            bk = k * N
                            for j in range(jj, j_end):
                                C[ci + j] += a_ik * B[bk + j]
    for v in (A, B, C, ba, bb, bc):
        v.release()
    for s in (sa, sb, sc):
        safe_close(s)


class Gemm:
    def __init__(self, N, p):
        import random
        self.N, self.p = N, p
        nb = N * N * 8
        self.shms = [shared_memory.SharedMemory(create=True, size=nb) for _ in range(3)]
        rng = random.Random(12345)
        self.views = []
        for s in self.shms:
            base = s.buf[:nb]
            self.views.append((base, base.cast("d")))
        self.views[0][1][:] = array("d", [rng.random() for _ in range(N * N)])
        self.views[1][1][:] = array("d", [rng.random() for _ in range(N * N)])

    def run(self, variant, bs=0):
        N, p = self.N, self.p
        procs = [mp.Process(target=gemm_worker,
                            args=(variant, bs, self.shms[0].name, self.shms[1].name, self.shms[2].name,
                                  N, N * t // p, N * (t + 1) // p)) for t in range(p)]
        for x in procs:
            x.start()
        for x in procs:
            x.join()

    def snapshot_c(self):
        return array("d", self.views[2][1])

    def close(self):
        for base, mv in self.views:
            mv.release()
            base.release()
        for s in self.shms:
            safe_close(s)
            s.unlink()





def work(i):
    acc = 0.0
    x = 0.5 + i * 1e-3
    sin, cos, sqrt = math.sin, math.cos, math.sqrt
    for _ in range(1 + (i // 60) ** 2):
        acc += sin(x) * cos(x) + sqrt(x + acc * 1e-12)
    return acc


def p4_worker(mode, t, p, N, C, counter, lock, times):
    t0 = time.perf_counter_ns()
    if mode == 0:
        for i in range(t * N // p, (t + 1) * N // p):
            work(i)
    elif mode == 1:
        for i in range(t, N, p):
            work(i)
    else:
        while True:
            with lock:
                start = counter.value
                counter.value = start + C
            if start >= N:
                break
            for i in range(start, min(start + C, N)):
                work(i)
    times[t] = (time.perf_counter_ns() - t0) / 1e6


def run_sched(mode, p, N, C, runs):
    results = []
    for _ in range(runs):
        times = mp.Array("d", p)
        counter = mp.RawValue("i", 0)
        lock = mp.Lock()
        procs = [mp.Process(target=p4_worker, args=(mode, t, p, N, C, counter, lock, times)) for t in range(p)]
        t0 = time.perf_counter_ns()
        for x in procs:
            x.start()
        for x in procs:
            x.join()
        wall = (time.perf_counter_ns() - t0) / 1e6
        tw = list(times[:])
        lam = max(tw) / (sum(tw) / p)
        results.append((wall, lam, tw))
    results.sort(key=lambda r: r[0])
    return results[len(results) // 2]





def detect_hw(pphys_override):
    system = platform.system()
    plog = mp.cpu_count()
    info = {"os": f"{platform.system()} {platform.release()}", "python": platform.python_version(),
            "plog": plog, "raw": {}}
    pphys, line = None, None
    if system == "Linux":
        raw = sh('lscpu | grep -E "Model name|Socket|Core|Thread|L1d|L2|L3"')
        info["raw"]["lscpu"] = raw
        line = sh("getconf LEVEL1_DCACHE_LINESIZE")
        info["raw"]["line_size"] = line
        cores = sh("lscpu -p=CORE,SOCKET | grep -v '^#' | sort -u | wc -l")
        pphys = int(cores) if cores.isdigit() and int(cores) > 0 else None
    elif system == "Darwin":
        raw = sh("sysctl -n machdep.cpu.brand_string && sysctl hw.physicalcpu hw.logicalcpu "
                 "hw.cachelinesize hw.l1dcachesize hw.l2cachesize hw.l3cachesize")
        info["raw"]["sysctl"] = raw
        v = sh("sysctl -n hw.physicalcpu")
        pphys = int(v) if v.isdigit() else None
        line = sh("sysctl -n hw.cachelinesize")
    elif system == "Windows":
        raw = sh('powershell -NoProfile -Command "Get-CimInstance Win32_Processor | Select Name,'
                 'NumberOfCores,NumberOfLogicalProcessors,L2CacheSize,L3CacheSize | Format-List"')
        info["raw"]["powershell"] = raw
        v = sh('powershell -NoProfile -Command "(Get-CimInstance Win32_Processor | '
               'Measure-Object NumberOfCores -Sum).Sum"')
        pphys = int(v) if v.isdigit() else None
        line = "64 (assumed, not queryable on Windows)"
    if pphys_override:
        pphys = pphys_override
    if not pphys or pphys > plog:
        pphys = plog if not pphys else min(pphys, plog)
    info["pphys"] = pphys
    info["line_size"] = line if line and line != "0" else "64 (default)"
    return info





def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, required=True, help="Student ID (число)")
    ap.add_argument("--pphys", type=int, default=0)
    ap.add_argument("--phases", default="0,1,2,3,4")
    ap.add_argument("--quick", action="store_true", help="только для теста кода!")
    args = ap.parse_args()
    seed = args.id
    quick = args.quick
    phases = {int(x) for x in args.phases.split(",")}
    R = {"student_id": seed, "quick_mode": quick}
    t_start = time.time()

    hw = detect_hw(args.pphys)
    plog, pphys = hw["plog"], hw["pphys"]
    R["hw"] = hw

    log("# LAB OUTPUT")
    log(f"student_id={seed}  quick_mode={quick}  started={time.strftime('%Y-%m-%d %H:%M:%S')}")
    if quick:
        log("!!! QUICK MODE: числа НЕ годятся для сдачи, только проверка кода !!!")
    log()

   
    if 0 in phases:
        t_ph = time.time()
        log("## PHASE 0: Hardware & calibration")
        for k, v in hw["raw"].items():
            log(f"[raw:{k}]")
            log(v)
        log(f"line_size_raw={hw['line_size']}")
        md_table(["Parameter", "Value"], [
            ["OS / Python", f"{hw['os']} / Python {hw['python']}"],
            ["Physical cores (Pphys)", pphys],
            ["Logical threads (Plog)", plog],
            ["Cache line size", hw["line_size"]],
        ])
        n = 200_000 if quick else 2_000_000
        clock = time.perf_counter_ns
        t0 = clock()
        for _ in range(n):
            clock()
        t1 = clock()
        avg_ns = (t1 - t0) / n
        log(f"cpu_count={mp.cpu_count()}  avg_call_latency_ns={avg_ns:.2f}  (n={n})")
        tag = emit_verify_tag("PHASE0", seed, avg_ns, float(mp.cpu_count()))
        log(tag)
        R["phase0"] = {"avg_ns": avg_ns, "hash": tag}
        log(f"[phase 0 time: {time.time() - t_ph:.1f}s]\n")


    if 1 in phases:
        t_ph = time.time()
        log("## PHASE 1: False sharing")
        M = 24_000_000 // (8 if quick else 1)
        warm, runs = (1, 3) if quick else (2, 7)
        nslots = max(512, plog * 32 + 32)
        stride_rows, stride_med = [], {}
        for s in [1, 2, 4, 8, 16, 32]:
            st = run_benchmark(lambda: run_fs(pphys, s, M, nslots), warm, runs)
            stride_med[s] = st.median_ms
            mops = M / (st.median_ms * 1e3)
            stride_rows.append([s, s * 8, f"{st.median_ms:.1f}", f"{st.iqr_ms:.1f}", f"{mops:.2f}"])
            log(f"  stride={s} done: median={st.median_ms:.1f} ms")
        log(f"\nTable 1A (p = Pphys = {pphys}, M = {M})")
        md_table(["stride", "bytes", "median_ms", "iqr_ms", "Mops/s"], stride_rows)
        strides = [1, 2, 4, 8, 16, 32]
        best_ratio, best_i = 0, 0
        for i in range(len(strides) - 1):
            ratio = stride_med[strides[i]] / stride_med[strides[i + 1]]
            if ratio > best_ratio:
                best_ratio, best_i = ratio, i
        inflect_bytes = strides[best_i + 1] * 8
        log(f"Largest drop: stride {strides[best_i]} -> {strides[best_i + 1]} "
            f"(x{best_ratio:.2f}); inflection at {inflect_bytes} bytes")

        table1b = {}
        for s in (1, 16):
            rows, t1 = [], None
            for p in p_list(plog):
                st = run_benchmark(lambda: run_fs(p, s, M, nslots), warm, runs)
                if p == 1:
                    t1 = st.median_ms
                sp = t1 / st.median_ms
                ep = sp / p
                e = karp_flatt(sp, p) if p >= 2 else float("nan")
                rows.append([p, f"{st.median_ms:.1f}", f"{sp:.3f}", f"{ep:.3f}",
                             "-" if p == 1 else f"{e:.4f}"])
                table1b.setdefault(s, []).append({"p": p, "T": st.median_ms, "S": sp, "E": ep, "e": e})
                log(f"  stride={s} p={p} done")
            log(f"\nTable 1B (stride = {s})")
            md_table(["p", "T_p_ms", "S_p", "E_p", "Karp-Flatt e(p)"], rows)
        tag = emit_verify_tag("PHASE1", seed, stride_med[1], stride_med[16])
        log(tag)
        R["phase1"] = {"M": M, "table1a": {str(k): v for k, v in stride_med.items()},
                       "inflection_bytes": inflect_bytes, "table1b": {str(k): v for k, v in table1b.items()},
                       "hash": tag}
        log(f"[phase 1 time: {time.time() - t_ph:.1f}s]\n")


    if 2 in phases:
        t_ph = time.time()
        log("## PHASE 2: Strong & weak scaling")
        N_strong = 1_600_000 // (16 if quick else 1)
        N0 = 250_000 // (16 if quick else 1)
        warm, runs = (1, 3) if quick else (1, 5)
        plist = list(range(1, plog + 1))
        strong, weak = [], []
        for p in plist:
            T, TA = bench_pipeline(p, N_strong, warm, runs)
            strong.append({"p": p, "T": T, "TA": TA})
            log(f"  strong p={p}: {T:.1f} ms")
        T1 = strong[0]["T"]
        for r in strong:
            r["S"] = T1 / r["T"]
            r["E"] = r["S"] / r["p"]
        log(f"\nTable 2A (strong, N={N_strong})")
        md_table(["p", "T_ms", "S_p", "E_p"],
                 [[r["p"], f"{r['T']:.1f}", f"{r['S']:.3f}", f"{r['E']:.3f}"] for r in strong])

        for p in plist:
            T, TA = bench_pipeline(p, p * N0, warm, runs)
            weak.append({"p": p, "T": T, "TA": TA})
            log(f"  weak p={p}: {T:.1f} ms")
        Tw1 = weak[0]["T"]
        for r in weak:
            r["E"] = Tw1 / r["T"]
            r["alpha"] = r["TA"] / r["T"]
            r["SG"] = r["p"] - r["alpha"] * (r["p"] - 1)
        log(f"\nTable 2B (weak, N0={N0}, N(p)=p*N0)")
        md_table(["p", "N(p)", "T_ms", "E_weak", "TStageA_ms", "alpha", "S_Gustafson"],
                 [[r["p"], r["p"] * N0, f"{r['T']:.1f}", f"{r['E']:.3f}", f"{r['TA']:.2f}",
                   f"{r['alpha']:.4f}", f"{r['SG']:.3f}"] for r in weak])

        s_frac = strong[0]["TA"] / strong[0]["T"]
        s_inf = 1.0 / s_frac
        s_th = 1.0 / (s_frac + (1 - s_frac) / plog)
        s_obs = strong[-1]["S"]
        log(f"Serial fraction s = TStageA(1)/T1 = {s_frac:.5f}")
        log(f"Amdahl S_inf = {s_inf:.2f};  theoretical S(Plog={plog}) = {s_th:.3f};  observed = {s_obs:.3f}")
        S = {r["p"]: r["S"] for r in strong}
        marg = {}
        for p in plist[1:]:
            marg[p] = S[p] - S[p - 1]
        log("Marginal speedup dS(p) = S(p)-S(p-1):")
        md_table(["p", "dS"], [[p, f"{marg[p]:.3f}"] for p in plist[1:]])
        smt = {}
        if plog > pphys and pphys >= 2:
            before, after = marg[pphys], marg[pphys + 1]
            drop = (before - after) / before * 100 if before else float("nan")
            avg_after = (S[plog] - S[pphys]) / (plog - pphys)
            avg_before = (S[pphys] - S[1]) / (pphys - 1)
            log(f"Boundary Pphys={pphys} -> {pphys + 1}: dS {before:.3f} -> {after:.3f} "
                f"(drop {drop:.1f}%)")
            log(f"Avg marginal speedup per worker: p<=Pphys: {avg_before:.3f}, p>Pphys: {avg_after:.3f}")
            smt = {"dS_before": before, "dS_after": after, "drop_pct": drop,
                   "avg_before": avg_before, "avg_after": avg_after}
        else:
            log("Plog == Pphys (или Pphys<2): SMT-граница не определена")
        sg_plog = weak[-1]["SG"]
        tag = emit_verify_tag("PHASE2", seed, s_obs, sg_plog)
        log(tag)
        R["phase2"] = {"N_strong": N_strong, "N0": N0, "strong": strong, "weak": weak, "s": s_frac,
                       "S_inf": s_inf, "S_amdahl_plog": s_th, "S_obs_plog": s_obs, "smt": smt, "hash": tag}
        log(f"[phase 2 time: {time.time() - t_ph:.1f}s]\n")


    if 3 in phases:
        t_ph = time.time()
        log("## PHASE 3A: Roofline")
        size = (8 if quick else 96) * 1000 * 1000
        runs_bw = 3 if quick else 7
        t_bw = measure_bandwidth(pphys, runs_bw, size)
        bw = 2 * size / (t_bw * 1e-3) / 1e9
        flops_pw = 8.0e6 if quick else 8.0e7
        t_fp, flops_done = measure_compute(pphys, 3 if quick else 5, flops_pw)
        p_sus = pphys * flops_done / (t_fp * 1e-3) / 1e9
        ridge = p_sus / bw
        log(f"BW_sustained = {bw:.3f} GB/s  (median max-worker time {t_bw:.2f} ms, buffer {size/1e6:.0f} MB)")
        log(f"P_sustained  = {p_sus:.4f} GFLOP/s  ({pphys} procs, median time {t_fp:.1f} ms)")
        log(f"Ridge point I* = {ridge:.6f} FLOP/Byte")

        log("\n## PHASE 3B: GEMM")
        N = 96 if quick else 384
        warm, runs = (0, 1) if quick else (1, 3)
        gemm = Gemm(N, pphys)
        flops = 2 * N ** 3
        try:
            res = {}
            for name, var, bs in [("Variant 1 (i-j-k)", 1, 0), ("Variant 2 (i-k-j)", 2, 0)]:
                st = run_benchmark(lambda: gemm.run(var, bs), warm, runs)
                res[name] = (st.median_ms, flops / (st.median_ms * 1e6))
                log(f"  {name}: {st.median_ms:.1f} ms, {res[name][1]:.5f} GFLOP/s")
            bs_rows, bs_res = [], {}
            for bs in [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]:
                st = run_benchmark(lambda: gemm.run(3, bs), warm, runs)
                g = flops / (st.median_ms * 1e6)
                bs_res[bs] = (st.median_ms, g)
                w = 3 * bs * bs * 8
                bs_rows.append([bs, w, f"{w/1024:.2f}", f"{st.median_ms:.1f}", f"{g:.5f}"])
                log(f"  BS={bs}: {st.median_ms:.1f} ms, {g:.5f} GFLOP/s")
            bs_star = max(bs_res, key=lambda b: bs_res[b][1])

            gemm.run(2)
            c2 = gemm.snapshot_c()
            gemm.run(3, bs_star)
            c3 = gemm.snapshot_c()
            max_diff = max(abs(x - y) for x, y in zip(c2, c3))
        finally:
            gemm.close()
        log("\nTable 3B (GEMM, N={}, p={})".format(N, pphys))
        md_table(["Variant", "median_ms", "GFLOP/s"],
                 [[k, f"{v[0]:.1f}", f"{v[1]:.5f}"] for k, v in res.items()])
        md_table(["BS", "W_tile_bytes", "W_tile_KB", "median_ms", "GFLOP/s"], bs_rows)
        log(f"BS* = {bs_star}  W_tile(BS*) = {3*bs_star*bs_star*8} bytes ({3*bs_star*bs_star*8/1024:.2f} KB)")
        log(f"max |C3 - C2| = {max_diff:.3e}  (требование < 1e-9: {'OK' if max_diff < 1e-9 else 'FAIL'})")

        i_naive = 0.125
        i_tiled = bs_star / 12.0
        rows = []
        g1 = res["Variant 1 (i-j-k)"][1]
        g2 = res["Variant 2 (i-k-j)"][1]
        g3 = bs_res[bs_star][1]
        roof = {}
        for name, g, i in [("Variant 1", g1, i_naive), ("Variant 2", g2, i_naive),
                           (f"Variant 3 (BS*={bs_star})", g3, i_tiled)]:
            attain = min(p_sus, bw * i)
            roof[name] = {"I": i, "attainable": attain, "achieved": g, "pct": g / attain * 100}
            rows.append([name, f"{i:.4f}", f"{attain:.4f}", f"{g:.5f}", f"{g/attain*100:.2f}%"])
        log("\nRoofline mapping")
        md_table(["Kernel", "I (FLOP/B)", "Attainable GFLOP/s = min(P, BW*I)", "Achieved GFLOP/s", "% attainable"],
                 rows)
        tag = emit_verify_tag("PHASE3", seed, g1, g3)
        log(tag)
        R["phase3"] = {"bw": bw, "p_sus": p_sus, "ridge": ridge, "N": N, "variants": res,
                       "bs": {str(k): v for k, v in bs_res.items()}, "bs_star": bs_star,
                       "max_diff": max_diff, "roofline": roof, "hash": tag}
        log(f"[phase 3 time: {time.time() - t_ph:.1f}s]\n")


    if 4 in phases:
        t_ph = time.time()
        log("## PHASE 4: Scheduling & load imbalance")
        N = 3000 if quick else 12000
        runs = 1 if quick else 3
        rows, data = [], {}
        w1, l1, tw1 = run_sched(0, pphys, N, 0, runs)
        data["block"] = {"wall": w1, "lam": l1, "workers": tw1}
        rows.append(["Block static", "-", f"{w1:.1f}", f"{l1:.3f}"])
        log(f"  block: {w1:.1f} ms, lambda={l1:.3f}")
        w2, l2, tw2 = run_sched(1, pphys, N, 0, runs)
        data["cyclic"] = {"wall": w2, "lam": l2, "workers": tw2}
        rows.append(["Cyclic static", "-", f"{w2:.1f}", f"{l2:.3f}"])
        log(f"  cyclic: {w2:.1f} ms, lambda={l2:.3f}")
        dyn = {}
        for C in [1, 4, 16, 64, 256, 1024]:
            w, l, tw = run_sched(2, pphys, N, C, runs)
            dyn[C] = {"wall": w, "lam": l, "workers": tw}
            rows.append(["Dynamic", C, f"{w:.1f}", f"{l:.3f}"])
            log(f"  dynamic C={C}: {w:.1f} ms, lambda={l:.3f}")
        data["dynamic"] = {str(k): v for k, v in dyn.items()}
        log(f"\nTable 4 (N={N}, p={pphys})")
        md_table(["Strategy", "C", "T_wall_ms", "lambda"], rows)
        c_star = min(dyn, key=lambda c: dyn[c]["wall"])
        log(f"C* = {c_star} (T_wall = {dyn[c_star]['wall']:.1f} ms)")
        log("Per-worker active times (ms): block={}, cyclic={}".format(
            [round(x, 1) for x in tw1], [round(x, 1) for x in tw2]))
        tag = emit_verify_tag("PHASE4", seed, w1, dyn[c_star]["wall"])
        log(tag)
        data.update({"N": N, "c_star": c_star, "hash": tag})
        R["phase4"] = data
        log(f"[phase 4 time: {time.time() - t_ph:.1f}s]\n")

    log(f"TOTAL TIME: {(time.time() - t_start) / 60:.1f} min")
    log("\n=== RAW_JSON_BEGIN ===")
    log(json.dumps(R, default=str))
    log("=== RAW_JSON_END ===")
    fname = f"lab_output_{seed}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write("\n".join(LOG_LINES))
    


if __name__ == "__main__":
    mp.freeze_support()
    main()