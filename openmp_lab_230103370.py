# =====================================================================
# SECTION 0: LOCAL OPENMP ENVIRONMENT & DETERMINISTIC SEED HARNESS
# =====================================================================
import os

# Both variables must be set BEFORE importing numba.
os.environ["NUMBA_THREADING_LAYER"] = "omp"  # Force native OpenMP runtime (libgomp / iomp)
# Allow set_num_threads(8) even on a 4/6-core CPU (otherwise Numba raises ValueError).
os.environ.setdefault("NUMBA_NUM_THREADS", str(max(8, os.cpu_count() or 1)))

import math
import time, hashlib, json
import numpy as np
import numba as nb
from numba import njit, prange, set_num_threads, get_num_threads, get_thread_id

STUDENT_NAME = "Nursultan Nazarbay"
STUDENT_ID = 230103370
assert isinstance(STUDENT_ID, int) and STUDENT_ID > 0, "Set a valid numeric STUDENT_ID!"
assert (os.cpu_count() or 1) >= 4, "Local multi-core CPU (>=4 logical cores) required; no Colab!"


# Trigger OpenMP layer initialization and verify backend
@njit(parallel=True)
def _omp_probe():
    acc = 0
    for i in prange(64):
        acc += get_thread_id() + 1
    return acc


_ = _omp_probe()
assert nb.threading_layer() == "omp", f"Expected 'omp' layer, got {nb.threading_layer()}"

rng = np.random.default_rng(STUDENT_ID)

# Student-specific OpenMP parameters
N_T1 = 4_000_000 + (STUDENT_ID % 991) * 1024
BINS_T1 = 48 + (STUDENT_ID % 17)
ITERS_T2 = 35_000_000 + (STUDENT_ID % 503) * 10_000
PAD_STRIDE = 8 + (STUDENT_ID % 5) * 8  # >= 8 int64 slots (>= 64 bytes cache-line isolation)
ROWS_T3 = 12_000 + (STUDENT_ID % 257) * 16
SKEW_MOD = 37 + (STUDENT_ID % 19)
GRID_H = 384 + (STUDENT_ID % 4) * 32
GRID_W = 512 + (STUDENT_ID % 5) * 32
OMEGA_SOR = np.float64(1.72 + (STUDENT_ID % 11) * 0.015)

data_t1 = rng.integers(0, BINS_T1, size=N_T1, dtype=np.int32)
weights_t1 = rng.uniform(0.25, 2.50, size=N_T1).astype(np.float64)

print(f"Backend: {nb.threading_layer()} | Logical Cores: {os.cpu_count()} | Student: {STUDENT_ID}")

THREAD_COUNTS = [1, 2, 4, 8]
MAX_T = 8


def bench(fn, *args, repeat=3):
    """One warm-up call (JIT + caches), then best-of-`repeat` wall-clock time."""
    fn(*args)
    best = float("inf")
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best, result


# =====================================================================
# TASK 1: RACE-CONDITION FORENSICS & THREAD-PRIVATE HISTOGRAM REDUCTION
# =====================================================================
@njit(parallel=True)
def buggy_hist_omp(indices, weights, num_bins):
    hist = np.zeros(num_bins, dtype=np.int64)
    w_sums = np.zeros(num_bins, dtype=np.float64)
    for i in prange(indices.shape[0]):
        b = indices[i]
        hist[b] += 1  # Data race across OpenMP threads
        w_sums[b] += weights[i]  # Non-associative floating-point race
    return hist, w_sums


# Number of FIXED chunks for the float sums. It does not depend on the thread count,
# so the summation order is the same for 1, 2, 4 and 8 threads (bit-identical result).
N_CHUNKS_T1 = 64


@njit(parallel=True)
def safe_hist_omp(indices, weights, num_bins):
    max_threads = get_num_threads()
    n = indices.shape[0]
    chunk = (n + N_CHUNKS_T1 - 1) // N_CHUNKS_T1

    priv_hist = np.zeros((max_threads, num_bins), dtype=np.int64)  # row per THREAD
    priv_w = np.zeros((N_CHUNKS_T1, num_bins), dtype=np.float64)  # row per fixed CHUNK
    worker_hits = np.zeros((max_threads, 8), dtype=np.int64)  # padded: 1 slot per cache line

    for c in prange(N_CHUNKS_T1):
        tid = get_thread_id()  # worker id: writes only to its own row -> no race
        lo = c * chunk
        hi = min(lo + chunk, n)
        for i in range(lo, hi):
            b = indices[i]
            priv_hist[tid, b] += 1
            priv_w[c, b] += weights[i]  # sequential order inside the chunk
        worker_hits[tid, 0] += hi - lo

    # Deterministic reduction AFTER the parallel region (fixed order of rows)
    hist = np.zeros(num_bins, dtype=np.int64)
    w_sums = np.zeros(num_bins, dtype=np.float64)
    for t in range(max_threads):
        for b in range(num_bins):
            hist[b] += priv_hist[t, b]
    for c in range(N_CHUNKS_T1):
        for b in range(num_bins):
            w_sums[b] += priv_w[c, b]
    return hist, w_sums, worker_hits[:, 0].copy()


print("\n=== TASK 1 ===")
# Single-threaded reference (buggy kernel is correct with 1 thread)
set_num_threads(1)
ref_hist, ref_w = buggy_hist_omp(data_t1, weights_t1, BINS_T1)
assert int(ref_hist.sum()) == N_T1

# 1) Empirical race demonstration: 8 threads, 10 runs
set_num_threads(MAX_T)
buggy_hist_omp(data_t1, weights_t1, BINS_T1)  # warm-up
lost_list, maxdiff_list = [], []
for run in range(10):
    h, w = buggy_hist_omp(data_t1, weights_t1, BINS_T1)
    lost = N_T1 - int(np.sum(h))
    maxdiff = int(np.max(np.abs(h - ref_hist)))
    lost_list.append(lost)
    maxdiff_list.append(maxdiff)
    print(f"  buggy run {run + 1:2d}: lost counts = {lost:>9d} | max |hist - ref| = {maxdiff}")
print(f"  total lost over 10 runs = {sum(lost_list)} | worst discrepancy = {max(maxdiff_list)}")

# 2) Safe kernel on 1, 2, 4, 8 threads
bincount_ref = np.bincount(data_t1, minlength=BINS_T1).astype(np.int64)
wsum_ref = np.bincount(data_t1, weights=weights_t1, minlength=BINS_T1)
safe_results = {}
for nt in THREAD_COUNTS:
    set_num_threads(nt)
    t, (h, w, hits) = bench(safe_hist_omp, data_t1, weights_t1, BINS_T1)
    safe_results[nt] = (h, w)
    assert np.array_equal(h, bincount_ref), f"hist mismatch at {nt} threads"
    assert np.allclose(w, wsum_ref, rtol=1e-9), f"w_sums mismatch at {nt} threads"
    assert int(hits.sum()) == N_T1
    print(f"  safe_hist_omp @ {nt} thr: {t * 1e3:8.2f} ms | worker_hits = {hits.tolist()}")

for nt in THREAD_COUNTS[1:]:
    assert np.array_equal(safe_results[1][0], safe_results[nt][0])
    assert np.array_equal(safe_results[1][1], safe_results[nt][1]), "float sums NOT bit-identical!"
print("  OK: safe_hist_omp is bit-identical for 1, 2, 4, 8 threads")

set_num_threads(MAX_T)
hist_t1, wsums_t1, _hits = safe_hist_omp(data_t1, weights_t1, BINS_T1)

DIAGNOSIS_T1 = (
    "Floating-point addition is not associative, because every addition rounds the exact result "
    "to 53 bits of mantissa, so (a + b) + c can differ from a + (b + c) in the last bits. "
    "Even if the integer increments were atomic and therefore exact, the weights would still be "
    "added to each bin in whatever order the threads happen to reach it. "
    "That order depends on OS scheduling and cache timing, so two runs add the same numbers in "
    "different orders and give slightly different sums. "
    "Bit-exact results therefore need a fixed partition of the data and a fixed merge order, which "
    "safe_hist_omp gets from private buffers per fixed chunk that are merged in ascending order."
)

# =====================================================================
# TASK 2: ELIMINATING CACHE-LINE FALSE SHARING
# =====================================================================
@njit(parallel=True)
def unpadded_state_omp(iters, seed_val):
    n_threads = get_num_threads()
    state = np.zeros(n_threads, dtype=np.int64)
    for t in prange(n_threads):
        tid = get_thread_id()
        x = np.int64(seed_val + tid * 1315423911)
        for step in range(iters):
            x = (x * np.int64(6364136223846793005) + np.int64(1442695040888963407)) & np.int64(0x7FFFFFFFFFFFFFFF)
            state[tid] ^= (x >> 17)  # Writes directly to shared cache line on every iteration!
    return state


@njit(parallel=True)
def padded_state_omp(iters, seed_val, pad_stride):
    n_threads = get_num_threads()
    # pad_stride int64 slots = >= 64 bytes between consecutive workers' active slots
    padded_state = np.zeros((n_threads, pad_stride), dtype=np.int64)
    for t in prange(n_threads):
        tid = get_thread_id()
        x = np.int64(seed_val + tid * 1315423911)
        for step in range(iters):
            x = (x * np.int64(6364136223846793005) + np.int64(1442695040888963407)) & np.int64(0x7FFFFFFFFFFFFFFF)
            padded_state[tid, 0] ^= (x >> 17)  # the store stays INSIDE the loop
    return padded_state


print("\n=== TASK 2 ===")
print(f"  ITERS_T2 = {ITERS_T2:,} | PAD_STRIDE = {PAD_STRIDE} int64 slots = {PAD_STRIDE * 8} bytes")
t2_unpadded, t2_padded = {}, {}
for nt in THREAD_COUNTS:
    set_num_threads(nt)
    tu, su = bench(unpadded_state_omp, ITERS_T2, STUDENT_ID)
    tp, sp = bench(padded_state_omp, ITERS_T2, STUDENT_ID, PAD_STRIDE)
    assert np.array_equal(su, sp[:, 0]), f"unpadded != padded at {nt} threads"
    t2_unpadded[nt], t2_padded[nt] = tu, tp
    print(f"  {nt} thr | unpadded {tu:7.3f} s | padded {tp:7.3f} s | speedup padded/unpadded = {tu / tp:5.2f}x")
print(f"  Speedup ratio at 8 threads (unpadded / padded) = {t2_unpadded[8] / t2_padded[8]:.2f}x")
print("  OK: element-wise equality for all thread counts")

set_num_threads(MAX_T)
state_t2 = unpadded_state_omp(ITERS_T2, STUDENT_ID)

DIAGNOSIS_T2 = (
    "x86_64 and ARM64 cores move memory in 64-byte cache lines, and the MESI protocol lets only one "
    "core hold a line in the Modified state at a time. "
    "In unpadded_state_omp all eight int64 slots of state[] share one line, so every store by one "
    "thread invalidates that line in the other cores' L1 caches (M to I) and forces them to re-fetch it. "
    "The line ping-pongs between cores on every iteration, so the threads are serialized by coherence "
    "traffic although they write to different variables (false sharing). "
    "padded_state_omp puts each worker's slot into its own 64-byte line, so each line stays in the "
    "Modified/Exclusive state of exactly one core and no invalidations occur."
    "In my measurement on an Apple Silicon MacBook the speedup at 8 threads was about 0.97x, i.e. "
    "no visible difference; the compiler most likely kept the accumulator in a register inside the loop, "
    "and Apple cores use 128-byte cache lines, so the expected collapse did not appear."
)

# =====================================================================
# TASK 3: IRREGULAR WORKLOAD SKEW & DYNAMIC CHUNK SCHEDULING
# =====================================================================
work_units_t3 = np.array(
    [40 + ((r * 7919 + STUDENT_ID) % SKEW_MOD) ** 2 for r in range(ROWS_T3)], dtype=np.int64
)
CHUNK_T3 = 16


@njit
def row_score(r, units):
    acc = 0.0
    for k in range(units):
        x = r * 0.001 + k * 0.01
        acc += math.sin(x) * math.cos(x) + math.sqrt(abs(x) + 1.0)
    return acc


@njit(parallel=True)
def static_skew_omp(work_units, pad_stride):
    n = work_units.shape[0]
    out = np.zeros(n, dtype=np.float64)
    ops = np.zeros((get_num_threads(), pad_stride), dtype=np.int64)  # padded counters
    for r in prange(n):  # static: contiguous blocks of rows per thread
        tid = get_thread_id()
        out[r] = row_score(r, work_units[r])
        ops[tid, 0] += work_units[r]
    return out, ops[:, 0].copy()


@njit
def build_dynamic_schedule(work_units, n_threads, chunk):
    """Replays what a dynamic chunk cursor does: chunks are claimed in index order, and the
    next chunk always goes to the worker that is least loaded (finishes first)."""
    n_rows = work_units.shape[0]
    n_chunks = (n_rows + chunk - 1) // chunk
    sched = np.full((n_threads, n_chunks), -1, dtype=np.int64)
    counts = np.zeros(n_threads, dtype=np.int64)
    load = np.zeros(n_threads, dtype=np.int64)
    for c in range(n_chunks):
        lo = c * chunk
        hi = min(lo + chunk, n_rows)
        cost = 0
        for r in range(lo, hi):
            cost += work_units[r]
        t = 0
        for k in range(1, n_threads):
            if load[k] < load[t]:
                t = k
        sched[t, counts[t]] = c
        counts[t] += 1
        load[t] += cost
    return sched, counts


@njit(parallel=True)
def balanced_skew_omp(work_units, pad_stride, chunk):
    n = work_units.shape[0]
    n_threads = get_num_threads()
    sched, counts = build_dynamic_schedule(work_units, n_threads, chunk)
    out = np.zeros(n, dtype=np.float64)
    ops = np.zeros((n_threads, pad_stride), dtype=np.int64)
    for w in prange(n_threads):  # one OpenMP worker per schedule row
        tid = get_thread_id()
        for j in range(counts[w]):
            c = sched[w, j]
            lo = c * chunk
            hi = min(lo + chunk, n)
            for r in range(lo, hi):
                out[r] = row_score(r, work_units[r])
                ops[tid, 0] += work_units[r]
    return out, ops[:, 0].copy()


def imbalance(ops):
    active = ops[ops > 0]
    return float(active.max()) / float(active.min())


print("\n=== TASK 3 ===")
print(f"  ROWS_T3 = {ROWS_T3} | SKEW_MOD = {SKEW_MOD} | total work = {int(work_units_t3.sum()):,}")
for nt in THREAD_COUNTS:
    set_num_threads(nt)
    ts, (o_s, ops_s) = bench(static_skew_omp, work_units_t3, PAD_STRIDE)
    tb, (o_b, ops_b) = bench(balanced_skew_omp, work_units_t3, PAD_STRIDE, CHUNK_T3)
    diff = float(np.max(np.abs(o_s - o_b)))
    assert diff < 1e-11, f"static vs balanced mismatch at {nt} threads: {diff}"
    assert int(ops_s.sum()) == int(ops_b.sum()) == int(work_units_t3.sum())
    print(f"  {nt} thr | static {ts * 1e3:8.1f} ms (imb {imbalance(ops_s):.4f}) | "
          f"balanced {tb * 1e3:8.1f} ms (imb {imbalance(ops_b):.4f}) | max diff {diff:.2e}")

set_num_threads(MAX_T)
out_static, ops_static = static_skew_omp(work_units_t3, PAD_STRIDE)
out_t3, ops_balanced = balanced_skew_omp(work_units_t3, PAD_STRIDE, CHUNK_T3)
lir_static, lir_balanced = imbalance(ops_static), imbalance(ops_balanced)
print(f"  Load Imbalance @8 threads: static = {lir_static:.4f}x | balanced = {lir_balanced:.4f}x")
print(f"  ops/thread static   = {ops_static.tolist()}")
print(f"  ops/thread balanced = {ops_balanced.tolist()}")
assert np.max(np.abs(out_static - out_t3)) < 1e-11
assert lir_balanced < 1.15, "Load imbalance must be strictly below 1.15x"

DIAGNOSIS_T3 = (
    "Numba's default prange splits the row range into one contiguous block per thread, so a block "
    "that happens to contain many expensive rows (work grows with the square of the skew factor) "
    "makes its thread a straggler while the others wait at the implicit barrier. "
    "balanced_skew_omp instead splits the rows into small chunks of 16 and lets each worker claim "
    "the next chunk in index order whenever it becomes free, the same rule as a dynamic chunk cursor. "
    "Numba has no OpenMP atomic, so this claim order is replayed deterministically in "
    "build_dynamic_schedule using the known per-row cost, and the OpenMP workers then run their own "
    "chunk lists lock-free in a prange over workers. "
    "Every row is still computed sequentially by one worker, so the output is identical to the "
    "static version while the per-thread operation counts differ by far less than 15 percent."
)

# =====================================================================
# TASK 4: PARALLEL RED-BLACK SOR PDE SOLVER
# =====================================================================
def make_grid():
    U = np.zeros((GRID_H, GRID_W), dtype=np.float64)
    U[0, :] = 100.0
    U[-1, :] = -50.0
    U[:, 0] = 25.0 * np.sin(np.linspace(0, np.pi, GRID_H))
    U[:, -1] = 75.0 * np.cos(np.linspace(0, np.pi / 2, GRID_H))
    return U


@njit(parallel=True)
def sor_red_black_step(U, omega, color, delta_pad):
    """color=0: Red cells ((r+c)%2==0); color=1: Black cells ((r+c)%2==1).
    All 4 neighbours of a cell have the opposite color and are read-only in this sweep."""
    H, W = U.shape
    for r in prange(1, H - 1):
        tid = get_thread_id()
        c0 = 1 if ((r + 1) % 2) == color else 2  # first interior column with (r+c)%2 == color
        for c in range(c0, W - 1, 2):
            old = U[r, c]
            new = (1.0 - omega) * old + 0.25 * omega * (U[r - 1, c] + U[r + 1, c] + U[r, c - 1] + U[r, c + 1])
            U[r, c] = new
            d = abs(new - old)
            if d > delta_pad[tid, 0]:  # padded per-thread max -> no race
                delta_pad[tid, 0] = d


@njit
def solve_pde_omp(U, omega, sweeps, pad_stride):
    n_threads = get_num_threads()
    delta_pad = np.zeros((n_threads, pad_stride), dtype=np.float64)
    max_delta = 0.0
    for s in range(sweeps):
        delta_pad[:, :] = 0.0
        sor_red_black_step(U, omega, 0, delta_pad)  # Red sweep (barrier at the end of prange)
        sor_red_black_step(U, omega, 1, delta_pad)  # Black sweep sees the new Red values
        max_delta = 0.0
        for t in range(n_threads):
            if delta_pad[t, 0] > max_delta:
                max_delta = delta_pad[t, 0]
    return max_delta


print("\n=== TASK 4 ===")
SWEEPS = 600
print(f"  Grid {GRID_H}x{GRID_W} | omega = {float(OMEGA_SOR):.3f} | sweeps = {SWEEPS}")
t4_times, t4_grids, t4_deltas = {}, {}, {}
for nt in THREAD_COUNTS:
    set_num_threads(nt)
    solve_pde_omp(make_grid(), OMEGA_SOR, 2, PAD_STRIDE)  # warm-up
    best = float("inf")
    for _ in range(3):
        U = make_grid()
        t0 = time.perf_counter()
        md = solve_pde_omp(U, OMEGA_SOR, SWEEPS, PAD_STRIDE)
        best = min(best, time.perf_counter() - t0)
    t4_times[nt], t4_grids[nt], t4_deltas[nt] = best, U, md

for nt in THREAD_COUNTS:
    diff = float(np.max(np.abs(t4_grids[nt] - t4_grids[1])))
    assert diff < 1e-9, f"grid differs from 1-thread run at {nt} threads: {diff}"
    print(f"  {nt} thr | {t4_times[nt]:7.3f} s | speedup {t4_times[1] / t4_times[nt]:5.2f}x | "
          f"max_delta(last) {t4_deltas[nt]:.3e} | max_abs_diff vs 1 thr {diff:.2e}")

grid_t4 = t4_grids[MAX_T]

# =====================================================================
# FINAL SECTION: DETERMINISTIC VERIFICATION & TELEMETRY BUNDLE
# =====================================================================
set_num_threads(8 if (os.cpu_count() or 1) >= 8 else os.cpu_count())
for idx, diag in enumerate([DIAGNOSIS_T1, DIAGNOSIS_T2, DIAGNOSIS_T3], start=1):
    assert isinstance(diag, str) and len(diag.strip()) >= 80, f"Complete DIAGNOSIS_T{idx}!"

checksum_blob = (
    f"{STUDENT_ID}|"
    f"{int(np.sum(hist_t1))}|{float(np.sum(wsums_t1)):.4f}|"
    f"{int(np.bitwise_xor.reduce(state_t2))}|"
    f"{float(np.sum(out_t3)):.6f}|"
    f"{float(np.sum(grid_t4)):.6f}"
)
lab_digest = hashlib.sha256(checksum_blob.encode()).hexdigest()[:20].upper()

bundle = {
    "instructor": "Instructor - Sufyan bin Uzayr",
    "student_name": STUDENT_NAME,
    "student_id": STUDENT_ID,
    "omp_layer": nb.threading_layer(),
    "cpu_cores": os.cpu_count(),
    "verification_digest": lab_digest,
}
print("\n=== OPENMP PYTHON LAB VERIFICATION BUNDLE ===")
print(json.dumps(bundle, indent=2))