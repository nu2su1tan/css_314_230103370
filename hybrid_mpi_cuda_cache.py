import os, sys, json, math, time, hashlib, platform, subprocess
import numpy as np
from mpi4py import MPI
from numba import cuda, njit, float64, uint64

TILE_DIM = 16
SHARED_DIM = TILE_DIM + 2      
GLOBAL_NX = 512
GLOBAL_NY = 512
LOCAL_N = 256                  
PAD_N = LOCAL_N + 2            
STENCIL_STEPS = 300
SHM_ITERS = 40_000_000
LAMBDA_DAMP = 0.15
DX = 1.0 / 513


def get_gpu_uuid():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], encoding="utf-8")
    return out.strip().splitlines()[0].strip()


def detect_cpu_line_size():
    p = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
    if os.path.exists(p):
        with open(p) as f:
            return int(f.read().strip())
    return 64





def init_hybrid_context(github_user):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size != 4:
        raise RuntimeError(f"Need exactly 4 ranks, got {size}")

    cuda.select_device(0)
    gpu_uuid = comm.bcast(get_gpu_uuid() if rank == 0 else None, root=0)
    cpu_sig = f"{platform.machine()}::{os.cpu_count()}"
    line_bytes = detect_cpu_line_size()

    h = hashlib.sha256(f"{github_user.strip()}::{gpu_uuid}::{cpu_sig}".encode()).hexdigest()
    seed = int(h[:16], 16)   

    
    cart = comm.Create_cart(dims=[2, 2], periods=[False, False], reorder=False)
    coords = cart.Get_coords(cart.Get_rank())
    up, down = cart.Shift(0, 1)     
    left, right = cart.Shift(1, 1)  

    
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    n_slots = 64
    extra = 8  
    nbytes = (n_slots + extra) * 8 if node_comm.Get_rank() == 0 else 0
    win = MPI.Win.Allocate_shared(nbytes, 8, comm=node_comm)
    buf, _ = win.Shared_query(0)
    raw = np.ndarray(buffer=buf, dtype=np.uint64, shape=(n_slots + extra,))
    addr = raw.ctypes.data
    off = ((64 - addr % 64) % 64) // 8
    shm = raw[off:off + n_slots]
    assert shm.ctypes.data % 64 == 0, "shm not 64B aligned"
    offs = node_comm.allgather(off)
    assert len(set(offs)) == 1, "ranks see different alignment offsets"

    return dict(comm=comm, cart=cart, rank=rank, coords=coords,
                up=up, down=down, left=left, right=right,
                node_comm=node_comm, win=win, shm=shm,
                seed=seed, gpu_uuid=gpu_uuid, cpu_sig=cpu_sig,
                line_bytes=line_bytes, github_user=github_user.strip())





def initialize_fields(ctx):
    cy, cx = ctx["coords"]
    ii = np.arange(PAD_N)
    
    gy = cy * LOCAL_N + ii - 1
    gx = cx * LOCAL_N + ii - 1
    y = (gy + 1) * DX
    x = (gx + 1) * DX
    X, Y = np.meshgrid(x, y)       

    u = np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.01)
    kap = 1.0 + 0.5 * np.sin(np.pi * X) * np.sin(np.pi * Y)   
    src = 0.5 * np.exp(-((X - 0.25) ** 2 + (Y - 0.75) ** 2) / 0.005)

    
    rng = np.random.default_rng((ctx["seed"] + ctx["rank"]) % (2 ** 63))
    u = u + 1e-9 * rng.standard_normal(u.shape)

    return (np.ascontiguousarray(u, dtype=np.float64),
            np.ascontiguousarray(kap, dtype=np.float64),
            np.ascontiguousarray(src, dtype=np.float64))


@cuda.jit
def pack_halos_kernel(u, sendbuf):
    
    t = cuda.grid(1)
    if t >= 4 * LOCAL_N:
        return
    which = t // LOCAL_N
    k = t % LOCAL_N + 1          
    if which == 0:
        sendbuf[t] = u[1, k]            
    elif which == 1:
        sendbuf[t] = u[LOCAL_N, k]      
    elif which == 2:
        sendbuf[t] = u[k, 1]            
    else:
        sendbuf[t] = u[k, LOCAL_N]      


@cuda.jit
def unpack_halos_kernel(u, recvbuf):
    
    t = cuda.grid(1)
    if t >= 4 * LOCAL_N:
        return
    which = t // LOCAL_N
    k = t % LOCAL_N + 1
    if which == 0:
        u[0, k] = recvbuf[t]
    elif which == 1:
        u[LOCAL_N + 1, k] = recvbuf[t]
    elif which == 2:
        u[k, 0] = recvbuf[t]
    else:
        u[k, LOCAL_N + 1] = recvbuf[t]





@cuda.jit
def tiled_stencil_kernel(u, unew, kap, src, coef, lam, dt, r0, r1, c0, c1):
    tile = cuda.shared.array((SHARED_DIM, SHARED_DIM), float64)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    j = c0 + cuda.blockIdx.x * TILE_DIM + tx
    i = r0 + cuda.blockIdx.y * TILE_DIM + ty
    n = u.shape[0]
    m = u.shape[1]

    
    if i < n and j < m:
        tile[ty + 1, tx + 1] = u[i, j]
    else:
        tile[ty + 1, tx + 1] = 0.0
    
    if ty == 0:
        if i - 1 >= 0 and i - 1 < n and j < m:
            tile[0, tx + 1] = u[i - 1, j]
        else:
            tile[0, tx + 1] = 0.0
    if ty == TILE_DIM - 1:
        if i + 1 < n and j < m:
            tile[TILE_DIM + 1, tx + 1] = u[i + 1, j]
        else:
            tile[TILE_DIM + 1, tx + 1] = 0.0
    if tx == 0:
        if j - 1 >= 0 and j - 1 < m and i < n:
            tile[ty + 1, 0] = u[i, j - 1]
        else:
            tile[ty + 1, 0] = 0.0
    if tx == TILE_DIM - 1:
        if j + 1 < m and i < n:
            tile[ty + 1, TILE_DIM + 1] = u[i, j + 1]
        else:
            tile[ty + 1, TILE_DIM + 1] = 0.0

    cuda.syncthreads()

    if i >= r0 and i < r1 and j >= c0 and j < c1:
        c = tile[ty + 1, tx + 1]
        k0 = kap[i, j]
        ke = 0.5 * (k0 + kap[i, j + 1])
        kw = 0.5 * (k0 + kap[i, j - 1])
        kn = 0.5 * (k0 + kap[i - 1, j])
        ks = 0.5 * (k0 + kap[i + 1, j])
        lap = (ke * (tile[ty + 1, tx + 2] - c) - kw * (c - tile[ty + 1, tx])
               + ks * (tile[ty + 2, tx + 1] - c) - kn * (c - tile[ty, tx + 1]))
        unew[i, j] = c + dt * (coef * lap - lam * c + src[i, j])


def launch_region(d_u, d_un, d_k, d_s, coef, dt, r0, r1, c0, c1, stream):
    
    gx = (c1 - c0 + TILE_DIM - 1) // TILE_DIM
    gy = (r1 - r0 + TILE_DIM - 1) // TILE_DIM
    tiled_stencil_kernel[(gx, gy), (TILE_DIM, TILE_DIM), stream](
        d_u, d_un, d_k, d_s, coef, LAMBDA_DAMP, dt, r0, r1, c0, c1)


def post_halo_mpi(ctx, sbuf, rbuf):
    cart = ctx["cart"]
    reqs = []
    N = LOCAL_N
    
    reqs.append(cart.Irecv(rbuf[0:N],       source=ctx["up"],    tag=1))
    reqs.append(cart.Irecv(rbuf[N:2*N],     source=ctx["down"],  tag=0))
    reqs.append(cart.Irecv(rbuf[2*N:3*N],   source=ctx["left"],  tag=3))
    reqs.append(cart.Irecv(rbuf[3*N:4*N],   source=ctx["right"], tag=2))
    reqs.append(cart.Isend(sbuf[0:N],       dest=ctx["up"],      tag=0))
    reqs.append(cart.Isend(sbuf[N:2*N],     dest=ctx["down"],    tag=1))
    reqs.append(cart.Isend(sbuf[2*N:3*N],   dest=ctx["left"],    tag=2))
    reqs.append(cart.Isend(sbuf[3*N:4*N],   dest=ctx["right"],   tag=3))
    return reqs


class GpuState:
    def __init__(self, u, kap, src):
        self.d_u = cuda.to_device(u)
        self.d_un = cuda.to_device(np.zeros_like(u))
        self.d_k = cuda.to_device(kap)
        self.d_s = cuda.to_device(src)
        self.d_send = cuda.device_array(4 * LOCAL_N, dtype=np.float64)
        self.d_recv = cuda.device_array(4 * LOCAL_N, dtype=np.float64)
        self.sbuf = cuda.pinned_array(4 * LOCAL_N, dtype=np.float64)
        self.rbuf = cuda.pinned_array(4 * LOCAL_N, dtype=np.float64)
        self.rbuf[:] = 0.0
        self.s_int = cuda.stream()
        self.s_bnd = cuda.stream()

    def swap(self):
        self.d_u, self.d_un = self.d_un, self.d_u


HALO_BLOCKS = (4 * LOCAL_N + 127) // 128


def step_serialized(ctx, st, coef, dt):
    pack_halos_kernel[HALO_BLOCKS, 128](st.d_u, st.d_send)
    cuda.synchronize()
    st.d_send.copy_to_host(st.sbuf)
    reqs = post_halo_mpi(ctx, st.sbuf, st.rbuf)
    MPI.Request.Waitall(reqs)
    st.d_recv.copy_to_device(st.rbuf)
    unpack_halos_kernel[HALO_BLOCKS, 128](st.d_u, st.d_recv)
    
    launch_region(st.d_u, st.d_un, st.d_k, st.d_s, coef, dt, 1, LOCAL_N + 1, 1, LOCAL_N + 1, 0)
    cuda.synchronize()
    st.swap()


def step_overlapped(ctx, st, coef, dt):
    
    launch_region(st.d_u, st.d_un, st.d_k, st.d_s, coef, dt, 2, LOCAL_N, 2, LOCAL_N, st.s_int)

    
    pack_halos_kernel[HALO_BLOCKS, 128, st.s_bnd](st.d_u, st.d_send)
    st.d_send.copy_to_host(st.sbuf, stream=st.s_bnd)
    st.s_bnd.synchronize()

    reqs = post_halo_mpi(ctx, st.sbuf, st.rbuf)
    MPI.Request.Waitall(reqs)

    
    st.d_recv.copy_to_device(st.rbuf, stream=st.s_bnd)
    unpack_halos_kernel[HALO_BLOCKS, 128, st.s_bnd](st.d_u, st.d_recv)
    L = LOCAL_N
    launch_region(st.d_u, st.d_un, st.d_k, st.d_s, coef, dt, 1, 2, 1, L + 1, st.s_bnd)       
    launch_region(st.d_u, st.d_un, st.d_k, st.d_s, coef, dt, L, L + 1, 1, L + 1, st.s_bnd)   
    launch_region(st.d_u, st.d_un, st.d_k, st.d_s, coef, dt, 2, L, 1, 2, st.s_bnd)           
    launch_region(st.d_u, st.d_un, st.d_k, st.d_s, coef, dt, 2, L, L, L + 1, st.s_bnd)       

    st.s_bnd.synchronize()
    st.s_int.synchronize()
    st.swap()


def local_energy(st):
    u = st.d_u.copy_to_host()
    inner = u[1:LOCAL_N + 1, 1:LOCAL_N + 1]
    return float(np.sum(inner * inner) * DX * DX)





@njit(nogil=True)
def host_shm_accumulate(shm, idx_arr, iters):
    
    
    for _ in range(iters):
        k = idx_arr[0]
        shm[k] += 1


def benchmark_shm_strides(ctx, iters=SHM_ITERS):
    node_comm = ctx["node_comm"]
    shm = ctx["shm"]
    nr = node_comm.Get_rank()
    out = {}

    
    warm = np.array([nr], dtype=np.int64)
    host_shm_accumulate(shm, warm, 1000)
    node_comm.Barrier()

    for stride in (1, 8, 16):
        idx = np.array([nr * stride], dtype=np.int64)
        shm[:] = 0
        node_comm.Barrier()
        t0 = MPI.Wtime()
        host_shm_accumulate(shm, idx, iters)
        node_comm.Barrier()
        t1 = MPI.Wtime()
        mine = (t1 - t0) * 1000.0
        worst = node_comm.allreduce(mine, op=MPI.MAX)
        out[stride] = worst
        
        assert int(shm[nr * stride]) == iters
        node_comm.Barrier()
    return out


def audit_ptx():
    asm = tiled_stencil_kernel.inspect_asm()
    ptx = list(asm.values())[0] if isinstance(asm, dict) else asm
    return {
        "st.shared.f64": ptx.count("st.shared.f64"),
        "ld.shared.f64": ptx.count("ld.shared.f64"),
        "bar.sync": ptx.count("bar.sync"),
    }


def audit_llvm():
    ir = host_shm_accumulate.inspect_llvm()
    ir = list(ir.values())[0] if isinstance(ir, dict) else ir
    
    loads = sum(1 for l in ir.splitlines() if " load " in l)
    stores = sum(1 for l in ir.splitlines() if " store " in l)
    return {"load": loads, "store": stores}





def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: hybrid_mpi_cuda_cache.py <github_username>")
    ctx = init_hybrid_context(sys.argv[1])
    comm, cart, rank = ctx["comm"], ctx["cart"], ctx["rank"]

    u0, kap, src = initialize_fields(ctx)
    kappa_max = cart.allreduce(float(kap.max()), op=MPI.MAX)
    dt_max = DX * DX / (4.0 * kappa_max)
    dt = 0.8 * dt_max
    coef = 1.0 / (DX * DX)

    if rank == 0:
        print(f"[ctx] seed={ctx['seed']} gpu={ctx['gpu_uuid']} cpu={ctx['cpu_sig']} line={ctx['line_bytes']}B")
        print(f"[ctx] kappa_max={kappa_max:.6f} dt={dt:.6e} dt_max={dt_max:.6e}")

    
    st_a = GpuState(u0, kap, src)
    
    warm = GpuState(u0, kap, src)
    step_serialized(ctx, warm, coef, dt)
    step_overlapped(ctx, warm, coef, dt)
    cuda.synchronize()
    comm.Barrier()

    t0 = time.perf_counter()
    for _ in range(STENCIL_STEPS):
        step_serialized(ctx, st_a, coef, dt)
    cuda.synchronize()
    comm.Barrier()
    t_ser = (time.perf_counter() - t0) * 1000.0
    ser_field = st_a.d_u.copy_to_host()

    
    st_b = GpuState(u0, kap, src)
    energies = {}
    energies[0] = cart.allreduce(local_energy(st_b), op=MPI.SUM)
    comm.Barrier()
    t0 = time.perf_counter()
    for s in range(1, STENCIL_STEPS + 1):
        step_overlapped(ctx, st_b, coef, dt)
        if s % 100 == 0:
            energies[s] = cart.allreduce(local_energy(st_b), op=MPI.SUM)  
    cuda.synchronize()
    comm.Barrier()
    t_ovl = (time.perf_counter() - t0) * 1000.0
    ovl_field = st_b.d_u.copy_to_host()

    max_diff_local = float(np.max(np.abs(ser_field - ovl_field)))
    max_diff = cart.allreduce(max_diff_local, op=MPI.MAX)
    t_ser = cart.allreduce(t_ser, op=MPI.MAX)
    t_ovl = cart.allreduce(t_ovl, op=MPI.MAX)

    if rank == 0:
        print(f"[stencil] gpu_serialized_ms={t_ser:.3f} gpu_overlapped_ms={t_ovl:.3f} ratio={t_ser / t_ovl:.4f}")
        print(f"[stencil] max_diff across ranks = {max_diff:.3e}")
        for s in sorted(energies):
            print(f"[energy] step {s:3d}: E = {energies[s]:.12e}")
    assert max_diff < 1e-12, f"parity failed {max_diff}"
    assert all(math.isfinite(v) for v in energies.values()), "energy blew up"

    
    comm.Barrier()
    shm_ms = benchmark_shm_strides(ctx)
    if rank == 0:
        for s in (1, 8, 16):
            print(f"[shm] stride={s:2d} ({s * 8:3d} B): {shm_ms[s]:.2f} ms")
        print(f"[shm] false sharing slowdown (stride1/stride8) = {shm_ms[1] / shm_ms[8]:.2f}x")
        print(f"[shm] stride8/stride16 = {shm_ms[8] / shm_ms[16]:.3f}")

    
    ptx = audit_ptx()
    llvm = audit_llvm()
    if rank == 0:
        print("[ptx ]", ptx)
        print("[llvm]", llvm)
        assert ptx["st.shared.f64"] > 0 and ptx["ld.shared.f64"] > 0 and ptx["bar.sync"] > 0

        manifest = {
            "github_username": ctx["github_user"],
            "instance_seed": ctx["seed"],
            "gpu_uuid": ctx["gpu_uuid"],
            "cpu_signature": ctx["cpu_sig"],
            "cpu_line_bytes": ctx["line_bytes"],
            "kappa_max": kappa_max,
            "dx": DX,
            "dt": dt,
            "dt_max": dt_max,
            "gpu_serialized_ms": t_ser,
            "gpu_overlapped_ms": t_ovl,
            "max_diff": max_diff,
            "energy": {str(k): v for k, v in energies.items()},
            "shm_ms": {str(k): v for k, v in shm_ms.items()},
            "false_sharing_ratio": shm_ms[1] / shm_ms[8],
            "ptx_audit": ptx,
            "llvm_audit": llvm,
        }
        blob = json.dumps(manifest, sort_keys=True).encode()
        manifest["manifest_sha256"] = hashlib.sha256(blob).hexdigest()
        with open("hybrid_lab_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print("manifest written")

    ctx["win"].Free()


if __name__ == "__main__":
    main()
