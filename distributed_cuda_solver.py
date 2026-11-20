import os
import sys
import json
import math
import hashlib
import subprocess
import numpy as np
from mpi4py import MPI
from numba import cuda, float64




TILE_DIM = 16
SHARED_DIM = TILE_DIM + 2          
GLOBAL_NX = 512
GLOBAL_NY = 512
STEPS = 400
LAMBDA_DAMP = 0.15
NY = 256                           
NX = 256                           
PAD = 258                          
HALO_LEN = 256                     
HALO_TOTAL = 4 * HALO_LEN          
DX = 1.0 / (GLOBAL_NX + 1)         
DY = 1.0 / (GLOBAL_NY + 1)


def get_gpu_uuid() -> str:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        encoding="utf-8",
    )
    return out.strip().splitlines()[0].strip()


def derive_student_seed(github_user: str, gpu_uuid: str) -> int:
    digest = hashlib.sha256(f"{github_user.strip()}::{gpu_uuid}".encode("utf8")).hexdigest()
    return int(digest[:16], 16)





def init_distributed_context(github_user: str):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    if size != 4:
        raise RuntimeError(f"Expected exactly 4 MPI ranks (2x2 grid), got {size}.")

    cart = comm.Create_cart(dims=[2, 2], periods=[False, False], reorder=False)
    rank = cart.Get_rank()
    row_rank, col_rank = cart.Get_coords(rank)

    
    
    nbr_up, nbr_down = cart.Shift(0, 1)
    nbr_left, nbr_right = cart.Shift(1, 1)

    cuda.select_device(0)

    gpu_uuid = get_gpu_uuid() if rank == 0 else None
    gpu_uuid = cart.bcast(gpu_uuid, root=0)
    instance_seed = derive_student_seed(github_user, gpu_uuid)

    return {
        "comm": comm, "cart": cart, "rank": rank,
        "row_rank": row_rank, "col_rank": col_rank,
        "nbr_up": nbr_up, "nbr_down": nbr_down,
        "nbr_left": nbr_left, "nbr_right": nbr_right,
        "gpu_uuid": gpu_uuid, "seed": instance_seed, "github_user": github_user,
    }





def initialize_subdomains(ctx):
    rr, cc = ctx["row_rank"], ctx["col_rank"]
    rng = np.random.default_rng(ctx["seed"])      
    phase = rng.uniform(0.0, 2.0 * np.pi, size=2)
    n_pulses = 3
    centers = rng.uniform(0.2, 0.8, size=(n_pulses, 2))   
    amps = rng.uniform(0.6, 1.4, size=n_pulses)
    sigma = 0.07

    idx = np.arange(PAD, dtype=np.float64)
    
    gy = (rr * NY + idx) / (GLOBAL_NY + 1)        
    gx = (cc * NX + idx) / (GLOBAL_NX + 1)        
    X, Y = np.meshgrid(gx, gy)                    

    kappa = 0.16 + 0.08 * np.sin(2 * np.pi * X + phase[0]) * np.cos(2 * np.pi * Y + phase[1])

    u = np.zeros((PAD, PAD), dtype=np.float64)
    for k in range(n_pulses):
        u += amps[k] * np.exp(-((X - centers[k, 0]) ** 2 + (Y - centers[k, 1]) ** 2)
                              / (2.0 * sigma ** 2))

    S = 0.02 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / (2.0 * 0.05 ** 2))

    
    if ctx["nbr_up"] == MPI.PROC_NULL:
        u[0, :] = 0.0
    if ctx["nbr_down"] == MPI.PROC_NULL:
        u[PAD - 1, :] = 0.0
    if ctx["nbr_left"] == MPI.PROC_NULL:
        u[:, 0] = 0.0
    if ctx["nbr_right"] == MPI.PROC_NULL:
        u[:, PAD - 1] = 0.0

    return np.ascontiguousarray(u), np.ascontiguousarray(kappa), np.ascontiguousarray(S)








@cuda.jit
def pack_halos_kernel(u, send_buf):
    t = cuda.grid(1)
    if t < 4 * 256:
        seg = t // 256
        k = t - seg * 256
        if seg == 0:
            send_buf[t] = u[1, 1 + k]
        elif seg == 1:
            send_buf[t] = u[256, 1 + k]
        elif seg == 2:
            send_buf[t] = u[1 + k, 1]
        else:
            send_buf[t] = u[1 + k, 256]


@cuda.jit
def unpack_halos_kernel(u, recv_buf):
    t = cuda.grid(1)
    if t < 4 * 256:
        seg = t // 256
        k = t - seg * 256
        if seg == 0:
            u[0, 1 + k] = recv_buf[t]
        elif seg == 1:
            u[257, 1 + k] = recv_buf[t]
        elif seg == 2:
            u[1 + k, 0] = recv_buf[t]
        else:
            u[1 + k, 257] = recv_buf[t]


def make_state(ctx, fields):
    u0, k0, s0 = fields
    st = {
        "u": cuda.to_device(u0),
        "un": cuda.to_device(u0.copy()),
        "kappa": cuda.to_device(k0),
        "S": cuda.to_device(s0),
        "d_send": cuda.device_array(HALO_TOTAL, dtype=np.float64),
        "d_recv": cuda.device_array(HALO_TOTAL, dtype=np.float64),
        "h_send": cuda.pinned_array(HALO_TOTAL, dtype=np.float64),
        "h_recv": cuda.pinned_array(HALO_TOTAL, dtype=np.float64),
        "sb": cuda.stream(),     
        "si": cuda.stream(),     
    }
    st["h_send"][:] = 0.0
    st["h_recv"][:] = 0.0      
    st["d_recv"].copy_to_device(st["h_recv"])
    return st


def stage_a_pack_and_d2h(st):
    """Stage A: pack on stream_boundary, async D2H into pinned buffer."""
    pack_halos_kernel[(HALO_TOTAL + 255) // 256, 256, st["sb"]](st["u"], st["d_send"])
    st["d_send"].copy_to_host(st["h_send"], stream=st["sb"])


def post_mpi_exchange(ctx, st):
    """Post non-blocking Irecv/Isend for all four neighbours (tag = travel direction)."""
    cart = ctx["cart"]
    hs, hr = st["h_send"], st["h_recv"]
    L = HALO_LEN
    s_up, s_dn, s_lf, s_rt = hs[0:L], hs[L:2 * L], hs[2 * L:3 * L], hs[3 * L:4 * L]
    r_up, r_dn, r_lf, r_rt = hr[0:L], hr[L:2 * L], hr[2 * L:3 * L], hr[3 * L:4 * L]
    reqs = [
        
        cart.Irecv(r_dn, source=ctx["nbr_down"], tag=0),
        cart.Isend(s_up, dest=ctx["nbr_up"], tag=0),
        
        cart.Irecv(r_up, source=ctx["nbr_up"], tag=1),
        cart.Isend(s_dn, dest=ctx["nbr_down"], tag=1),
        
        cart.Irecv(r_rt, source=ctx["nbr_right"], tag=2),
        cart.Isend(s_lf, dest=ctx["nbr_left"], tag=2),
        
        cart.Irecv(r_lf, source=ctx["nbr_left"], tag=3),
        cart.Isend(s_rt, dest=ctx["nbr_right"], tag=3),
    ]
    return reqs


def stage_c_h2d_and_unpack(st):
    st["d_recv"].copy_to_device(st["h_recv"], stream=st["sb"])
    unpack_halos_kernel[(HALO_TOTAL + 255) // 256, 256, st["sb"]](st["u"], st["d_recv"])


def exchange_halos_async(ctx, st, overlap_callback=None):
    """Full exchange: A (pack+D2H) -> sync -> B (Isend/Irecv, optional callback
    launching interior compute) -> Waitall -> C (H2D + unpack)."""
    stage_a_pack_and_d2h(st)
    st["sb"].synchronize()
    reqs = post_mpi_exchange(ctx, st)
    if overlap_callback is not None:
        overlap_callback()
    MPI.Request.Waitall(reqs)
    stage_c_h2d_and_unpack(st)






@cuda.jit
def tiled_stencil_kernel(u, kappa, S, u_new, r_start, r_end, c_start, c_end,
                         dt, inv_h2, lam):
    smem = cuda.shared.array((SHARED_DIM, SHARED_DIM), dtype=float64)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    r0 = r_start + cuda.blockIdx.y * TILE_DIM
    c0 = c_start + cuda.blockIdx.x * TILE_DIM
    gr = r0 + ty
    gc = c0 + tx

    
    if gr <= r_end and gc <= c_end:
        smem[ty + 1, tx + 1] = u[gr, gc]
    
    if ty == 0 and gc <= c_end:
        smem[0, tx + 1] = u[r0 - 1, gc]
    if ty == TILE_DIM - 1 and gc <= c_end and r0 + TILE_DIM <= r_end:
        smem[TILE_DIM + 1, tx + 1] = u[r0 + TILE_DIM, gc]
    if tx == 0 and gr <= r_end:
        smem[ty + 1, 0] = u[gr, c0 - 1]
    if tx == TILE_DIM - 1 and gr <= r_end and c0 + TILE_DIM <= c_end:
        smem[ty + 1, TILE_DIM + 1] = u[gr, c0 + TILE_DIM]

    cuda.syncthreads()

    if gr < r_end and gc < c_end:
        uc = smem[ty + 1, tx + 1]
        un = smem[ty, tx + 1]
        us = smem[ty + 2, tx + 1]
        uw = smem[ty + 1, tx]
        ue = smem[ty + 1, tx + 2]
        kc = kappa[gr, gc]
        k_n = 0.5 * (kappa[gr - 1, gc] + kc)
        k_s = 0.5 * (kappa[gr + 1, gc] + kc)
        k_w = 0.5 * (kappa[gr, gc - 1] + kc)
        k_e = 0.5 * (kappa[gr, gc + 1] + kc)
        diff = ((k_s * (us - uc) - k_n * (uc - un)) +
                (k_e * (ue - uc) - k_w * (uc - uw))) * inv_h2
        u_new[gr, gc] = uc + dt * (diff - lam * uc * uc * uc + S[gr, gc])


def launch_stencil(ctx, st, box, stream):
    r0, r1, c0, c1 = box
    gy = (r1 - r0 + TILE_DIM - 1) // TILE_DIM
    gx = (c1 - c0 + TILE_DIM - 1) // TILE_DIM
    if gy <= 0 or gx <= 0:
        return
    tiled_stencil_kernel[(gx, gy), (TILE_DIM, TILE_DIM), stream](
        st["u"], st["kappa"], st["S"], st["un"],
        r0, r1, c0, c1, ctx["dt"], ctx["inv_h2"], LAMBDA_DAMP)


INTERIOR_BOX = (2, NY, 2, NX)                     
RING_BOXES = [(1, 2, 1, NX + 1),                  
              (NY, NY + 1, 1, NX + 1),            
              (2, NY, 1, 2),                      
              (2, NY, NX, NX + 1)]                
FULL_BOX = (1, NY + 1, 1, NX + 1)


def step_serialized(ctx, st):
    exchange_halos_async(ctx, st, overlap_callback=None)
    launch_stencil(ctx, st, FULL_BOX, st["sb"])
    st["sb"].synchronize()


def step_overlapped(ctx, st):
    stage_a_pack_and_d2h(st)                       
    st["sb"].synchronize()
    reqs = post_mpi_exchange(ctx, st)              
    launch_stencil(ctx, st, INTERIOR_BOX, st["si"])  
    MPI.Request.Waitall(reqs)                      
    stage_c_h2d_and_unpack(st)
    for box in RING_BOXES:
        launch_stencil(ctx, st, box, st["sb"])
    st["si"].synchronize()
    st["sb"].synchronize()





def global_diagnostics(ctx, old_h, new_h):
    cart = ctx["cart"]
    e_loc = np.array([new_h[1:NY + 1, 1:NX + 1].sum() * DX * DY])
    d = (new_h[1:NY + 1, 1:NX + 1] - old_h[1:NY + 1, 1:NX + 1])
    r_loc = np.array([(d * d).sum() * DX * DY])
    e_glob = np.zeros(1)
    r_glob = np.zeros(1)
    cart.Allreduce(e_loc, e_glob, op=MPI.SUM)
    cart.Allreduce(r_loc, r_glob, op=MPI.SUM)
    return float(e_glob[0]), float(math.sqrt(r_glob[0]))


def run_mode(ctx, fields, mode):
    cart = ctx["cart"]
    st = make_state(ctx, fields)
    step = step_serialized if mode == "serialized" else step_overlapped

    e0 = np.zeros(1)
    cart.Allreduce(np.array([fields[0][1:NY + 1, 1:NX + 1].sum() * DX * DY]), e0, op=MPI.SUM)
    energy_trace = {0: float(e0[0])}
    residual_trace = {}
    wall = 0.0
    ev_ms = 0.0
    ev_a, ev_b = cuda.event(), cuda.event()

    cart.Barrier()
    for s in range(1, STEPS + 1):
        ev_a.record(st["sb"])
        t0 = MPI.Wtime()
        step(ctx, st)
        ev_b.record(st["sb"])
        ev_b.synchronize()
        wall += MPI.Wtime() - t0
        ev_ms += cuda.event_elapsed_time(ev_a, ev_b)

        if s % 100 == 0:
            new_h = st["un"].copy_to_host()
            old_h = st["u"].copy_to_host()
            e, r = global_diagnostics(ctx, old_h, new_h)
            energy_trace[s] = e
            residual_trace[s] = r
        st["u"], st["un"] = st["un"], st["u"]      

    final = st["u"].copy_to_host()
    t_wall = np.zeros(1)
    t_ev = np.zeros(1)
    cart.Allreduce(np.array([wall * 1e3]), t_wall, op=MPI.MAX)
    cart.Allreduce(np.array([ev_ms]), t_ev, op=MPI.MAX)
    return {"final": final, "energy": energy_trace, "residual": residual_trace,
            "wall_ms": float(t_wall[0]), "event_ms": float(t_ev[0])}


def ptx_audit():
    asm = tiled_stencil_kernel.inspect_asm()
    ptx = next(iter(asm.values())) if isinstance(asm, dict) else asm
    counts = {
        "st.shared.f64": ptx.count("st.shared.f64"),
        "ld.shared.f64": ptx.count("ld.shared.f64"),
        "bar.sync": ptx.count("bar.sync"),
    }
    passed = (counts["st.shared.f64"] > 0 and counts["ld.shared.f64"] > 0
              and counts["bar.sync"] > 0)
    return counts, passed


def run_solver(ctx):
    cart = ctx["cart"]
    rank = ctx["rank"]
    fields = initialize_subdomains(ctx)

    
    kmax_loc = np.array([fields[1].max()])
    kmax = np.zeros(1)
    cart.Allreduce(kmax_loc, kmax, op=MPI.MAX)
    kmax = float(kmax[0])
    dt_max = DX * DX / (4.0 * kmax)
    ctx["dt"] = 0.8 * dt_max
    ctx["inv_h2"] = 1.0 / (DX * DX)

    
    for m in (step_serialized, step_overlapped):
        w = make_state(ctx, fields)
        for _ in range(2):
            m(ctx, w)
            w["u"], w["un"] = w["un"], w["u"]
    cart.Barrier()

    res_s = run_mode(ctx, fields, "serialized")
    res_o = run_mode(ctx, fields, "overlapped")

    
    loc = np.array([np.max(np.abs(res_s["final"] - res_o["final"]))])
    glob = np.zeros(1)
    cart.Allreduce(loc, glob, op=MPI.MAX)
    parity = float(glob[0])

    counts, passed = ptx_audit()

    if rank == 0:
        print(f"[config] ranks=4 grid=2x2 github_user={ctx['github_user']}")
        print(f"[config] gpu_uuid={ctx['gpu_uuid']}  INSTANCE_SEED={ctx['seed']}")
        print(f"[cfl] kappa_max={kmax:.8f} dx={DX:.10e} dt_max={dt_max:.6e} dt_used={ctx['dt']:.6e}")
        for name, r in (("serialized", res_s), ("overlapped", res_o)):
            print(f"[{name}] wall_ms={r['wall_ms']:.3f} cuda_event_ms={r['event_ms']:.3f}")
            for s in sorted(r["energy"]):
                rs = r["residual"].get(s)
                extra = f"  L2_update={rs:.6e}" if rs is not None else ""
                print(f"[{name}] step {s:3d}  E={r['energy'][s]:.12e}{extra}")
        print(f"[parity] max|serialized-overlapped| = {parity:.3e}")
        print(f"[ptx] {counts}  ptx_audit_passed={passed}")
        e = [res_o["energy"][s] for s in sorted(res_o["energy"])]
        print(f"[energy] monotone decreasing: {all(e[i + 1] < e[i] for i in range(len(e) - 1))}")

        manifest = {
            "github_username": ctx["github_user"],
            "gpu_uuid": ctx["gpu_uuid"],
            "instance_seed": ctx["seed"],
            "mpi_ranks": 4,
            "process_grid": [2, 2],
            "global_grid": [GLOBAL_NY, GLOBAL_NX],
            "steps": STEPS,
            "kappa_max": kmax,
            "dt_max": dt_max,
            "dt_used": ctx["dt"],
            "serialized_ms": res_s["wall_ms"],
            "overlapped_ms": res_o["wall_ms"],
            "serialized_cuda_event_ms": res_s["event_ms"],
            "overlapped_cuda_event_ms": res_o["event_ms"],
            "energy_trace_overlapped": {str(k): v for k, v in res_o["energy"].items()},
            "energy_trace_serialized": {str(k): v for k, v in res_s["energy"].items()},
            "parity_max_diff": parity,
            "ptx_counts": counts,
            "ptx_audit_passed": passed,
        }
        with open("submission_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print("[manifest] submission_manifest.json written")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: distributed_cuda_solver.py <github_username>")
    context = init_distributed_context(sys.argv[1])
    run_solver(context)
