"""Fused selective-scan kernels in Triton (forward + backward).

Why: the PyTorch scans in models/scan.py materialize (B, L, D, N) tensors for a, b and h in
GPU DRAM.  The arithmetic is trivial, so they are memory-bandwidth bound.  Mamba's key
systems idea is to keep the state in on-chip memory: each program owns a (BLOCK_D, N) tile of
the state in registers and walks the sequence, reading only the (B, L, D) and (B, L, N)
inputs and writing the (B, L, D) output.  DRAM traffic drops from O(B L D N) to O(B L D).

Forward  : h_t = exp(Δ_t A) h_{t-1} + B̄_t x_t,   y_t = <h_t, C_t>
Backward : walks the sequence in reverse with the adjoint recurrence
           dh_t = dy_t C_t + exp(Δ_{t+1} A) dh_{t+1}
           and turns dh_t into dΔ, dA, dB, dC, dx with the discretization's derivatives.
           The h_{t-1} it needs are recomputed by the forward kernel into a scratch buffer,
           one channel tile at a time (bounded by HBUF_BUDGET).
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


ZOH, EULER, BILINEAR = 0, 1, 2
METHODS = {"zoh": ZOH, "euler": EULER, "bilinear": BILINEAR}


@triton.jit
def _fwd_kernel(x_ptr, dt_ptr, A_ptr, B_ptr, C_ptr, h0_ptr, y_ptr, last_ptr, hbuf_ptr,
                L, D, d_start, d_end, D_tile,
                N: tl.constexpr, BLOCK_D: tl.constexpr, METHOD: tl.constexpr,
                HAS_H0: tl.constexpr, WRITE_Y: tl.constexpr, STORE_H: tl.constexpr,
                NUM_STAGES: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_d = d_start + pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, N)
    mask_d = offs_d < d_end
    mask2 = mask_d[:, None]

    A = tl.load(A_ptr + offs_d[:, None] * N + offs_n[None, :], mask=mask2, other=-1.0)
    if HAS_H0:
        h = tl.load(h0_ptr + pid_b * D * N + offs_d[:, None] * N + offs_n[None, :], mask=mask2, other=0.0)
    else:
        h = tl.zeros((BLOCK_D, N), dtype=tl.float32)

    x_base = x_ptr + pid_b * L * D + offs_d
    dt_base = dt_ptr + pid_b * L * D + offs_d
    y_base = y_ptr + pid_b * L * D + offs_d
    B_base = B_ptr + pid_b * L * N + offs_n
    C_base = C_ptr + pid_b * L * N + offs_n
    hb_base = hbuf_ptr + pid_b * L * D_tile * N + (offs_d - d_start)[:, None] * N + offs_n[None, :]
    for t in tl.range(0, L, num_stages=NUM_STAGES):
        x = tl.load(x_base + t * D, mask=mask_d, other=0.0)
        dt = tl.load(dt_base + t * D, mask=mask_d, other=0.0)
        Bt = tl.load(B_base + t * N)
        dA = dt[:, None] * A
        if METHOD == 0:
            a = tl.exp(dA)
            bx = libdevice.expm1(dA) / A * (Bt[None, :] * x[:, None])
        elif METHOD == 1:
            a = tl.exp(dA)
            bx = dt[:, None] * Bt[None, :] * x[:, None]
        else:
            inv = 1.0 / (1.0 - 0.5 * dA)
            a = (1.0 + 0.5 * dA) * inv
            bx = dt[:, None] * inv * Bt[None, :] * x[:, None]
        h = a * h + bx
        if STORE_H:
            tl.store(hb_base + t * D_tile * N, h, mask=mask2)
        if WRITE_Y:
            Ct = tl.load(C_base + t * N)
            tl.store(y_base + t * D, tl.sum(h * Ct[None, :], axis=1), mask=mask_d)
    if WRITE_Y:
        tl.store(last_ptr + pid_b * D * N + offs_d[:, None] * N + offs_n[None, :], h, mask=mask2)


@triton.jit
def _combine(a1, b1, a2, b2):
    # (a2, b2) ∘ (a1, b1): apply step 1, then step 2
    return a1 * a2, a2 * b1 + b2


@triton.jit
def _fwd_chunked_kernel(x_ptr, dt_ptr, A_ptr, B_ptr, C_ptr, h0_ptr, y_ptr, last_ptr, hbuf_ptr,
                        L, D, d_start, d_end, D_tile,
                        N: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_T: tl.constexpr, METHOD: tl.constexpr,
                        HAS_H0: tl.constexpr, WRITE_Y: tl.constexpr, STORE_H: tl.constexpr):
    """Same contract as _fwd_kernel, but processes BLOCK_T time steps at once: loads a
    (BLOCK_T, BLOCK_D*N) block, runs a parallel prefix scan over time in registers
    (tl.associative_scan), then applies the carried-in state.  Sequential steps: L / BLOCK_T.
    The (channel, state) axes are flattened to one axis of size BLOCK_D*N."""
    DN: tl.constexpr = BLOCK_D * N
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    j = tl.arange(0, DN)
    offs_d = d_start + pid_d * BLOCK_D + j // N                             # channel of column j
    offs_n = j % N                                                          # state index of column j
    mask_d = offs_d < d_end
    offs_t = tl.arange(0, BLOCK_T)

    A = tl.load(A_ptr + offs_d * N + offs_n, mask=mask_d, other=-1.0)       # (DN,)
    if HAS_H0:
        h = tl.load(h0_ptr + pid_b * D * N + offs_d * N + offs_n, mask=mask_d, other=0.0)
    else:
        h = tl.zeros((DN,), dtype=tl.float32)
    is_last = (offs_t == BLOCK_T - 1)[:, None]

    for c in range(0, L, BLOCK_T):
        t = c + offs_t
        mt = t < L
        m2 = mt[:, None] & mask_d[None, :]
        row = pid_b * L * D + t[:, None] * D + offs_d[None, :]
        x = tl.load(x_ptr + row, mask=m2, other=0.0)                        # (T, DN), x[t, d(j)]
        dt = tl.load(dt_ptr + row, mask=m2, other=0.0)                      # padded steps: Δ=0 -> identity
        Bt = tl.load(B_ptr + pid_b * L * N + t[:, None] * N + offs_n[None, :], mask=mt[:, None], other=0.0)
        dA = dt * A[None, :]
        if METHOD == 0:
            a = tl.exp(dA)
            b = libdevice.expm1(dA) / A[None, :] * (Bt * x)
        elif METHOD == 1:
            a = tl.exp(dA)
            b = dt * Bt * x
        else:
            inv = 1.0 / (1.0 - 0.5 * dA)
            a = (1.0 + 0.5 * dA) * inv
            b = dt * inv * Bt * x
        a_cum, b_cum = tl.associative_scan((a, b), axis=0, combine_fn=_combine)
        hs = a_cum * h[None, :] + b_cum                                      # (T, DN) states
        if STORE_H:
            hb = hbuf_ptr + pid_b * L * D_tile * N + t[:, None] * D_tile * N                 + (offs_d - d_start)[None, :] * N + offs_n[None, :]
            tl.store(hb, hs, mask=m2)
        if WRITE_Y:
            Ct = tl.load(C_ptr + pid_b * L * N + t[:, None] * N + offs_n[None, :], mask=mt[:, None], other=0.0)
            yv = tl.sum(tl.reshape(hs * Ct, (BLOCK_T, BLOCK_D, N)), axis=2)  # (T, BLOCK_D)
            od = d_start + pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
            tl.store(y_ptr + pid_b * L * D + t[:, None] * D + od[None, :], yv,
                     mask=mt[:, None] & (od < d_end)[None, :])
        h = tl.sum(tl.where(is_last, hs, 0.0), axis=0)                       # carry = last state
    if WRITE_Y:
        tl.store(last_ptr + pid_b * D * N + offs_d * N + offs_n, h, mask=mask_d)


@triton.jit
def _bwd_kernel(x_ptr, dt_ptr, A_ptr, B_ptr, C_ptr, h0_ptr, dy_ptr, dlast_ptr, hbuf_ptr,
                dx_ptr, ddt_ptr, dA_part_ptr, dB_ptr, dC_ptr, dh0_ptr,
                L, D, d_start, d_end, D_tile,
                N: tl.constexpr, BLOCK_D: tl.constexpr, METHOD: tl.constexpr, HAS_H0: tl.constexpr,
                NUM_STAGES: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_d = d_start + pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, N)
    mask_d = offs_d < d_end
    mask2 = mask_d[:, None]
    tile2 = offs_d[:, None] * N + offs_n[None, :]

    A = tl.load(A_ptr + tile2, mask=mask2, other=-1.0)
    dh = tl.load(dlast_ptr + pid_b * D * N + tile2, mask=mask2, other=0.0)
    if HAS_H0:
        h0 = tl.load(h0_ptr + pid_b * D * N + tile2, mask=mask2, other=0.0)
    dA_acc = tl.zeros((BLOCK_D, N), dtype=tl.float32)

    row = pid_b * L * D + offs_d
    hb_base = hbuf_ptr + pid_b * L * D_tile * N + (offs_d - d_start)[:, None] * N + offs_n[None, :]
    bn = pid_b * L * N + offs_n          # dB, dC are reduced over channel tiles with atomics
    for i in tl.range(0, L, num_stages=NUM_STAGES):
        t = L - 1 - i
        x = tl.load(x_ptr + row + t * D, mask=mask_d, other=0.0)
        dt = tl.load(dt_ptr + row + t * D, mask=mask_d, other=0.0)
        dy = tl.load(dy_ptr + row + t * D, mask=mask_d, other=0.0)
        Bt = tl.load(B_ptr + pid_b * L * N + t * N + offs_n)
        Ct = tl.load(C_ptr + pid_b * L * N + t * N + offs_n)
        h = tl.load(hb_base + t * D_tile * N, mask=mask2, other=0.0)
        hp = tl.load(hb_base + (t - 1) * D_tile * N, mask=mask2 & (t > 0), other=0.0)
        if HAS_H0:
            hp += tl.where(t == 0, h0, 0.0)

        tl.atomic_add(dC_ptr + bn + t * N, tl.sum(dy[:, None] * h, axis=0))
        dh += dy[:, None] * Ct[None, :]                         # total dL/dh_t
        z = dt[:, None] * A
        if METHOD == 2:
            u = 1.0 / (1.0 - 0.5 * z)
            a = (1.0 + 0.5 * z) * u
        else:
            a = tl.exp(z)
        da = dh * hp
        Bx = Bt[None, :] * x[:, None]
        if METHOD == 0:
            e = libdevice.expm1(z)
            coef = e / A                                          # b = coef * B x
            # d(coef)/dA = dt^2 * f(z),  f(z) = (z e^z - expm1 z) / z^2 = 1/2 + z/3 + z^2/8 + ...
            f = tl.where(tl.abs(z) < 1e-2, 0.5 + z / 3.0 + z * z / 8.0, (z * a - e) / (z * z))
            ddt = tl.sum(da * A * a + dh * a * Bx, axis=1)
            dA_acc += da * dt[:, None] * a + dh * dt[:, None] * dt[:, None] * f * Bx
            dBt = tl.sum(dh * coef * x[:, None], axis=0)
            dx = tl.sum(dh * coef * Bt[None, :], axis=1)
        elif METHOD == 1:
            ddt = tl.sum(da * A * a + dh * Bx, axis=1)
            dA_acc += da * dt[:, None] * a
            dBt = tl.sum(dh * dt[:, None] * x[:, None], axis=0)
            dx = tl.sum(dh * dt[:, None] * Bt[None, :], axis=1)
        else:
            # a = (1 + z/2) u, b = dt u B x with u = 1/(1 - z/2):  da/dz = u^2,  db/ddt = u^2 B x
            u2 = u * u
            ddt = tl.sum(da * A * u2 + dh * u2 * Bx, axis=1)
            dA_acc += da * dt[:, None] * u2 + dh * 0.5 * dt[:, None] * dt[:, None] * u2 * Bx
            dBt = tl.sum(dh * dt[:, None] * u * x[:, None], axis=0)
            dx = tl.sum(dh * dt[:, None] * u * Bt[None, :], axis=1)
        tl.atomic_add(dB_ptr + bn + t * N, dBt)
        tl.store(dx_ptr + row + t * D, dx, mask=mask_d)
        tl.store(ddt_ptr + row + t * D, ddt, mask=mask_d)
        dh = dh * a                                               # flows to h_{t-1}
    tl.store(dA_part_ptr + pid_b * D * N + tile2, dA_acc, mask=mask2)
    if HAS_H0:
        tl.store(dh0_ptr + pid_b * D * N + tile2, dh, mask=mask2)


# Tuned on an RTX 3050 Laptop (B=16, L=512, D=512, N=16): differences between configs
# were within ~20%; the loop is bound by per-step latency, not bandwidth.
BLOCK_D = 16
NUM_WARPS = 1
NUM_STAGES = 1


# Forward kernel choice: "sequential" (default) or "chunked" (block-parallel scan in registers via
# tl.associative_scan).  Both are tested; on the RTX 3050 the chunked variant was not faster
# (register pressure of the (BLOCK_T, BLOCK_D*N) block), so it is kept as an alternative.
FWD_IMPL = "sequential"
CHUNK_BLOCK_D, CHUNK_BLOCK_T, CHUNK_WARPS = 16, 16, 4
# Scratch budget (elements) for recomputed states in the backward pass: 2^27 fp32 = 512 MB.
HBUF_BUDGET = 1 << 27


def _block_d(Bsz, D):
    """Channel tile per program: shrink it when B*D is small so the GPU still gets >= ~128 programs."""
    bd = BLOCK_D
    while bd > 4 and Bsz * triton.cdiv(D, bd) < 128:
        bd //= 2
    return bd


def _launch_fwd(x, delta, A, Bm, Cm, h0, y, last, hbuf, d_start, d_end, method, write_y, store_h):
    Bsz, L, D = x.shape
    N = A.shape[1]
    D_tile = d_end - d_start
    dummy = x
    if FWD_IMPL == "chunked":
        grid = (Bsz, triton.cdiv(D_tile, CHUNK_BLOCK_D))
        _fwd_chunked_kernel[grid](x, delta, A, Bm, Cm, h0 if h0 is not None else dummy,
                                  y if y is not None else dummy, last if last is not None else dummy,
                                  hbuf if hbuf is not None else dummy, L, D, d_start, d_end, D_tile,
                                  N=N, BLOCK_D=CHUNK_BLOCK_D, BLOCK_T=CHUNK_BLOCK_T, METHOD=METHODS[method],
                                  HAS_H0=h0 is not None, WRITE_Y=write_y, STORE_H=store_h, num_warps=CHUNK_WARPS)
        return
    bd = _block_d(Bsz, D_tile)
    grid = (Bsz, triton.cdiv(D_tile, bd))
    _fwd_kernel[grid](x, delta, A, Bm, Cm, h0 if h0 is not None else dummy, y if y is not None else dummy,
                      last if last is not None else dummy, hbuf if hbuf is not None else dummy,
                      L, D, d_start, d_end, D_tile, N=N, BLOCK_D=bd, METHOD=METHODS[method],
                      HAS_H0=h0 is not None, WRITE_Y=write_y, STORE_H=store_h,
                      NUM_STAGES=NUM_STAGES, num_warps=NUM_WARPS)


class TritonSelectiveScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, delta, A, Bm, Cm, h0, method):
        x, delta, A, Bm, Cm = (t.contiguous() for t in (x, delta, A, Bm, Cm))
        h0 = None if h0 is None else h0.contiguous()
        Bsz, L, D = x.shape
        y = torch.empty_like(x)
        last = torch.empty(Bsz, D, A.shape[1], device=x.device, dtype=x.dtype)
        _launch_fwd(x, delta, A, Bm, Cm, h0, y, last, None, 0, D, method, True, False)
        ctx.save_for_backward(x, delta, A, Bm, Cm, h0)
        ctx.method = method
        return y, last

    @staticmethod
    def backward(ctx, dy, dlast):
        x, delta, A, Bm, Cm, h0 = ctx.saved_tensors
        Bsz, L, D = x.shape
        N = A.shape[1]
        dy = dy.contiguous()
        dlast = torch.zeros(Bsz, D, N, device=x.device) if dlast is None else dlast.contiguous()
        dx, ddelta = torch.empty_like(x), torch.empty_like(delta)
        bd = _block_d(Bsz, D)
        dA_part = torch.empty(Bsz, D, N, device=x.device)
        dB = torch.zeros(Bsz, L, N, device=x.device)
        dC = torch.zeros(Bsz, L, N, device=x.device)
        dh0 = torch.empty_like(h0) if h0 is not None else None
        # channel tiles sized so the recomputed-state scratch buffer stays within budget
        tile = max(bd, (HBUF_BUDGET // (Bsz * L * N)) // bd * bd)
        for d0 in range(0, D, tile):
            d1 = min(D, d0 + tile)
            hbuf = torch.empty(Bsz, L, d1 - d0, N, device=x.device)
            _launch_fwd(x, delta, A, Bm, Cm, h0, None, None, hbuf, d0, d1, ctx.method, False, True)
            grid = (Bsz, triton.cdiv(d1 - d0, bd))
            _bwd_kernel[grid](x, delta, A, Bm, Cm, h0 if h0 is not None else x, dy, dlast, hbuf,
                              dx, ddelta, dA_part, dB, dC, dh0 if dh0 is not None else x,
                              L, D, d0, d1, d1 - d0,
                              N=N, BLOCK_D=bd, METHOD=METHODS[ctx.method],
                              HAS_H0=h0 is not None, NUM_STAGES=NUM_STAGES, num_warps=NUM_WARPS)
            del hbuf
        return dx, ddelta, dA_part.sum(0), dB, dC, dh0, None


def selective_scan_triton(x, delta, A, Bm, Cm, h0=None, method="zoh", d_block=None):
    """Same contract as models.scan.selective_scan (float32 inputs on CUDA)."""
    if method not in METHODS:
        raise ValueError(f"triton scan supports {list(METHODS)}, got {method!r}")
    N = A.shape[1]
    if N & (N - 1):
        raise ValueError("d_state must be a power of two for the Triton kernel")
    return TritonSelectiveScanFn.apply(x, delta, A, Bm, Cm, h0, method)
