"""Selective scan: discretization + four scan algorithms + a memory-efficient autograd op.

The selective SSM (Mamba / S6) runs, independently for every channel d and state index n,
the diagonal linear recurrence

    h_t = a_t * h_{t-1} + b_t,        y_t = sum_n C_t[n] * h_t[n]  (+ D * x_t)

where a_t = exp(Δ_t A) and b_t = B̄_t x_t come from discretizing the continuous system
dh/dt = A h + B x with an input-dependent step Δ_t (this input dependence is "selection").

Because (a, b) pairs compose associatively,

    (a2, b2) ∘ (a1, b1) = (a2 * a1,  a2 * b1 + b2)      # "apply 1, then 2"

the recurrence is a *scan* and can be computed in many ways:

  scan_sequential   O(L) work, O(L) depth         the textbook loop (reference)
  scan_hillis_steele O(L log L) work, O(log L) depth  all prefixes doubled each step
  scan_blelloch     O(L) work, O(2 log L) depth    work-efficient up-sweep / down-sweep
  scan_chunked      O(L) work, O(L/C + log(L/C)) depth  sequential inside chunks of size C,
                                                   log-depth scan across chunk boundaries

SelectiveScanFn wraps any of them in a custom autograd.Function that stores only the small
(B, L, D) / (B, L, N) inputs and *recomputes* the (B, L, D, N) states in the backward pass,
whose core is itself a reverse scan:  dh_t = g_t + a_{t+1} * dh_{t+1}.
"""

import math

import torch

# ----------------------------------------------------------------------------------------
# Discretization (continuous (Δ, A, B) -> discrete (a, b)); A is diagonal and negative.
# ----------------------------------------------------------------------------------------

DISCRETIZATIONS = ("zoh", "euler", "bilinear")


def discretize(delta, A, Bm, x, method="zoh"):
    """delta, x: (B, L, D); A: (D, N) with A < 0; Bm: (B, L, N) -> a, b: (B, L, D, N).

    zoh      exact zero-order hold:  a = exp(ΔA),  b = (exp(ΔA) - 1) / A * B x
    euler    Mamba's simplification: a = exp(ΔA),  b = Δ B x
    bilinear Tustin transform:       a = (1 + ΔA/2) / (1 - ΔA/2),  b = Δ / (1 - ΔA/2) * B x
    """
    dA = delta.unsqueeze(-1) * A                       # (B, L, D, N)
    Bx = x.unsqueeze(-1) * Bm.unsqueeze(2)             # (B, L, D, N)
    if method == "zoh":
        return torch.exp(dA), torch.expm1(dA) / A * Bx
    if method == "euler":
        return torch.exp(dA), delta.unsqueeze(-1) * Bx
    if method == "bilinear":
        inv = 1.0 / (1.0 - 0.5 * dA)
        return (1.0 + 0.5 * dA) * inv, delta.unsqueeze(-1) * inv * Bx
    raise ValueError(f"unknown discretization {method!r}")


def zoh_matrix(A, B, delta):
    """General (dense) ZOH, used as a reference in tests: Â = expm(AΔ), B̂ = A^{-1}(Â - I) B."""
    Ad = torch.matrix_exp(A * delta)
    eye = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    return Ad, torch.linalg.solve(A, (Ad - eye) @ B)


# ----------------------------------------------------------------------------------------
# Scan algorithms.  All take a, b of shape (B, L, *rest) and optional h0 of shape (B, *rest)
# and return all states h of shape (B, L, *rest).
# ----------------------------------------------------------------------------------------

def scan_sequential(a, b, h0=None):
    h = torch.zeros_like(b[:, 0]) if h0 is None else h0
    out = torch.empty_like(b)
    for t in range(a.shape[1]):
        h = a[:, t] * h + b[:, t]
        out[:, t] = h
    return out


def _fold_h0(a, b, h0):
    if h0 is None:
        return b
    b = b.clone()
    b[:, 0] = b[:, 0] + a[:, 0] * h0
    return b


def scan_hillis_steele(a, b, h0=None):
    """Inclusive scan by recursive doubling: after step k every position holds the
    composition of the previous 2^k elements."""
    b = _fold_h0(a, b, h0)
    L = a.shape[1]
    offset = 1
    while offset < L:
        b = torch.cat([b[:, :offset], torch.addcmul(b[:, offset:], a[:, offset:], b[:, :-offset])], 1)
        if offset * 2 < L:  # a is not needed after the last step
            a = torch.cat([a[:, :offset], a[:, offset:] * a[:, :-offset]], 1)
        offset *= 2
    return b


def scan_blelloch(a, b, h0=None):
    """Work-efficient (Blelloch 1990) scan: up-sweep builds a reduction tree, down-sweep
    pushes exclusive prefixes back down.  Pads L to a power of two with identities (1, 0)."""
    B, L = a.shape[:2]
    rest = a.shape[2:]
    P = 1 << (L - 1).bit_length()
    A = torch.ones((B, P) + rest, dtype=a.dtype, device=a.device)
    S = torch.zeros((B, P) + rest, dtype=b.dtype, device=b.device)
    A[:, :L] = a
    S[:, :L] = b

    def views(s):
        Av, Sv = A.view((B, P // s, s) + rest), S.view((B, P // s, s) + rest)
        return Av[:, :, s // 2 - 1], Av[:, :, s - 1], Sv[:, :, s // 2 - 1], Sv[:, :, s - 1]

    s = 2
    while s <= P:                       # up-sweep: right <- right ∘ left
        Al, Ar, Sl, Sr = views(s)
        Sr.addcmul_(Ar, Sl)
        Ar.mul_(Al)
        s *= 2
    A[:, P - 1] = 1.0                   # down-sweep from the identity
    S[:, P - 1] = 0.0
    s = P
    while s >= 2:                       # left <- right;  right <- left_old ∘ right
        Al, Ar, Sl, Sr = views(s)
        tA, tS = Al.clone(), Sl.clone()
        Al.copy_(Ar)
        Sl.copy_(Sr)
        Sr.mul_(tA).add_(tS)
        Ar.mul_(tA)
        s //= 2
    # A, S now hold exclusive prefixes; the state entering step t is A_t * h0 + S_t.
    prev = S[:, :L] if h0 is None else torch.addcmul(S[:, :L], A[:, :L], h0.unsqueeze(1))
    return torch.addcmul(b, a, prev)


def default_chunk(L):
    return max(4, 1 << math.ceil(math.log2(max(1.0, math.sqrt(L)))))


def scan_chunked(a, b, h0=None, chunk_size=None):
    """Two-level scan.  (1) all chunks scan their C elements in parallel (C sequential
    steps on (B, L/C, ...) tensors); (2) chunk summaries are combined with a log-depth
    scan; (3) each position is corrected with the state that entered its chunk."""
    B, L = a.shape[:2]
    rest = a.shape[2:]
    C = chunk_size or default_chunk(L)
    nc = -(-L // C)
    if nc * C != L:
        pad = (B, nc * C - L) + rest
        a = torch.cat([a, torch.ones(pad, dtype=a.dtype, device=a.device)], 1)
        b = torch.cat([b, torch.zeros(pad, dtype=b.dtype, device=b.device)], 1)
    a4 = a.reshape((B, nc, C) + rest)
    b4 = b.reshape((B, nc, C) + rest)
    h = torch.empty_like(b4)
    h[:, :, 0] = b4[:, :, 0]
    for t in range(1, C):
        h[:, :, t] = torch.addcmul(b4[:, :, t], a4[:, :, t], h[:, :, t - 1])
    P = torch.cumprod(a4, dim=2)                                  # decay since chunk start
    h_end = scan_hillis_steele(P[:, :, -1], h[:, :, -1], h0)      # state at each chunk end
    zero = torch.zeros_like(h_end[:, :1]) if h0 is None else h0.unsqueeze(1)
    h_in = torch.cat([zero, h_end[:, :-1]], 1)                    # state entering each chunk
    h.addcmul_(P, h_in.unsqueeze(2))
    return h.reshape((B, nc * C) + rest)[:, :L]


SCANS = {
    "sequential": scan_sequential,
    "hillis_steele": scan_hillis_steele,
    "blelloch": scan_blelloch,
    "chunked": scan_chunked,
}


# ----------------------------------------------------------------------------------------
# Memory-efficient selective scan with a hand-written backward pass.
# ----------------------------------------------------------------------------------------

# Element budget for one transient (B, L, d_block, N) buffer when d_block is not given:
# 2^24 float32 elements = 64 MB, so the ~10 live buffers of the backward pass stay < 1 GB.
SCAN_ELEMENT_BUDGET = 1 << 24


def _blocks(shape_bld, N, d_block):
    Bsz, L, D = shape_bld
    if not d_block:
        d_block = max(1, SCAN_ELEMENT_BUDGET // (Bsz * L * N))
    d_block = min(d_block, D)
    return [(i, min(i + d_block, D)) for i in range(0, D, d_block)]


class SelectiveScanFn(torch.autograd.Function):
    """y = selective_scan(x, Δ, A, B, C) with recomputation.

    Saved for backward: only x, Δ, A, B, C, h0 (no (B, L, D, N) tensors).  Channels are
    processed in blocks of d_block (default: sized to SCAN_ELEMENT_BUDGET) so the transient
    (B, L, d_block, N) buffers stay small,
    the same tiling idea as Mamba's fused CUDA kernel, expressed in PyTorch.
    """

    @staticmethod
    def forward(ctx, x, delta, A, Bm, Cm, h0, method, scan_name, d_block, chunk_size):
        scan_fn = _scan_fn(scan_name, chunk_size)
        Bsz, L, D = x.shape
        y = torch.empty_like(x)
        last = torch.empty(Bsz, D, A.shape[1], dtype=x.dtype, device=x.device)
        for i, j in _blocks(x.shape, A.shape[1], d_block):
            a, b = discretize(delta[..., i:j], A[i:j], Bm, x[..., i:j], method)
            h = scan_fn(a, b, None if h0 is None else h0[:, i:j])
            del a, b
            y[..., i:j] = torch.einsum("bldn,bln->bld", h, Cm)
            last[:, i:j] = h[:, -1]
            del h
        ctx.save_for_backward(x, delta, A, Bm, Cm, h0)
        ctx.cfg = (method, scan_name, d_block, chunk_size)
        return y, last

    @staticmethod
    def backward(ctx, dy, dlast):
        x, delta, A, Bm, Cm, h0 = ctx.saved_tensors
        method, scan_name, d_block, chunk_size = ctx.cfg
        scan_fn = _scan_fn(scan_name, chunk_size)
        dx, ddelta, dA = torch.zeros_like(x), torch.zeros_like(delta), torch.zeros_like(A)
        dBm, dCm = torch.zeros_like(Bm), torch.zeros_like(Cm)
        dh0 = None if h0 is None else torch.zeros_like(h0)
        for i, j in _blocks(x.shape, A.shape[1], d_block):
            with torch.enable_grad():
                ins = [t.detach().requires_grad_() for t in (delta[..., i:j], A[i:j], Bm, x[..., i:j])]
                a, b = discretize(ins[0], ins[1], ins[2], ins[3], method)
            with torch.no_grad():
                h0_blk = None if h0 is None else h0[:, i:j]
                h = scan_fn(a.detach(), b.detach(), h0_blk)                    # recompute
                dy_blk = dy[..., i:j]
                dCm += torch.einsum("bld,bldn->bln", dy_blk, h)
                g = dy_blk.unsqueeze(-1) * Cm.unsqueeze(2)                     # dL/dh_t (direct)
                if dlast is not None:
                    g[:, -1] += dlast[:, i:j]
                a_next = torch.cat([a.detach()[:, 1:], torch.ones_like(a[:, :1])], 1)
                dh = scan_fn(a_next.flip(1), g.flip(1)).flip(1)                # reverse scan
                del g, a_next
                h_prev = torch.cat([torch.zeros_like(h[:, :1]) if h0 is None else h0_blk.unsqueeze(1),
                                    h[:, :-1]], 1)
                del h
                da = dh * h_prev
                del h_prev
                if dh0 is not None:
                    dh0[:, i:j] = dh[:, 0] * a.detach()[:, 0]
            g_delta, g_A, g_B, g_x = torch.autograd.grad((a, b), ins, (da, dh))
            ddelta[..., i:j] = g_delta
            dA[i:j] = g_A
            dBm += g_B
            dx[..., i:j] = g_x
            del a, b, da, dh
        return dx, ddelta, dA, dBm, dCm, dh0, None, None, None, None


def _scan_fn(name, chunk_size):
    if name == "chunked":
        return lambda a, b, h0=None: scan_chunked(a, b, h0, chunk_size)
    return SCANS[name]


def selective_scan_reference(x, delta, A, Bm, Cm, h0=None, method="zoh"):
    """Naive implementation: a Python loop over time, differentiated by autograd.
    Stores every intermediate state, so memory is O(B L D N)."""
    Bsz, L, D = x.shape
    h = x.new_zeros(Bsz, D, A.shape[1]) if h0 is None else h0
    ys = []
    for t in range(L):
        a, b = discretize(delta[:, t:t + 1], A, Bm[:, t:t + 1], x[:, t:t + 1], method)
        h = a[:, 0] * h + b[:, 0]
        ys.append(torch.einsum("bdn,bn->bd", h, Cm[:, t]))
    return torch.stack(ys, 1), h


def selective_scan(x, delta, A, Bm, Cm, h0=None, method="zoh", scan="chunked",
                   d_block=None, chunk_size=None):
    """Returns (y, h_last).  x, delta: (B, L, D); A: (D, N); Bm, Cm: (B, L, N); h0: (B, D, N).

    scan = 'reference' (autograd loop) | 'sequential' | 'hillis_steele' | 'blelloch'
           | 'chunked' | 'triton' (fused kernels, see kernels/triton_scan.py)
           | 'auto' (triton when available on CUDA, else chunked)
    Computation is done in (at least) float32 regardless of autocast.
    """
    if scan == "auto":
        scan = "triton" if _triton_ok(x, A, method) else "chunked"
    dtype = x.dtype
    ct = torch.promote_types(dtype, torch.float32)
    with torch.autocast(device_type=x.device.type, enabled=False):
        args = [t.to(ct) if t is not None else None for t in (x, delta, A, Bm, Cm, h0)]
        if scan == "reference":
            y, last = selective_scan_reference(*args, method=method)
        elif scan == "triton":
            from kernels.triton_scan import selective_scan_triton
            y, last = selective_scan_triton(*args, method=method, d_block=d_block)
        else:
            y, last = SelectiveScanFn.apply(*args, method, scan, d_block, chunk_size)
    return y.to(dtype), last


_HAS_TRITON = None


def _triton_ok(x, A, method):
    global _HAS_TRITON
    if _HAS_TRITON is None:
        try:
            import kernels.triton_scan  # noqa: F401
            _HAS_TRITON = True
        except Exception:
            _HAS_TRITON = False
    N = A.shape[1]
    return _HAS_TRITON and x.is_cuda and method in DISCRETIZATIONS and N & (N - 1) == 0
