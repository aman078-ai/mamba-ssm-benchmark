"""The fused Triton kernels must match the PyTorch reference (forward and all gradients)."""

import pytest
import torch

from models.scan import selective_scan

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
try:
    import triton  # noqa: F401
    HAS_TRITON = True
except Exception:
    HAS_TRITON = False
needs_triton = pytest.mark.skipif(not HAS_TRITON, reason="needs triton")


def inputs(B, L, D, N, h0, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(0)
    mk = lambda *s: torch.randn(*s, generator=g).to(device)
    x = mk(B, L, D).requires_grad_()
    delta = (torch.rand(B, L, D, generator=g) * 0.3 + 1e-3).to(device).requires_grad_()
    A = (-torch.rand(D, N, generator=g) * 4 - 0.05).to(device).requires_grad_()
    Bm, Cm = mk(B, L, N).requires_grad_(), mk(B, L, N).requires_grad_()
    hz = mk(B, D, N).requires_grad_() if h0 else None
    return x, delta, A, Bm, Cm, hz


@cuda
@needs_triton
@pytest.mark.parametrize("method", ["zoh", "euler", "bilinear"])
@pytest.mark.parametrize("shape", [(2, 1, 32, 16), (2, 37, 50, 16), (3, 130, 96, 8)])
@pytest.mark.parametrize("h0", [False, True])
def test_triton_matches_reference(method, shape, h0):
    ins = inputs(*shape, h0)
    leaves = [t for t in ins if t is not None]
    y_ref, last_ref = selective_scan(*ins, method=method, scan="chunked")
    y, last = selective_scan(*ins, method=method, scan="triton")
    torch.testing.assert_close(y, y_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(last, last_ref, rtol=1e-4, atol=1e-4)
    w1, w2 = torch.randn_like(y), torch.randn_like(last)
    g_ref = torch.autograd.grad((y_ref * w1).sum() + (last_ref * w2).sum(), leaves)
    g = torch.autograd.grad((y * w1).sum() + (last * w2).sum(), leaves)
    for name, gi, gr in zip("x delta A B C h0".split(), g, g_ref):
        torch.testing.assert_close(gi, gr, rtol=2e-3, atol=2e-3, msg=lambda m: f"grad {name}: {m}")


@cuda
@needs_triton
@pytest.mark.parametrize("method", ["zoh", "euler", "bilinear"])
def test_triton_block_parallel_forward_matches(method):
    """The alternative block-parallel forward kernel (tl.associative_scan) gives the same result."""
    import kernels.triton_scan as K
    ins = inputs(2, 77, 40, 16, True)
    y_ref, last_ref = selective_scan(*ins, method=method, scan="chunked")
    old = K.FWD_IMPL
    try:
        K.FWD_IMPL = "chunked"
        y, last = selective_scan(*ins, method=method, scan="triton")
        g = torch.autograd.grad(y.sum(), ins[0])[0]
    finally:
        K.FWD_IMPL = old
    g_ref = torch.autograd.grad(y_ref.sum(), ins[0])[0]
    torch.testing.assert_close(y, y_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(last, last_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(g, g_ref, rtol=2e-3, atol=2e-3)
