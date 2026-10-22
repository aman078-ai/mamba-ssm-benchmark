"""GPU-only checks: deterministic kernel mode, training path vs. generation path, long sequences."""

import pytest
import torch

from models.lm import LM, LMConfig
from models.scan import selective_scan

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
try:
    import triton  # noqa: F401
    HAS_TRITON = True
except Exception:
    HAS_TRITON = False
needs_triton = pytest.mark.skipif(not HAS_TRITON, reason="needs triton")


def scan_inputs(B=3, L=300, D=96, N=16):
    torch.manual_seed(0)
    x = torch.randn(B, L, D, device="cuda", requires_grad=True)
    dt = (torch.rand(B, L, D, device="cuda") * 0.2 + 1e-3).requires_grad_()
    A = (-torch.rand(D, N, device="cuda") * 4 - 0.05).requires_grad_()
    Bm = torch.randn(B, L, N, device="cuda", requires_grad=True)
    Cm = torch.randn(B, L, N, device="cuda", requires_grad=True)
    return x, dt, A, Bm, Cm


@cuda
@needs_triton
def test_triton_backward_is_bitwise_deterministic_in_deterministic_mode():
    ins = scan_inputs()

    def grads():
        y, _ = selective_scan(*ins, scan="triton")
        return torch.autograd.grad(y.square().sum(), ins)

    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        g1, g2 = grads(), grads()
    finally:
        torch.use_deterministic_algorithms(False)
    for a, b in zip(g1, g2):
        assert torch.equal(a, b)
    ref = torch.autograd.grad(selective_scan(*ins, scan="chunked")[0].square().sum(), ins)
    for a, b in zip(g1, ref):                                   # and it is still correct
        torch.testing.assert_close(a, b, rtol=2e-3, atol=2e-3)


@cuda
@pytest.mark.parametrize("layers", ["MMM", "MAM", "RMR"])
def test_gpu_full_forward_matches_stepwise_decoding_fp32(layers):
    """The training path (Triton scan / chunkwise retention) and the generation path agree on the GPU."""
    torch.manual_seed(0)
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        m = LM(LMConfig(layers=layers, d_model=64, n_head=4, d_state=16)).cuda().eval()
        x = torch.randint(0, 256, (2, 96), device="cuda")
        with torch.no_grad():
            full = m(x)
            st = m.init_state(2, 96, "cuda", torch.float32)
            steps = torch.cat([m(x[:, t:t + 1], st, t) for t in range(96)], 1)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old
    torch.testing.assert_close(steps, full, rtol=1e-4, atol=1e-4)


@cuda
def test_long_sequence_forward_backward_is_finite():
    torch.manual_seed(0)
    m = LM(LMConfig(layers="MM", d_model=64, d_state=16)).cuda()
    x = torch.randint(0, 256, (1, 16384), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = m(x).float().logsumexp(-1).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


@cuda
@needs_triton
def test_triton_matches_chunked_scan_on_long_sequences():
    x, dt, A, Bm, Cm = (t.detach() for t in scan_inputs(B=1, L=8192, D=64, N=16))
    with torch.no_grad():
        y_t, h_t = selective_scan(x, dt, A, Bm, Cm, scan="triton")
        y_c, h_c = selective_scan(x, dt, A, Bm, Cm, scan="chunked")
    torch.testing.assert_close(y_t, y_c, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(h_t, h_c, rtol=1e-3, atol=1e-3)
