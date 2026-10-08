"""Correctness of the scan algorithms, discretization and the custom backward pass."""

import numpy as np
import pytest
import torch

from models.scan import (DISCRETIZATIONS, SCANS, discretize, scan_sequential, selective_scan,
                         selective_scan_reference, zoh_matrix)

torch.manual_seed(0)
DT = torch.float64


def rand_ab(B=2, L=37, D=3, N=4):
    a = torch.rand(B, L, D, N, dtype=DT) * 0.9 + 0.05
    b = torch.randn(B, L, D, N, dtype=DT)
    return a, b


def numpy_bruteforce(a, b, h0):
    """Explicit triple loop over (batch, time, channel) in NumPy."""
    a, b, h0 = a.numpy(), b.numpy(), h0.numpy()
    out = np.zeros_like(b)
    for i in range(a.shape[0]):
        for d in range(a.shape[2]):
            for n in range(a.shape[3]):
                h = h0[i, d, n]
                for t in range(a.shape[1]):
                    h = a[i, t, d, n] * h + b[i, t, d, n]
                    out[i, t, d, n] = h
    return out


@pytest.mark.parametrize("L", [1, 2, 7, 16, 37, 64, 100])
@pytest.mark.parametrize("name", list(SCANS))
@pytest.mark.parametrize("with_h0", [False, True])
def test_scans_match_sequential(name, L, with_h0):
    a, b = rand_ab(L=L)
    h0 = torch.randn(2, 3, 4, dtype=DT) if with_h0 else None
    ref = scan_sequential(a, b, h0)
    torch.testing.assert_close(SCANS[name](a, b, h0), ref, rtol=1e-10, atol=1e-10)


def test_sequential_matches_numpy_bruteforce():
    a, b = rand_ab(L=25)
    h0 = torch.randn(2, 3, 4, dtype=DT)
    np.testing.assert_allclose(scan_sequential(a, b, h0).numpy(), numpy_bruteforce(a, b, h0), rtol=1e-12)


@pytest.mark.parametrize("chunk", [1, 3, 8, 64])
def test_chunked_any_chunk_size(chunk):
    from models.scan import scan_chunked
    a, b = rand_ab(L=50)
    torch.testing.assert_close(scan_chunked(a, b, None, chunk), scan_sequential(a, b))


def selective_inputs(B=2, L=19, D=5, N=4, h0=False):
    x = torch.randn(B, L, D, dtype=DT, requires_grad=True)
    delta = (torch.rand(B, L, D, dtype=DT) * 0.5 + 0.01).requires_grad_()
    A = (-torch.rand(D, N, dtype=DT) * 2 - 0.1).requires_grad_()
    Bm = torch.randn(B, L, N, dtype=DT, requires_grad=True)
    Cm = torch.randn(B, L, N, dtype=DT, requires_grad=True)
    hz = torch.randn(B, D, N, dtype=DT, requires_grad=True) if h0 else None
    return x, delta, A, Bm, Cm, hz


@pytest.mark.parametrize("scan", ["sequential", "hillis_steele", "blelloch", "chunked"])
@pytest.mark.parametrize("method", DISCRETIZATIONS)
@pytest.mark.parametrize("h0", [False, True])
def test_custom_backward_matches_autograd(scan, method, h0):
    inputs = selective_inputs(h0=h0)
    leaves = [t for t in inputs if t is not None]
    y_ref, last_ref = selective_scan_reference(*inputs, method=method)
    w1, w2 = torch.randn_like(y_ref), torch.randn_like(last_ref)
    g_ref = torch.autograd.grad((y_ref * w1).sum() + (last_ref * w2).sum(), leaves)

    y, last = selective_scan(*inputs, method=method, scan=scan, d_block=2)
    torch.testing.assert_close(y, y_ref)
    torch.testing.assert_close(last, last_ref)
    g = torch.autograd.grad((y * w1).sum() + (last * w2).sum(), leaves)
    for gi, gr in zip(g, g_ref):
        torch.testing.assert_close(gi, gr)


def test_gradcheck_selective_scan():
    x, delta, A, Bm, Cm, h0 = selective_inputs(B=1, L=6, D=2, N=3, h0=True)
    fn = lambda *t: selective_scan(*t, method="zoh", scan="chunked")[0]
    assert torch.autograd.gradcheck(fn, (x, delta, A, Bm, Cm, h0))


def test_diagonal_zoh_matches_dense_matrix_exponential():
    D, N = 3, 5
    A = -torch.rand(D, N, dtype=DT) * 3 - 0.1
    Bv = torch.randn(N, dtype=DT)
    delta = torch.rand(D, dtype=DT) * 0.7
    x = torch.randn(D, dtype=DT)
    a, b = discretize(delta.view(1, 1, D), A, Bv.view(1, 1, N), x.view(1, 1, D), "zoh")
    for d in range(D):
        Ad, Bd = zoh_matrix(torch.diag(A[d]), Bv.unsqueeze(1), float(delta[d]))
        torch.testing.assert_close(a[0, 0, d], torch.diagonal(Ad))
        torch.testing.assert_close(b[0, 0, d], Bd.squeeze(1) * x[d])


def test_zoh_is_exact_for_piecewise_constant_input():
    """Integrate dh/dt = A h + B u with RK4 over one step; ZOH must match the true solution."""
    A = torch.tensor([[-1.0, 0.5], [-0.3, -2.0]], dtype=DT)       # dense, non-symmetric
    B = torch.tensor([[1.0], [0.5]], dtype=DT)
    h = torch.tensor([[0.2], [-0.4]], dtype=DT)
    u, delta, steps = 0.7, 0.3, 2000
    f = lambda s: A @ s + B * u
    s, dt = h.clone(), delta / steps
    for _ in range(steps):
        k1 = f(s); k2 = f(s + dt / 2 * k1); k3 = f(s + dt / 2 * k2); k4 = f(s + dt * k3)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    Ad, Bd = zoh_matrix(A, B, delta)
    torch.testing.assert_close(Ad @ h + Bd * u, s, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("method", ["euler", "bilinear"])
def test_discretizations_converge_to_zoh(method):
    A = -torch.rand(4, 3, dtype=DT) - 0.5
    x, Bm = torch.randn(1, 1, 4, dtype=DT), torch.randn(1, 1, 3, dtype=DT)
    errs = []
    for dt in (1e-1, 1e-2, 1e-3):
        delta = torch.full((1, 1, 4), dt, dtype=DT)
        za, zb = discretize(delta, A, Bm, x, "zoh")
        ma, mb = discretize(delta, A, Bm, x, method)
        errs.append(((za - ma).abs().max() + (zb - mb).abs().max()).item())
    assert errs[0] > errs[1] > errs[2] and errs[2] < 1e-5
