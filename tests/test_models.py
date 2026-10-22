"""Model-level correctness: causality, cache/state decoding == full forward, equivalences."""

import pytest
import torch
import torch.nn.functional as F

from models.lm import LM, LMConfig, build
from models.retention import Retention

torch.manual_seed(0)
SMALL = dict(d_model=32, n_head=4, d_state=8)
ARCHS = {
    "transformer": "AA",
    "mamba": "MMM",
    "retnet": "RR",
    "hybrid": "MAM",
}


def small(layers, **kw):
    torch.manual_seed(0)
    return LM(LMConfig(layers=layers, **{**SMALL, **kw})).double().eval()


@pytest.mark.parametrize("name", ARCHS)
def test_causality(name):
    m = small(ARCHS[name])
    x = torch.randint(0, 256, (2, 40))
    x2 = x.clone()
    x2[:, 25:] = torch.randint(0, 256, (2, 15))
    with torch.no_grad():
        torch.testing.assert_close(m(x)[:, :25], m(x2)[:, :25])
        assert not torch.allclose(m(x)[:, 25:], m(x2)[:, 25:])


@pytest.mark.parametrize("name", ARCHS)
def test_incremental_decoding_matches_full_forward(name):
    """KV cache (attention), recurrent SSM state (Mamba), recurrent retention state."""
    m = small(ARCHS[name])
    x = torch.randint(0, 256, (2, 30))
    with torch.no_grad():
        full = m(x)
        states = m.init_state(2, 30, x.device, torch.float64)
        parts = [m(x[:, :11], states, 0)]                  # prefill 11 tokens
        parts.append(m(x[:, 11:20], states, 11))           # chunked prefill of 9 more
        for t in range(20, 30):                            # then one token at a time
            parts.append(m(x[:, t:t + 1], states, t))
    torch.testing.assert_close(torch.cat(parts, 1), full)


@pytest.mark.parametrize("scan", ["sequential", "hillis_steele", "blelloch", "chunked", "reference"])
def test_mamba_scan_modes_agree(scan):
    x = torch.randint(0, 256, (2, 33))
    ref = small("MM", scan="sequential")
    m = small("MM", scan=scan, d_block=24)
    with torch.no_grad():
        torch.testing.assert_close(m(x), ref(x))


def test_mamba_lti_ablation_has_constant_delta():
    m = small("M", selective=False)
    mixer = m.blocks[0].mixer
    u = torch.randn(2, 10, 32, dtype=torch.float64)
    _, delta = mixer(u, return_delta=True)
    assert torch.allclose(delta, delta[:, :1].expand_as(delta))


def test_retention_forms_agree():
    torch.manual_seed(0)
    r = Retention(32, 4, chunk_size=8).double()
    x = torch.randn(2, 45, 32, dtype=torch.float64)
    torch.testing.assert_close(r(x), r.forward_parallel(x))


def test_attention_manual_matches_sdpa():
    x = torch.randint(0, 256, (2, 24))
    a, b = small("AA", attn_impl="manual"), small("AA", attn_impl="sdpa")
    with torch.no_grad():
        torch.testing.assert_close(a(x), b(x))


def test_state_size_constant_for_ssm_and_linear_for_kv():
    for layers, grows in (("AA", True), ("MM", False), ("RR", False)):
        m = small(layers)
        s1 = LM.state_nbytes(m.init_state(1, 100, "cpu"))
        s2 = LM.state_nbytes(m.init_state(1, 1000, "cpu"))
        assert (s2 == 10 * s1) if grows else (s1 == s2)


@pytest.mark.parametrize("name", ["transformer", "mamba", "retnet", "hybrid"])
def test_presets_are_parameter_matched(name):
    ref = build("transformer").num_params()
    assert abs(build(name).num_params() / ref - 1) < 0.03


@pytest.mark.parametrize("name", ARCHS)
def test_can_overfit_a_batch(name):
    torch.manual_seed(0)
    m = LM(LMConfig(layers=ARCHS[name], **SMALL))
    x = torch.randint(0, 256, (4, 33))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    for step in range(150):
        loss = F.cross_entropy(m(x[:, :-1]).reshape(-1, 256), x[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
    assert first > 5.0 and loss.item() < 0.5, (first, loss.item())


@pytest.mark.parametrize("pos", ["rope", "none", "alibi"])
def test_position_encodings_causal_and_cache_consistent(pos):
    """NoPE and ALiBi baselines: causal, and KV-cache decoding equals the full forward pass."""
    m = small("AA", pos_encoding=pos)
    x = torch.randint(0, 256, (2, 30))
    x2 = x.clone()
    x2[:, 20:] = torch.randint(0, 256, (2, 10))
    with torch.no_grad():
        full = m(x)
        torch.testing.assert_close(m(x2)[:, :20], full[:, :20])
        states = m.init_state(2, 30, x.device, torch.float64)
        parts = [m(x[:, :13], states, 0), m(x[:, 13:21], states, 13)]
        parts += [m(x[:, t:t + 1], states, t) for t in range(21, 30)]
    torch.testing.assert_close(torch.cat(parts, 1), full)


@pytest.mark.parametrize("pos", ["rope", "none", "alibi"])
@pytest.mark.parametrize("impl", ["chunked", "sdpa"])
def test_attention_implementations_agree(pos, impl):
    x = torch.randint(0, 256, (2, 45))
    ref = small("AA", pos_encoding=pos, attn_impl="manual")
    other = small("AA", pos_encoding=pos, attn_impl=impl, attn_query_chunk=8)
    with torch.no_grad():
        torch.testing.assert_close(other(x), ref(x))


def test_alibi_penalizes_distance():
    from models.transformer import alibi_slopes
    s = alibi_slopes(4)
    assert torch.all(s > 0) and torch.all(s[1:] < s[:-1])
