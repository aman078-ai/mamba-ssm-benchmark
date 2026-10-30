"""Passkey insertion places STORE and RECALL blocks correctly and masks exactly the recalled bytes."""

import numpy as np
import torch

from data.passkey import RECALL, STORE, insert_passkey, passkey_batch


def test_insert_passkey_layout():
    g = torch.Generator().manual_seed(0)
    buf = torch.full((4, 101), 100, dtype=torch.long)
    mask = insert_passkey(buf, key_len=5, gap_min=20, gap_max=60, generator=g)
    for i in range(4):
        r0 = 101 - 6
        assert buf[i, r0] == RECALL
        s0 = int((buf[i] == STORE).nonzero()[0])
        gap = r0 - s0
        assert 20 <= gap <= 60
        assert torch.equal(buf[i, s0 + 1:s0 + 6], buf[i, r0 + 1:r0 + 6])        # same key stored and recalled
        y = buf[i, 1:]
        assert torch.equal(y[mask[i]], buf[i, r0 + 1:r0 + 6])                    # mask = recalled key bytes
        assert mask[i].sum() == 5


def test_passkey_batch_gap():
    text = np.random.default_rng(0).integers(32, 127, 10_000).astype(np.uint8)
    x, y, mask = passkey_batch(text, 3, gap=500, key_len=8, generator=torch.Generator().manual_seed(1))
    assert x.shape == y.shape == mask.shape and mask.sum() == 3 * 8
    for i in range(3):
        s0 = int((x[i] == STORE).nonzero()[0]); r0 = int((x[i] == RECALL).nonzero()[0])
        assert r0 - s0 == 500 + 9


def test_insert_passkey_random_end():
    g = torch.Generator().manual_seed(0)
    buf = torch.full((64, 201), 100, dtype=torch.long)
    mask = insert_passkey(buf, key_len=5, gap_min=20, gap_max=60, generator=g, random_end=True)
    ends = set()
    for i in range(64):
        s0 = int((buf[i] == STORE).nonzero()[0]); r0 = int((buf[i] == RECALL).nonzero()[0])
        assert 20 <= r0 - s0 <= 60 and s0 >= 0 and r0 + 6 <= 201
        assert torch.equal(buf[i, s0 + 1:s0 + 6], buf[i, r0 + 1:r0 + 6])
        assert torch.equal(buf[i, 1:][mask[i]], buf[i, r0 + 1:r0 + 6])
        ends.add(r0)
    assert len(ends) > 20                                                        # position is not fixed


def test_insert_passkey_stream():
    from data.passkey import insert_passkey_stream
    g = torch.Generator().manual_seed(0)
    T1, K = 201, 5
    buf = torch.full((400, T1), 100, dtype=torch.long)
    mask = insert_passkey_stream(buf, key_len=K, gap_min=10, gap_max=150, p_present=0.5, generator=g)
    present = (buf == STORE).any(1)
    assert 0.4 < present.float().mean() < 0.6                                    # absent from some windows
    recall_pos, truncated = [], 0
    for i in range(400):
        if not present[i]:
            assert not mask[i].any() and not (buf[i] == RECALL).any()
            continue
        s0 = int((buf[i] == STORE).nonzero()[0])
        r = (buf[i] == RECALL).nonzero()
        if len(r) == 0:
            truncated += 1
            assert not mask[i].any()
            continue
        r0 = int(r[0]); recall_pos.append(r0)
        assert 10 <= r0 - s0 <= 150
        n = min(K, T1 - r0 - 1)
        assert torch.equal(buf[i, r0 + 1:r0 + 1 + n], buf[i, s0 + 1:s0 + 1 + n])  # recalled bytes = stored key
        assert int(mask[i].sum()) == n and torch.equal(buf[i, 1:][mask[i]], buf[i, r0 + 1:r0 + 1 + n])
    assert truncated > 0                                                         # RECALL may fall outside window
    assert min(recall_pos) < 60 and max(recall_pos) > 190                        # not tied to the window end
