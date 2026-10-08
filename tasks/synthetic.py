"""Synthetic sequence tasks that isolate specific capabilities.

All generators return (x, y) LongTensors of shape (B, L); y = -100 where no prediction is scored,
so the loss/accuracy at position t asks "what token should the model output after reading x[:t+1]?".

MQAR (multi-query associative recall; Arora et al. 2023, "Zoology")
    k1 v1 k2 v2 ... kn vn  0 0 ... 0  k3 v3 k1 v1 ...    -> after each queried key, output its value
    Tests *recall capacity*: attention can look the pair up exactly; a fixed-size state must
    compress all n pairs, so accuracy should fall as n grows relative to the state size.

Selective copying (Gu & Dao 2023, Mamba paper, Sec. 3.1)
    . . 7 . . . 3 . 9 . . . 1 . .  # # # #            -> after each marker, output the data tokens in order
    Tests *selection*: the data tokens sit at random positions among noise, so a model has to decide
    per token whether to store it.  A linear time-invariant SSM (fixed Δ, B, C) cannot do this.
"""

import torch

IGNORE = -100


def mqar(batch, seq_len, n_kv, vocab=128, generator=None):
    """Keys from [1, vocab/2), values from [vocab/2, vocab), filler token 0.

    Layout:  k1 v1 ... kn vn | 0 0 ... 0 | kπ1 vπ1 ... kπn vπn
    The n key-value pairs come first; the same pairs are queried in random order at the end of
    the sequence, so the gap between storing and recalling grows with seq_len (used for the
    length-extrapolation test).  The model must output vπi after reading kπi.
    """
    g = generator
    half = vocab // 2
    assert 4 * n_kv <= seq_len and n_kv < half
    keys = torch.argsort(torch.rand(batch, half - 1, generator=g), dim=1)[:, :n_kv] + 1
    vals = torch.randint(half, vocab, (batch, n_kv), generator=g)
    order = torch.argsort(torch.rand(batch, n_kv, generator=g), dim=1)
    qk, qv = keys.gather(1, order), vals.gather(1, order)
    x = torch.zeros(batch, seq_len, dtype=torch.long)
    y = torch.full((batch, seq_len), IGNORE, dtype=torch.long)
    x[:, 0:2 * n_kv:2] = keys
    x[:, 1:2 * n_kv:2] = vals
    q0 = seq_len - 2 * n_kv
    x[:, q0::2] = qk
    x[:, q0 + 1::2] = qv
    y[:, q0::2] = qv
    return x, y


def selective_copying(batch, seq_len=256, n_tokens=16, n_colors=16, generator=None):
    """Tokens: 0 = noise, 1..n_colors = data, n_colors+1 = marker.  Total length seq_len + n_tokens."""
    g = generator
    marker = n_colors + 1
    data = torch.randint(1, n_colors + 1, (batch, n_tokens), generator=g)
    pos = torch.sort(torch.argsort(torch.rand(batch, seq_len, generator=g), dim=1)[:, :n_tokens], dim=1).values
    x = torch.zeros(batch, seq_len + n_tokens, dtype=torch.long)
    rows = torch.arange(batch)[:, None]
    x[rows, pos] = data
    x[:, seq_len:] = marker
    y = torch.full_like(x, IGNORE)
    # after reading the k-th marker (position seq_len + k) the model must emit the k-th data token
    y[:, seq_len:] = data
    return x, y


TASK_VOCAB = {"mqar": 128, "selective_copying": 18}
