"""Passkey insertion: force a language model to hold information over a long gap (stage 5b).

Each training window gets   ... [STORE] k1 .. kK ... (gap of ordinary text) ... [RECALL] k1 .. kK
with K random lowercase letters.  The bytes after RECALL are predictable only from memory of the STORE block, so a
model that minimizes the ordinary next-byte loss is pushed to keep a long-horizon memory.  Bytes 1 and 2 never occur
in enwik8, so the markers are unambiguous.
"""

import torch

STORE, RECALL = 1, 2


def insert_passkey(buf, key_len, gap_min, gap_max, generator=None, random_end=False):
    """buf: (B, T+1) int64 windows (modified in place).  The RECALL block ends the window; the STORE block starts
    `gap` bytes before it, gap ~ U[gap_min, gap_max].  Returns the boolean mask over targets y = buf[:, 1:] marking
    the recalled key bytes.

    With random_end=False the RECALL marker always sits at the same absolute position, which a model can learn to
    predict from position alone (stage 5b found Mamba does exactly that).  random_end=True places the RECALL block
    uniformly among the positions that leave room for the gap, so its position carries no information."""
    B, T1 = buf.shape
    block = key_len + 1
    r_last = T1 - block                                               # latest possible RECALL marker position
    gap_max = min(gap_max, r_last - block)
    assert gap_min <= gap_max, "window too short for this gap range"
    keys = torch.randint(ord("a"), ord("z") + 1, (B, key_len), generator=generator)
    gaps = torch.randint(gap_min, gap_max + 1, (B,), generator=generator)
    mask = torch.zeros(B, T1 - 1, dtype=torch.bool)
    for i in range(B):
        r0 = r_last
        if random_end:
            r0 = int(torch.randint(int(gaps[i]) + block, r_last + 1, (1,), generator=generator))
        s0 = r0 - int(gaps[i])
        buf[i, s0] = STORE
        buf[i, s0 + 1:s0 + block] = keys[i]
        buf[i, r0] = RECALL
        buf[i, r0 + 1:r0 + block] = keys[i]
        mask[i, r0:r0 + key_len] = True                               # y[t] = buf[t + 1]
    return mask


def passkey_batch(text, batch, gap, key_len, generator=None, prefix=64):
    """Evaluation sequences: `prefix` bytes of text, STORE block, `gap` bytes of text, RECALL block.
    text: 1-D uint8 array of held-out text.  Returns x, y, mask (recalled key bytes)."""
    block = key_len + 1
    T1 = prefix + block + gap + block
    starts = torch.randint(0, len(text) - T1 - 1, (batch,), generator=generator)
    buf = torch.stack([torch.as_tensor(text[int(s):int(s) + T1].astype("int64")) for s in starts])
    mask = insert_passkey(buf, key_len, gap + block, gap + block, generator)
    return buf[:, :-1], buf[:, 1:], mask
