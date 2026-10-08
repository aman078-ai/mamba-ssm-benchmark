"""enwik8 byte-level language modeling data.

enwik8 is the first 10^8 bytes of an English Wikipedia XML dump (Hutter Prize).
Standard split (Al-Rfou et al., 2018; Dai et al., 2019): first 90M bytes train,
next 5M validation, last 5M test. Vocabulary = 256 byte values. Metric = bits
per character (bpc) = mean cross-entropy in nats / ln(2).

Usage:
    from data.enwik8 import Enwik8
    ds = Enwik8()                                  # downloads + caches on first use
    x, y = ds.random_batch('train', batch_size=8, seq_len=512, device='cuda')
    for x, y in ds.eval_batches('test', batch_size=8, seq_len=512): ...
"""

import os
import urllib.request
import zipfile

import numpy as np
import torch

URL = "http://mattmahoney.net/dc/enwik8.zip"
N_TRAIN, N_VALID, N_TEST = 90_000_000, 5_000_000, 5_000_000
DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "enwik8")


def prepare(root: str = DEFAULT_ROOT) -> str:
    """Download enwik8 (if needed) and write train/valid/test .bin files.

    Raises on failure instead of silently substituting fake data.
    """
    os.makedirs(root, exist_ok=True)
    if all(os.path.exists(os.path.join(root, f"{s}.bin")) for s in ("train", "valid", "test")):
        return root
    zpath = os.path.join(root, "enwik8.zip")
    if not os.path.exists(zpath):
        print(f"[enwik8] downloading {URL} ...")
        urllib.request.urlretrieve(URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        raw = np.frombuffer(z.read("enwik8"), dtype=np.uint8)
    if raw.size != N_TRAIN + N_VALID + N_TEST:
        raise RuntimeError(f"enwik8 has {raw.size} bytes, expected 100,000,000")
    splits = {
        "train": raw[:N_TRAIN],
        "valid": raw[N_TRAIN:N_TRAIN + N_VALID],
        "test": raw[N_TRAIN + N_VALID:],
    }
    for name, arr in splits.items():
        arr.tofile(os.path.join(root, f"{name}.bin"))
    print(f"[enwik8] wrote train/valid/test .bin to {root}")
    return root


class Enwik8:
    vocab_size = 256

    def __init__(self, root: str = DEFAULT_ROOT):
        prepare(root)
        self.splits = {
            s: np.memmap(os.path.join(root, f"{s}.bin"), dtype=np.uint8, mode="r")
            for s in ("train", "valid", "test")
        }

    def random_batch(self, split, batch_size, seq_len, device="cpu", generator=None):
        """Uniformly sampled contiguous windows. Returns x, y of shape (B, T)."""
        data = self.splits[split]
        hi = len(data) - seq_len - 1
        starts = torch.randint(0, hi, (batch_size,), generator=generator).numpy()
        buf = np.stack([data[s:s + seq_len + 1] for s in starts]).astype(np.int64)
        buf = torch.from_numpy(buf)
        if device != "cpu":
            buf = buf.pin_memory().to(device, non_blocking=True)
        return buf[:, :-1], buf[:, 1:]

    def eval_batches(self, split, batch_size, seq_len, device="cpu", max_bytes=None):
        """Non-overlapping windows covering the split (or its first max_bytes)."""
        data = self.splits[split]
        n = len(data) if max_bytes is None else min(len(data), max_bytes + 1)
        n_windows = (n - 1) // seq_len
        for i in range(0, n_windows, batch_size):
            idx = range(i, min(i + batch_size, n_windows))
            buf = np.stack([data[j * seq_len: j * seq_len + seq_len + 1] for j in idx]).astype(np.int64)
            buf = torch.from_numpy(buf).to(device)
            yield buf[:, :-1], buf[:, 1:]

    def text(self, split, start, length):
        return bytes(self.splits[split][start:start + length]).decode("utf-8", errors="replace")
