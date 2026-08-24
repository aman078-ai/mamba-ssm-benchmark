"""enwik8 byte-level language modeling dataset.

Usage:
    from data.enwik8 import Enwik8
    data = Enwik8(root='data/enwik8', train=True, seqlen=256)
    x, y = data.get()  # returns (batch, seq_len) tensors
"""

import os
import urllib.request
import gzip
import numpy as np
import torch
from torch.utils.data import Dataset


URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/enwik8"
FNAME = "enwik8.gz"


def _download(root: str):
    """Download enwik8 if not present. Falls back to synthetic data if URL fails."""
    os.makedirs(root, exist_ok=True)
    fpath = os.path.join(root, FNAME)
    if os.path.exists(fpath):
        return fpath
    # Try downloading
    try:
        print(f"Downloading enwik8 from {URL}...")
        urllib.request.urlretrieve(URL + "/" + FNAME, fpath)
        print("Download complete.")
        return fpath
    except Exception:
        print("Download failed. Generating synthetic byte data...")
        # Generate 1MB of synthetic byte data for testing
        synthetic = np.random.randint(0, 256, size=1_000_000, dtype=np.uint8)
        np.savez_compressed(os.path.join(root, "synthetic.npz"), data=synthetic)
        return fpath.replace(FNAME, "synthetic.npz")


def _load(enwik8_path: str):
    """Load enwik8, return train/valid/test as numpy arrays of uint8."""
    if enwik8_path.endswith("synthetic.npz"):
        # Synthetic fallback
        arr = np.load(enwik8_path)
        data = arr["data"]
        # Split: 900k train, 50k valid, 50k test
        n = len(data)
        train_data = data[:900_000]
        valid_data = data[900_000:950_000]
        test_data = data[950_000:n]
        return train_data, valid_data, test_data
    with gzip.open(enwik8_path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    n = len(data)
    train_data = data[:90_000_000]
    valid_data = data[90_000_000:95_000_000]
    test_data = data[95_000_000:n]
    return train_data, valid_data, test_data


class Enwik8:
    """Byte-level LM dataset over enwik8.

    Returns minibatches of shape (batch_size, seqlen).
    """

    def __init__(self, root: str = "data/enwik8", train: bool = True,
                 seqlen: int = 256, batch_size: int = 1):
        self.root = root
        self.train = train
        self.seqlen = seqlen
        self.batch_size = batch_size

        enwik8_path = _download(root)
        train_data, valid_data, test_data = _load(enwik8_path)

        self.data = train_data if self.train else valid_data
        print(f"[enwik8] {self.train=}, data shape: {self.data.shape}, "
              f"seqlen={self.seqlen}, batch_size={self.batch_size}")

        # Map bytes (0-255) to indices
        self.vocab_size = 256

    def get(self):
        """Get next minibatch of shape (batch_size, seqlen+1).
        Returns x, y where y is x shifted by 1 (next-token prediction).
        After calling get() enough times, reset to iterate from start.
        """
        # We need seqlen+1 tokens: x -> y next-token
        buf_len = self.seqlen + 1
        total_tokens = len(self.data)

        # If we've consumed all data, reset
        if not hasattr(self, '_idx'):
            self._idx = 0

        # Check if we need to wrap around
        remaining = total_tokens - self._idx
        if remaining < buf_len:
            # Wrap: take what's left, then from the beginning
            # Reset for next call
            self._idx = 0
            remaining = total_tokens

        # Grab a chunk a bit larger than needed to handle batching
        chunk = self.data[self._idx:self._idx + buf_len * self.batch_size]
        self._idx += buf_len * self.batch_size

        # Reshape to (batch_size, seqlen+1)
        # If chunk is smaller than needed (end of data), zero-pad
        needed = self.batch_size * buf_len
        if len(chunk) < needed:
            # Pad with zeros
            pad_len = needed - len(chunk)
            chunk = np.pad(chunk, (0, pad_len), mode='constant')
        else:
            # Trim to exact needed size
            chunk = chunk[:needed]

        chunk = chunk.reshape(self.batch_size, buf_len).astype(np.int64)
        x = torch.from_numpy(chunk[:, :self.seqlen])   # input
        y = torch.from_numpy(chunk[:, 1:])              # target (shifted)
        return x, y

    def iter_batches(self):
        """Infinite iterator over (x, y) pairs of shape (batch, seqlen)."""
        while True:
            yield self.get()