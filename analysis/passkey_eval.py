"""Passkey recall at gaps shorter and longer than the training length (stage 5b).

    python -m analysis.passkey_eval --run_dir results/research/<experiment>/<run>

Writes <run_dir>/passkey.json: for each gap, the fraction of recalled key bytes predicted exactly (teacher-forced
argmax), exact-match rate of whole keys, and bits per recalled byte. Held-out text comes from the validation split.
Chance is 1/26 per byte (random lowercase letters).
"""

import argparse
import json
import math
import os

import torch
import torch.nn.functional as F

from analysis.length_generalization import load
from data.enwik8 import Enwik8
from data.passkey import passkey_batch
from runtime import environment, setup_device


@torch.no_grad()
def evaluate(model, text, gap, key_len, n, device, seed=0):
    g = torch.Generator().manual_seed(seed + gap)
    bs = max(1, min(64, 65536 // (gap + 128)))
    hits = keys_ok = total = 0
    nll = 0.0
    while total < n:
        b = min(bs, n - total)
        x, y, m = passkey_batch(text, b, gap, key_len, g)
        x, y, m = x.to(device), y.to(device), m.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(x).float()
        pred_ok = (logits.argmax(-1) == y) & m
        hits += pred_ok.sum().item()
        keys_ok += (pred_ok.sum(1) == key_len).sum().item()
        nll += F.cross_entropy(logits[m], y[m], reduction="sum").item()
        total += b
    return {"byte_acc": hits / (total * key_len), "key_exact": keys_ok / total,
            "bits_per_key_byte": nll / (total * key_len) / math.log(2), "n": total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--max_gap", type=int, default=16384)
    ap.add_argument("--n", type=int, default=256)
    args = ap.parse_args()
    device = setup_device(verbose=False)
    cfg = json.load(open(os.path.join(args.run_dir, "config.json")))
    L, key_len = cfg["args"]["seq_len"], cfg["args"].get("passkey_len") or 8
    model = load(args.run_dir, device)
    text = Enwik8().splits["valid"]
    gaps = [L // 2] + [L * 2 ** k for k in range(10) if L * 2 ** k <= args.max_gap]
    out = {"train_len": L, "key_len": key_len, "chance_byte_acc": 1 / 26, "by_gap": {}}
    for gap in gaps:
        try:
            out["by_gap"][gap] = evaluate(model, text, gap, key_len, args.n if gap <= 4 * L else args.n // 2, device)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            out["by_gap"][gap] = None
        print(f"  gap {gap:6d}: {out['by_gap'][gap]}", flush=True)
    out["environment"] = environment()
    with open(os.path.join(args.run_dir, "passkey.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {args.run_dir}/passkey.json")


if __name__ == "__main__":
    main()
