"""Evaluate trained enwik8 models.

1. Test bpc on the full 5 MB test split at the training length (512).
2. Length extrapolation: bpc on the first 2 MB of test at context lengths 512 ... 8192.
3. Loss vs. position inside a 4096-byte context (does the model use, or break on, long context?).

    python -m analysis.evaluate_lm
"""

import argparse
import json
import math
import os

import torch
import torch.nn.functional as F

from data.enwik8 import Enwik8
from models.lm import LM, LMConfig
from runtime import setup_device
from training.train_lm import evaluate_bpc

LN2 = math.log(2)


def load(run_dir, device):
    ck = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu")
    model = LM(LMConfig(**ck["cfg"]))
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def position_loss(model, ds, seq_len, device, max_bytes, n_bins=64):
    """Mean loss (bpc) for each position bin inside windows of seq_len."""
    sums = torch.zeros(seq_len, device=device)
    count = 0
    for x, y in ds.eval_batches("test", 1, seq_len, device, max_bytes):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        sums += F.cross_entropy(logits.float()[0], y[0], reduction="none")
        count += 1
    per_pos = (sums / count / LN2).cpu()
    edges = torch.logspace(0, math.log10(seq_len), n_bins + 1).long().unique()
    bins = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    return [{"start": a, "end": b, "bpc": per_pos[a:b].mean().item()} for a, b in bins]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="transformer,mamba,retnet,hybrid")
    ap.add_argument("--root", default="results/enwik8")
    ap.add_argument("--lengths", default="512,1024,2048,4096,8192")
    ap.add_argument("--extrap_bytes", type=int, default=2_000_000)
    ap.add_argument("--position_len", type=int, default=4096)
    ap.add_argument("--position_bytes", type=int, default=1_000_000)
    args = ap.parse_args(argv)
    device = setup_device()
    ds = Enwik8()
    out_path = os.path.join(args.root, "eval.json")
    results = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for run in args.runs.split(","):
        run_dir = os.path.join(args.root, run)
        if not os.path.exists(os.path.join(run_dir, "model.pt")):
            print(f"skip {run}: no checkpoint")
            continue
        model = load(run_dir, device)
        rec = {"params": model.num_params()}
        rec["test_bpc"] = evaluate_bpc(model, ds, "test", 512, 32, device)
        print(f"[{run}] test bpc @512 = {rec['test_bpc']:.4f}", flush=True)
        rec["bpc_by_length"] = {}
        for L in (int(v) for v in args.lengths.split(",")):
            bpc, bs = None, max(1, 16384 // L)
            while bs >= 1 and bpc is None:      # halve the batch on OOM; None if even batch 1 fails
                try:
                    bpc = evaluate_bpc(model, ds, "test", L, bs, device, args.extrap_bytes)
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    bs //= 2
            rec["bpc_by_length"][L] = bpc
            print(f"[{run}]   bpc @{L} = {bpc}", flush=True)
        rec["position_loss"] = position_loss(model, ds, args.position_len, device, args.position_bytes)
        results[run] = rec
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        del model
        torch.cuda.empty_cache()
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
