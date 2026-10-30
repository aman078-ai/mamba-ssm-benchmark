"""Confirm the headline numbers of the main study on the held-out TEST split (never used for tuning or analysis).

    python -m analysis.test_split_check

For every run of the main study (stage4_pilot, stage4b_scalar_decay, stage5a_seeds): bits per byte at the training
length and the same-bytes context effect (8 windows of 16K bytes; the validation numbers use 16), both on the test
split.  Several processes may run at once (e.g. `--only mamba`); results are merged into one file.  Writes
results/research/test_split.json and results/research/test_split.md (mean and 95% t-interval over seeds).
"""

import glob
import json
import os
import argparse
import re

import torch

from analysis.aggregate_seeds import BANDS, EXPERIMENTS, fmt, stats
from analysis.length_generalization import bpc, context_effect, load
from analysis.research_plots import family
from data.enwik8 import Enwik8
from runtime import environment, setup_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="substring filter on run names, e.g. 'mamba'")
    ap.add_argument("--windows", type=int, default=8)
    args = ap.parse_args()
    device = setup_device(verbose=False)
    ds = Enwik8()
    path = os.path.join("results", "research", "test_split.json")

    def saved():
        return json.load(open(path))["runs"] if os.path.exists(path) else {}
    runs = saved()
    for exp in EXPERIMENTS:
        for cfg_path in sorted(glob.glob(os.path.join("results", "research", exp, "*", "config.json"))):
            d = os.path.dirname(cfg_path)
            key = f"{exp}/{os.path.basename(d)}"
            if args.only not in key or key in saved() or not os.path.exists(os.path.join(d, "model.pt")):
                continue
            cfg = json.load(open(cfg_path))
            L = cfg["args"]["seq_len"]
            model = load(d, device)
            with torch.no_grad():
                b = bpc(model, ds, L, 1_000_000, device, split="test")
                ce = context_effect(model, ds, L, 16384, args.windows, device, split="test")
            runs = saved()                                  # merge with results written by other processes
            runs[key] = {"name": re.sub(r"_seed\d+$", "", os.path.basename(d)), "family": family(cfg),
                         "train_len": L, "seed": cfg["args"]["seed"], "bpc_at_train_len": b, "context_effect": ce,
                         "windows": args.windows}
            print(key, round(b, 4), {k: round(v, 4) for k, v in ce.items()}, flush=True)
            json.dump({"split": "test", "runs": runs, "environment": environment()}, open(path, "w"), indent=2)
            del model
            torch.cuda.empty_cache()
    runs = saved()
    groups = {}
    for r in runs.values():
        groups.setdefault(r["name"], []).append(r)
    lines = ["| configuration | training length | bpc at training length (test) | context effect 2–4× (test) | "
             "context effect 8–16× (test) |", "|---|---|---|---|---|"]
    for name, rs in sorted(groups.items(), key=lambda kv: (kv[1][0]["train_len"], kv[1][0]["family"])):
        ce = {k: stats([r["context_effect"].get(k) for r in rs]) for k in BANDS}
        lines.append(f"| {rs[0]['family']} | {rs[0]['train_len']} | {fmt(stats([r['bpc_at_train_len'] for r in rs]))} | "
                     f"{fmt(ce['2-4x'], signed=True)} | {fmt(ce['8-16x'], signed=True)} |")
    open(os.path.join("results", "research", "test_split.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
