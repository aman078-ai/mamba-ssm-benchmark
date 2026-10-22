"""Run an experiment grid from a JSON config.  Resumable: runs whose summary.json exists are skipped.

    python -m experiments.run configs/stage3_lr_sweep.json
    python -m experiments.run configs/stage3_lr_sweep.json --dry-run      # list the runs only
    python -m experiments.run configs/stage3_lr_sweep.json --only mamba   # runs whose name contains "mamba"

Config format:
    {
      "name": "...", "out": "results/research/<name>",
      "common":   {train_lm arguments shared by every run, e.g. "steps": 3000},
      "variants": {"<variant>": {"model": "<preset>", "override": {LMConfig fields}, ...other train_lm args}},
      "grid":     {"<key>": [values, ...], ...},            # applied to every variant (cartesian product)
      "post":     [["-m", "analysis.length_generalization", "--run_dir", "{run_dir}"], ...]   # optional, per run
    }
Grid keys that are LMConfig fields (e.g. d_state) go into the model override; the rest (e.g. lr, seed, seq_len) are
train_lm arguments.  Each run executes in a fresh process so memory caps and seeds start clean.
"""

import argparse
import dataclasses
import itertools
import json
import os
import subprocess
import sys

from models.lm import LMConfig

CFG_FIELDS = {f.name for f in dataclasses.fields(LMConfig)}


def fmt(v):
    return f"{v:g}" if isinstance(v, float) else str(v)


def expand(cfg):
    grid = cfg.get("grid", {})
    keys = list(grid)
    varying = [k for k in keys if len(grid[k]) > 1]
    for vname, vargs in cfg["variants"].items():
        for combo in itertools.product(*(grid[k] for k in keys)):
            args = {**cfg.get("common", {}), **vargs}
            override = dict(cfg.get("common", {}).get("override", {}), **vargs.get("override", {}))
            for k, v in zip(keys, combo):
                if k in CFG_FIELDS:
                    override[k] = v
                else:
                    args[k] = v
            args["override"] = override
            name = vname + "".join(f"_{k}{fmt(v)}" for k, v in zip(keys, combo) if k in varying)
            yield name, args


def argv_for(name, args, out):
    argv = ["-m", "training.train_lm", "--run", name, "--out", out]
    for k, v in args.items():
        if k == "override":
            argv += ["--override", json.dumps(v)]
        elif isinstance(v, bool):
            argv += [f"--{k}"] if v else []
        else:
            argv += [f"--{k}", str(v)]
    return argv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    out = cfg.get("out", os.path.join("results", "research", cfg["name"]))
    os.makedirs(out, exist_ok=True)
    runs = [(n, a) for n, a in expand(cfg) if args.only in n]
    manifest = {"config": args.config, "spec": cfg, "runs": {}}
    mpath = os.path.join(out, "manifest.json")
    if os.path.exists(mpath):
        manifest["runs"] = json.load(open(mpath)).get("runs", {})
    print(f"[{cfg['name']}] {len(runs)} runs -> {out}")
    for i, (name, a) in enumerate(runs, 1):
        run_dir = os.path.join(out, name)
        done = os.path.exists(os.path.join(run_dir, "summary.json"))
        print(f"  [{i}/{len(runs)}] {name}{'  (done, skipping)' if done else ''}", flush=True)
        if args.dry_run:
            print("      " + " ".join(argv_for(name, a, out)))
            continue
        status = "done"
        if not done:
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "train.log"), "w") as log:
                rc = subprocess.run([sys.executable, *argv_for(name, a, out)], stdout=log, stderr=subprocess.STDOUT).returncode
            status = "done" if rc == 0 else f"failed (exit {rc})"
        for post in cfg.get("post", []) if status == "done" else []:
            post = [p.replace("{run_dir}", run_dir) for p in post]
            marker = os.path.join(run_dir, ".post_" + "_".join(p.strip("-").replace("/", "_") for p in post[:2]))
            if os.path.exists(marker):
                continue
            with open(os.path.join(run_dir, "post.log"), "a") as log:
                rc = subprocess.run([sys.executable, *post], stdout=log, stderr=subprocess.STDOUT).returncode
            if rc == 0:
                open(marker, "w").close()
            else:
                status = f"post failed (exit {rc})"
        manifest["runs"][name] = {"status": status, "args": a}
        with open(mpath, "w") as f:
            json.dump(manifest, f, indent=2)
        if status != "done":
            print(f"      -> {status}; see {run_dir}", flush=True)


if __name__ == "__main__":
    main()
