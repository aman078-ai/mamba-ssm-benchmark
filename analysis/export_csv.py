"""Flatten every JSON/JSONL result into CSV files under results/csv/ (one row per measurement).

    python -m analysis.export_csv
"""

import csv
import glob
import json
import os

R = "results"
OUT = os.path.join(R, "csv")


def j(path):
    return json.load(open(path)) if os.path.exists(path) else None


def jl(path):
    return [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []


def write(name, rows):
    if not rows:
        return
    os.makedirs(OUT, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(OUT, f"{name}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT}/{name}.csv ({len(rows)} rows)")


def flat(prefix, d):
    return {f"{prefix}_{k}": v for k, v in (d or {}).items()}


def main():
    d = j(f"{R}/bench/scan.json")
    if d:
        write("bench_scan", [{"scan": r["scan"], "L": r["L"], **flat("fwd", r["fwd"]), **flat("fwd_bwd", r["fwd_bwd"])}
                             for r in d["runs"]])
    d = j(f"{R}/bench/models.json")
    if d:
        write("bench_models", [{"variant": r["variant"], "L": r["L"], "params": r["params"], **flat("fwd", r["fwd"]),
                                **flat("train", r["train"])} for r in d["runs"]])
        write("bench_train_throughput", [{"variant": v, **r} for v, r in d.get("train_b16_l512", {}).items()])
    d = j(f"{R}/bench/decode.json")
    if d:
        write("bench_decode", d["runs"])

    ev = j(f"{R}/enwik8/eval.json") or {}
    runs, curves = [], []
    for path in sorted(glob.glob(f"{R}/enwik8*/*/summary.json")):
        s = j(path)
        run_dir = os.path.dirname(path)
        cfg = j(os.path.join(run_dir, "config.json")) or {}
        e = ev.get(s["run"], {}) if run_dir.startswith(os.path.join(R, "enwik8") + os.sep) else {}
        runs.append({"experiment": os.path.basename(os.path.dirname(run_dir)), **s,
                     "lr": cfg.get("args", {}).get("lr"), "seed": cfg.get("args", {}).get("seed"),
                     "test_bpc": e.get("test_bpc"),
                     **{f"test_bpc_ctx{k}": v for k, v in e.get("bpc_by_length", {}).items()}})
        for r in jl(os.path.join(run_dir, "log.jsonl")):
            if "val_bpc" in r:
                curves.append({"experiment": runs[-1]["experiment"], "run": s["run"], **r})
    write("enwik8_runs", runs)
    write("enwik8_val_curves", curves)
    write("synthetic_mqar", jl(f"{R}/synthetic/mqar.jsonl"))
    write("synthetic_selective", [{k: v for k, v in r.items() if k != "acc_by_length"} |
                                  {f"acc_len{k}": v for k, v in r.get("acc_by_length", {}).items()}
                                  for r in jl(f"{R}/synthetic/selective.jsonl")])

    research = []
    for path in sorted(glob.glob(f"{R}/research/*/*/summary.json")):
        s = j(path)
        run_dir = os.path.dirname(path)
        cfg = j(os.path.join(run_dir, "config.json")) or {}
        lg = j(os.path.join(run_dir, "length_generalization.json")) or {}
        research.append({"experiment": os.path.basename(os.path.dirname(run_dir)), **s,
                         **{k: v for k, v in cfg.get("args", {}).items() if k in ("lr", "seed", "seq_len")},
                         **{f"cfg_{k}": v for k, v in cfg.get("model_cfg", {}).items()
                            if k in ("d_model", "d_state", "layers", "pos_encoding", "selective")},
                         **{f"bpc_ctx{k}": v for k, v in lg.get("bpc_by_length", {}).items()}})
    write("research_runs", research)


if __name__ == "__main__":
    main()
