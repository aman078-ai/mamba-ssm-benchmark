"""Combine seeds across experiments (stage4_pilot / stage4b_scalar_decay = seed 0, stage5a_seeds = seeds 1-2).

    python -m analysis.aggregate_seeds

Writes results/research/aggregate.json, results/research/aggregate.md and plots/research/final_context_effect.png.
Mean, sample standard deviation and a 95% t-interval (df = n - 1) per configuration.
"""

import glob
import json
import os
import re
import statistics
from collections import defaultdict

import matplotlib.pyplot as plt

from analysis import style
from analysis.research_plots import COLORS, family

EXPERIMENTS = ["stage4_pilot", "stage4b_scalar_decay", "stage5a_seeds"]
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
BANDS = ["1-2x", "2-4x", "4-8x", "8-16x", "16-32x"]


def stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    m = statistics.mean(vals)
    if len(vals) < 2:
        return {"mean": m, "sd": None, "ci95": None, "n": len(vals), "values": vals}
    sd = statistics.stdev(vals)
    return {"mean": m, "sd": sd, "ci95": T95.get(len(vals) - 1, 2.0) * sd / len(vals) ** 0.5, "n": len(vals), "values": vals}


def collect():
    groups = defaultdict(list)
    for exp in EXPERIMENTS:
        for path in glob.glob(os.path.join("results", "research", exp, "*", "length_generalization.json")):
            d = os.path.dirname(path)
            name = re.sub(r"_seed\d+$", "", os.path.basename(d))
            cfg = json.load(open(os.path.join(d, "config.json")))
            groups[name].append({"cfg": cfg, "lg": json.load(open(path)), "seed": cfg["args"]["seed"]})
    out = {}
    for name, runs in sorted(groups.items()):
        L = runs[0]["lg"]["train_len"]
        rec = {"family": family(runs[0]["cfg"]), "train_len": L, "params": runs[0]["cfg"]["non_embedding_params"],
               "seeds": sorted(r["seed"] for r in runs),
               "bpc_at_train_len": stats([r["lg"]["bpc_by_length"].get(str(L)) for r in runs]),
               "context_effect": {b: stats([r["lg"].get("context_effect", {}).get(b) for r in runs]) for b in BANDS}}
        if runs[0]["lg"].get("mamba"):
            rec["horizon_p99_median_layer"] = stats([statistics.median(h["p99"] for h in r["lg"]["mamba"]["horizons"])
                                                     for r in runs])
            rec["state_rms_growth_1k_to_end"] = stats([max(l[-1] / l[0] for l in r["lg"]["mamba"]["state_rms"]["by_layer"])
                                                       for r in runs])
        out[name] = rec
    return out


def fmt(s, digits=3, signed=False):
    if not s:
        return "—"
    m = f"{s['mean']:+.{digits}f}" if signed else f"{s['mean']:.{digits}f}"
    return m + (f" ± {s['ci95']:.{digits}f}" if s.get("ci95") is not None else "") + f" (n={s['n']})"


def table(agg):
    lines = ["| configuration | training length | params | bpc at training length | context effect 2–4× | context effect 8–16× |",
             "|---|---|---|---|---|---|"]
    for name, r in sorted(agg.items(), key=lambda kv: (kv[1]["train_len"], kv[1]["family"])):
        ce = r["context_effect"]
        lines.append(f"| {r['family']} | {r['train_len']} | {r['params'] / 1e6:.2f}M | {fmt(r['bpc_at_train_len'])} | "
                     f"{fmt(ce['2-4x'], signed=True)} | {fmt(ce['8-16x'], signed=True)} |")
    return "\n".join(lines)


def figure(agg):
    style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=False)
    for ax, L in zip(axes, (256, 1024)):
        rows = [(n, r) for n, r in agg.items() if r["train_len"] == L and r["context_effect"]["8-16x"]]
        rows.sort(key=lambda nr: nr[1]["context_effect"]["8-16x"]["mean"])
        for i, (n, r) in enumerate(rows):
            s = r["context_effect"]["8-16x"]
            ax.barh(i, s["mean"], color=COLORS.get(r["family"], "#888"), height=0.6,
                    xerr=s["ci95"] if s["ci95"] else None, capsize=3)
        ax.set_yticks(range(len(rows)), [r["family"] for _, r in rows])
        ax.set_xscale("symlog", linthresh=0.01)
        ax.axvline(0, color="#52514e", lw=0.8)
        ax.set_title(f"Trained on {L}-byte windows: effect of extra context at 8–16×", loc="left")
        ax.set_xlabel("bits per byte, full context − ~training-length context (< 0 helps, > 0 hurts; 95% CI)")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    os.makedirs(os.path.join("plots", "research"), exist_ok=True)
    fig.savefig(os.path.join("plots", "research", "final_context_effect.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    agg = collect()
    with open(os.path.join("results", "research", "aggregate.json"), "w") as f:
        json.dump(agg, f, indent=2)
    md = table(agg)
    with open(os.path.join("results", "research", "aggregate.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(md)
    figure(agg)
    print("saved results/research/aggregate.{json,md}, plots/research/final_context_effect.png")


if __name__ == "__main__":
    main()
