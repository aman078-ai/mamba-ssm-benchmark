"""Figures for the length-generalization study (results/research/<experiment>/ -> plots/research/<experiment>_*.png).

    python -m analysis.research_plots stage4_pilot
"""

import glob
import json
import os
import statistics
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from analysis import style
from analysis.style import INK, INK_2, SLOTS, direct_labels, fmt_len

OUT = os.path.join("plots", "research")


def family(cfg):
    m = cfg["model_cfg"]
    if "A" in m["layers"] and "M" not in m["layers"]:
        return f"Transformer ({m.get('pos_encoding', 'rope')})"
    if not m.get("selective", True):
        return f"Mamba LTI (N={m['d_state']})"
    if m.get("a_mode", "s4d") == "scalar":
        return f"Mamba scalar decay (N={m['d_state']})"
    return f"Mamba (N={m['d_state']})"


COLORS = {"Mamba (N=4)": SLOTS[3], "Mamba (N=16)": SLOTS[1], "Mamba (N=64)": SLOTS[4],
          "Mamba LTI (N=16)": SLOTS[5], "Mamba scalar decay (N=16)": "#b5532a", "Mamba scalar decay (N=64)": "#7a2e0e", "Transformer (rope)": SLOTS[0], "Transformer (none)": SLOTS[2],
          "Transformer (alibi)": "#4a3aa7"}


def load(exp):
    runs = []
    for path in sorted(glob.glob(os.path.join("results", "research", exp, "*", "length_generalization.json"))):
        d = os.path.dirname(path)
        runs.append({"name": os.path.basename(d), "cfg": json.load(open(os.path.join(d, "config.json"))),
                     "lg": json.load(open(path))})
    return runs


def agg(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def fig_extrapolation(exp, runs):
    by_len = defaultdict(lambda: defaultdict(list))          # train_len -> family -> [lg]
    for r in runs:
        by_len[r["lg"]["train_len"]][family(r["cfg"])].append(r["lg"])
    tls = sorted(by_len)
    fig, axes = plt.subplots(1, len(tls), figsize=(6.4 * len(tls), 4.4), squeeze=False)
    for ax, tl in zip(axes[0], tls):
        labels = []
        for fam, lgs in sorted(by_len[tl].items()):
            lengths = sorted({int(k) for lg in lgs for k in lg["bpc_by_length"]})
            pts = []
            for L in lengths:
                m, sd = agg([lg["bpc_by_length"].get(str(L)) for lg in lgs])
                if m is not None:
                    pts.append((L / tl, m, sd))
            if not pts:
                continue
            xs, ys, sds = zip(*pts)
            c = COLORS.get(fam, INK_2)
            ax.plot(xs, ys, marker="o", color=c, label=f"{fam} (n={len(lgs)})")
            if max(sds) > 0:
                ax.fill_between(xs, [y - s for y, s in zip(ys, sds)], [y + s for y, s in zip(ys, sds)], color=c, alpha=0.15)
            labels.append((xs[-1], min(ys[-1], 4.0), fam))
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}×"))
        ax.set_ylim(top=min(4.0, ax.get_ylim()[1]))
        ax.axvline(1, color=INK_2, lw=0.8, ls="--")
        ax.set_title(f"Trained on {tl}-byte windows", loc="left")
        ax.set_xlabel("evaluation length / training length")
        ax.set_ylabel("validation bits per byte")
        direct_labels(ax, labels)
    axes[0][0].legend(loc="upper left", fontsize=7.5)
    fig.suptitle(f"Length generalization ({exp}; mean ± sd over seeds where n > 1)", x=0.01, ha="left",
                 fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, f"{exp}_extrapolation.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def band_mean(pl, lo, hi):
    vals = [b["bpc"] for b in pl if b["start"] >= lo and b["end"] <= hi]
    return sum(vals) / len(vals) if vals else None


def scores(lg):
    """Context effect on the same bytes (full context minus ~L context), from length_generalization.py.
    < 0 = extra context helps (utilization), > 0 = extra context hurts (degradation).  Bands are fixed in advance."""
    ce = lg.get("context_effect", {})
    return {"near_bpc": lg["bpc_by_length"].get(str(lg["train_len"])),
            "effect_2_4x": ce.get("2-4x"), "effect_8_16x": ce.get("8-16x"), "effect_16_32x": ce.get("16-32x")}


def fig_position(exp, runs):
    by_len = defaultdict(list)
    for r in runs:
        by_len[r["lg"]["train_len"]].append(r)
    tls = sorted(by_len)
    fig, axes = plt.subplots(1, len(tls), figsize=(6.4 * len(tls), 4.4), squeeze=False)
    for ax, tl in zip(axes[0], tls):
        for r in sorted(by_len[tl], key=lambda r: family(r["cfg"])):
            pl = r["lg"]["position_loss"]
            fam = family(r["cfg"])
            ax.plot([(b["start"] + b["end"]) / 2 / tl for b in pl], [b["bpc"] for b in pl],
                    color=COLORS.get(fam, INK_2), lw=1.4, label=f"{fam} (seed {r['cfg']['args']['seed']})")
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}×"))
        ax.set_ylim(1.4, 4.0)
        ax.axvline(1, color=INK_2, lw=0.8, ls="--")
        ax.set_title(f"Loss at each position, trained on {tl}-byte windows", loc="left")
        ax.set_xlabel("position / training length")
        ax.set_ylabel("validation bits per byte")
    axes[0][0].legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"{exp}_position.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def table(exp, runs):
    rows = []
    for r in runs:
        sc = scores(r["lg"])
        rows.append({"run": r["name"], "family": family(r["cfg"]), "train_len": r["lg"]["train_len"],
                     "seed": r["cfg"]["args"]["seed"], "params": r["cfg"]["non_embedding_params"], **sc})
    rows.sort(key=lambda x: (x["train_len"], x["family"], x["seed"]))
    path = os.path.join("results", "research", exp, "scores.json")
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    fmt = lambda v: "n/a" if v is None else f"{v:+.3f}"
    print(f"{'run':34s} {'L':>5s} {'params':>8s} {'bpc@L':>7s} {'eff 2-4x':>9s} {'eff 8-16x':>9s} {'eff16-32x':>9s}")
    for x in rows:
        print(f"{x['run']:34s} {x['train_len']:5d} {x['params'] / 1e6:7.2f}M {x['near_bpc'] or 0:7.3f} "
              f"{fmt(x['effect_2_4x']):>9s} {fmt(x['effect_8_16x']):>9s} {fmt(x['effect_16_32x']):>9s}")
    print(f"saved {path}")


def fig_state_and_horizon(exp, runs):
    mruns = [r for r in runs if r["lg"].get("mamba")]
    if not mruns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    ax = axes[0]
    for r in mruns:
        sr = r["lg"]["mamba"]["state_rms"]
        top = [max(layer[i] for layer in sr["by_layer"]) for i in range(len(sr["positions"]))]
        fam = family(r["cfg"])
        ax.plot(sr["positions"], [t / top[0] for t in top], marker=".", color=COLORS.get(fam, INK_2),
                ls="-" if r["lg"]["train_len"] == 256 else "--",
                label=f"{fam}, trained at {r['lg']['train_len']}")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_len))
    ax.set_yscale("log")
    ax.set_title("Largest per-layer state RMS vs. position (relative to 1K)", loc="left")
    ax.set_xlabel("position in a single long sequence")
    ax.set_ylabel("state RMS / value at 1K")
    ax.legend(fontsize=7)
    ax = axes[1]
    for r in mruns:
        hz = r["lg"]["mamba"]["horizons"]
        fam = family(r["cfg"])
        ax.plot(range(1, len(hz) + 1), [h["p99"] for h in hz], marker="o", color=COLORS.get(fam, INK_2),
                ls="-" if r["lg"]["train_len"] == 256 else "--", label=f"{fam}, trained at {r['lg']['train_len']}")
    ax.set_yscale("log")
    ax.set_title("99th-percentile memory horizon per layer", loc="left")
    ax.set_xlabel("layer")
    ax.set_ylabel("horizon 1 / (|A| · mean Δ), bytes")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"{exp}_mechanism.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    style.apply()
    exp = sys.argv[1] if len(sys.argv) > 1 else "stage4_pilot"
    runs = load(exp)
    print(f"{exp}: {len(runs)} runs with length_generalization.json")
    if runs:
        fig_extrapolation(exp, runs)
        fig_position(exp, runs)
        fig_state_and_horizon(exp, runs)
        table(exp, runs)
        print(f"saved plots to {OUT}/")


if __name__ == "__main__":
    main()
