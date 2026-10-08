"""Generate every figure in plots/ from the JSON results (skips figures whose data is missing).

    python -m analysis.make_plots
"""

import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from analysis import style
from analysis.style import ARCH, INK, INK_2, SCAN, direct_labels, fmt_len, line, oom_marker, ref_slope

R = "results"
P = "plots"
ARCHS = ["transformer", "mamba", "retnet", "hybrid"]


def load_json(path):
    return json.load(open(path)) if os.path.exists(path) else None


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def log_len_axis(ax, which="x"):
    axis = ax.xaxis if which == "x" else ax.yaxis
    (ax.set_xscale if which == "x" else ax.set_yscale)("log", base=2)
    axis.set_major_formatter(mticker.FuncFormatter(fmt_len))


def save(fig, name):
    os.makedirs(P, exist_ok=True)
    fig.savefig(os.path.join(P, name), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {P}/{name}")


# ------------------------------------------------------------------------------------------
def series(runs, key, metric, field="ms", filt=None):
    """-> {name: (xs, ys, oom_x)} from bench records."""
    out = {}
    for r in runs:
        name = r.get("variant") or r.get("scan") or r.get("model")
        if filt and name not in filt:
            continue
        m = r[metric] if metric else r
        xs, ys, oom = out.setdefault(name, ([], [], [None]))
        if m.get("oom"):
            if oom[0] is None:
                oom[0] = r[key]
        elif field in m:
            xs.append(r[key])
            ys.append(m[field])
    return {k: (v[0], v[1], v[2][0]) for k, v in out.items()}


def plot_lines(ax, data, specs, order, label_ends=True):
    labels = []
    for name in order:
        if name not in data or not data[name][0]:
            continue
        xs, ys, oom = data[name]
        spec = specs[name]
        line(ax, xs, ys, spec)
        if oom is not None:
            oom_marker(ax, oom, ys[-1], spec["color"], text=not label_ends)
        if label_ends:
            name_txt = spec["label"].split(" (")[0] + (" *" if spec.get("ls") == "--" else "")
            labels.append((oom if oom is not None else xs[-1], ys[-1], name_txt + (" · OOM" if oom is not None else "")))
    return labels


def fig_scan():
    d = load_json(f"{R}/bench/scan.json")
    if not d:
        return
    order = ["reference", "sequential", "hillis_steele", "blelloch", "chunked", "triton"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (metric, field, title, ylab) in zip(axes, [
            ("fwd", "ms", "Forward time", "ms (median)"),
            ("fwd_bwd", "ms", "Forward + backward time", "ms (median)"),
            ("fwd_bwd", "peak_mb", "Peak memory, forward + backward", "MB above inputs")]):
        data = series(d["runs"], "L", metric, field)
        plot_lines(ax, data, SCAN, order, label_ends=False)
        log_len_axis(ax)
        ax.set_yscale("log")
        ax.set_title(title, loc="left")
        ax.set_xlabel("sequence length L")
        ax.set_ylabel(ylab)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.07))
    c = d["config"]
    fig.suptitle(f"Selective-scan algorithms on one Mamba layer (B={c['B']}, D={c['D']}, N={c['N']}), "
                 f"{d['env']['gpu']}", x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout()
    save(fig, "scan_algorithms.png")


def fig_models():
    d = load_json(f"{R}/bench/models.json")
    if not d:
        return
    order = ["transformer", "transformer_sdpa", "mamba", "mamba_torchscan", "retnet", "hybrid"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    panels = [("fwd", "ms", "Inference forward pass", "ms (median)"),
              ("fwd", "peak_mb", "Inference peak memory", "MB"),
              ("train", "ms", "Training step (fwd + bwd)", "ms (median)")]
    for ax, (metric, field, title, ylab) in zip(axes, panels):
        data = series(d["runs"], "L", metric, field)
        labels = plot_lines(ax, data, ARCH, order)
        log_len_axis(ax)
        ax.set_yscale("log")
        ax.set_title(title, loc="left")
        ax.set_xlabel("sequence length (bytes), batch 1")
        ax.set_ylabel(ylab)
        x0, x1 = min(r["L"] for r in d["runs"]), max(r["L"] for r in d["runs"])
        ylim = ax.get_ylim()
        y0 = ylim[0] * 1.3
        ref_slope(ax, x0, y0, min(x1, x0 * (ylim[1] / y0) ** 0.5), 2, "∝ L²")    # stop at the top edge
        ref_slope(ax, x0, y0, min(x1, x0 * ylim[1] / y0), 1, "∝ L")
        ax.set_ylim(ylim)
        direct_labels(ax, [l for l in labels if "*" not in l[2]])
    axes[1].axhline(d["env"]["vram_gib"] * 1024, color=INK_2, lw=0.9, ls="--")
    axes[1].annotate(f"GPU memory ({d['env']['vram_gib']:.0f} GB)", (axes[1].get_xlim()[0], d["env"]["vram_gib"] * 1024),
                     xytext=(4, 3), textcoords="offset points", fontsize=7.5, color=INK_2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Speed and memory vs. sequence length (~4.7M-param models, bf16)  ·  × = out of memory",
                 x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout()
    save(fig, "speed_memory_vs_length.png")


def fig_decode():
    d = load_json(f"{R}/bench/decode.json")
    if not d:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, (field, title, ylab) in zip(axes, [("ms_graph", "GPU time per generated token (CUDA graph)", "ms / token"),
                                               ("state_mb", "Cache / state memory", "MB")]):
        data = series(d["runs"], "context", None, field)
        labels = plot_lines(ax, data, ARCH, ARCHS)
        log_len_axis(ax)
        ax.set_yscale("log")
        ax.set_title(title, loc="left")
        ax.set_xlabel("tokens already in context")
        ax.set_ylabel(ylab)
        direct_labels(ax, labels)
    eager = [r["ms"] for r in d["runs"] if "ms" in r]
    if eager:
        axes[0].annotate(f"Eager PyTorch adds ~{min(eager):.0f}-{max(eager):.0f} ms of kernel-launch overhead per token "
                         "on this machine; the graph replay removes it.", (0, -0.2), xycoords="axes fraction",
                         fontsize=7.5, color=INK_2)
    fig.suptitle("Autoregressive generation: KV cache grows with context, SSM state does not",
                 x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout()
    save(fig, "generation_latency_memory.png")


def fig_enwik8():
    logs = {a: load_jsonl(f"{R}/enwik8/{a}/log.jsonl") for a in ARCHS}
    ev = load_json(f"{R}/enwik8/eval.json") or {}
    if not any(logs.values()):
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    labels = []
    for a in ARCHS:
        pts = [(r["tokens"] / 1e6, r["val_bpc"]) for r in logs[a] if "val_bpc" in r and r["step"] > 0]
        if pts:
            xs, ys = zip(*pts)
            line(ax, xs, ys, ARCH[a], ms=3.5)
            labels.append((xs[-1], ys[-1], f"{ARCH[a]['label'].split(' (')[0]}  {ys[-1]:.3f}"))
    ax.set_ylim(top=min(3.0, ax.get_ylim()[1]))
    ax.set_title("Validation bits-per-character", loc="left")
    ax.set_xlabel("training tokens (millions)")
    ax.set_ylabel("bpc (lower is better)")
    direct_labels(ax, labels)

    ax = axes[1]
    labels = []
    for a in ARCHS:
        if a in ev:
            items = sorted((int(k), v) for k, v in ev[a]["bpc_by_length"].items() if v is not None)
            if items:
                xs, ys = zip(*items)
                line(ax, xs, ys, ARCH[a])
                labels.append((xs[-1], ys[-1], ARCH[a]["label"].split(" (")[0]))
    log_len_axis(ax)
    ax.axvline(512, color=INK_2, lw=0.9, ls="--")
    ax.annotate("training length", (512, ax.get_ylim()[1]), xytext=(4, -12), textcoords="offset points",
                fontsize=7.5, color=INK_2)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_minor_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.set_title("Length extrapolation (test set)", loc="left")
    ax.set_xlabel("evaluation context length")
    ax.set_ylabel("bpc")
    direct_labels(ax, labels)

    ax = axes[2]
    labels = []
    for a in ARCHS:
        if a in ev and ev[a].get("position_loss"):
            pl = ev[a]["position_loss"]
            xs = [(b["start"] + b["end"]) / 2 for b in pl]
            ys = [b["bpc"] for b in pl]
            line(ax, xs, ys, ARCH[a], marker=None)
            labels.append((xs[-1], ys[-1], ARCH[a]["label"].split(" (")[0]))
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_len))
    ax.axvline(512, color=INK_2, lw=0.9, ls="--")
    ax.set_ylim(1.0, 4.0)
    ax.set_title("Loss at each position of a 4K-byte context", loc="left")
    ax.set_xlabel("position in context")
    ax.set_ylabel("bpc")
    direct_labels(ax, labels)
    fig.suptitle("enwik8 byte-level language modeling (~4.7M non-embedding params each, 32.8M training tokens)",
                 x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout()
    save(fig, "enwik8_quality.png")


def fig_ablation():
    from analysis.summarize import ABLATIONS, ablation_curve
    curves = [(n, ablation_curve(d), desc) for n, d, desc in ABLATIONS]
    if sum(1 for _, c, _ in curves if c) < 2:
        return
    specs = {"mamba": dict(color=style.SLOTS[1], marker="s", label="selective, ZOH (default)"),
             "mamba_lti": dict(color=style.SLOTS[1], marker="s", label="no selection (LTI)", ls="--", mfc=style.SURFACE),
             "mamba_euler": dict(color=style.SLOTS[0], marker="o", label="selective, Euler"),
             "mamba_bilinear": dict(color=style.SLOTS[2], marker="^", label="selective, bilinear")}
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    labels = []
    for n, c, _ in curves:
        c = [(s_, v) for s_, v in c if s_ > 0]
        if c:
            xs, ys = zip(*c)
            line(ax, xs, ys, specs[n])
            labels.append((xs[-1], ys[-1], f"{specs[n]['label']}  {ys[-1]:.3f}"))
    ax.set_title("Mamba ablations on enwik8 (same LR schedule, first 1000 steps)", loc="left")
    ax.set_xlabel("training step (16 × 512 bytes each)")
    ax.set_ylabel("validation bpc")
    direct_labels(ax, labels)
    fig.tight_layout()
    save(fig, "mamba_ablations.png")


def fig_mqar():
    rows = load_jsonl(f"{R}/synthetic/mqar.jsonl")
    if not rows:
        return
    dms = sorted({r["d_model"] for r in rows})
    has_ext = any("acc_by_length" in r for r in rows)
    ncols = len(dms) + (1 if has_ext else 0)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4.2), squeeze=False)
    axes = axes[0]
    for ax, dm in zip(axes, dms):
        labels = []
        for a in ARCHS:
            pts = sorted((r["n_kv"], r["acc"]) for r in rows if r["arch"] == a and r["d_model"] == dm)
            if pts:
                xs, ys = zip(*pts)
                line(ax, xs, ys, ARCH[a])
                labels.append((xs[-1], ys[-1], ARCH[a]["label"].split(" (")[0]))
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_ylim(-0.03, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.set_title(f"Recall accuracy, d_model = {dm}", loc="left")
        ax.set_xlabel("key-value pairs to remember (sequence length 128)")
        ax.set_ylabel("accuracy")
        direct_labels(ax, labels)
    if has_ext:
        ax = axes[-1]
        labels = []
        for a in ARCHS:
            r = [r for r in rows if r["arch"] == a and "acc_by_length" in r]
            if r:
                items = sorted((int(k), v) for k, v in r[0]["acc_by_length"].items())
                xs = [k for k, v in items if v is not None]
                ys = [v for k, v in items if v is not None]
                if xs:
                    line(ax, xs, ys, ARCH[a])
                    labels.append((xs[-1], ys[-1], ARCH[a]["label"].split(" (")[0]))
                    oom = [k for k, v in items if v is None]
                    if oom:
                        oom_marker(ax, oom[0], ys[-1], ARCH[a]["color"])
        log_len_axis(ax)
        ax.axvline(128, color=INK_2, lw=0.9, ls="--")
        ax.set_ylim(-0.03, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ex = [r for r in rows if "acc_by_length" in r][0]
        ax.set_title(f"Same task, longer gaps ({ex['n_kv']} pairs, d={ex['d_model']})", loc="left")
        ax.set_xlabel("evaluation sequence length (trained at 128)")
        ax.set_ylabel("accuracy")
        direct_labels(ax, labels)
    fig.suptitle("Multi-query associative recall after 3,000 steps: models with attention learn exact recall, pure Mamba does not yet",
                 x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout()
    save(fig, "mqar_recall.png")


def fig_selective():
    rows = load_jsonl(f"{R}/synthetic/selective.jsonl")
    if not rows:
        return
    order = ["mamba", "mamba_lti", "transformer", "retnet", "hybrid"]
    rows = {r["arch"]: r for r in rows}
    names = [a for a in order if a in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 1.2]})
    ax = axes[0]
    accs = [rows[a]["acc"] for a in names]
    bars = ax.barh(range(len(names)), accs, color=[ARCH[a]["color"] for a in names], height=0.62,
                   hatch=["//" if ARCH[a].get("ls") == "--" else "" for a in names], edgecolor=style.SURFACE, lw=2)
    ax.set_yticks(range(len(names)), [ARCH[a]["label"] for a in names])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    for b, v in zip(bars, accs):
        ax.annotate(f"{v:.1%}", (b.get_width(), b.get_y() + b.get_height() / 2), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=INK)
    ax.axvline(1 / 16, color=INK_2, lw=0.9, ls=":")
    ax.annotate("chance", (1 / 16, len(names) - 0.4), xytext=(3, 0), textcoords="offset points", fontsize=7.5, color=INK_2)
    ax.set_title("Accuracy at the trained length (8 tokens, 128 noise)", loc="left")
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    labels = []
    for a in names:
        if "acc_by_length" in rows[a]:
            items = sorted((int(k), v) for k, v in rows[a]["acc_by_length"].items() if v is not None)
            xs, ys = zip(*items)
            line(ax, xs, ys, ARCH[a])
            labels.append((xs[-1], ys[-1], ARCH[a]["label"].split(" (")[0] + (" (LTI)" if a == "mamba_lti" else "")))
    log_len_axis(ax)
    ax.set_ylim(-0.03, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.axvline(128, color=INK_2, lw=0.9, ls="--")
    ax.set_title("Longer noise spans than seen in training", loc="left")
    ax.set_xlabel("noise span length (trained at 128)")
    ax.set_ylabel("accuracy")
    direct_labels(ax, labels)
    fig.suptitle("Selective copying after 3,000 steps: one attention layer is enough to solve it; pure Mamba has not learned it yet",
                 x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout()
    save(fig, "selective_copying.png")


def fig_delta():
    d = load_json(f"{R}/delta.json")
    if not d:
        return
    text = d["text"]
    M = np.array(d["delta_rel"])                       # (layers, T): Δ / layer mean
    fig, ax = plt.subplots(figsize=(15, 3.6))
    im = ax.imshow(np.log2(M), aspect="auto", cmap="Blues", vmin=-1, vmax=2, interpolation="nearest")
    ax.set_yticks(range(M.shape[0]), [f"layer {i + 1}" for i in range(M.shape[0])], fontsize=7)
    ax.set_xticks(range(len(text)), [c if c != " " else "␣" for c in text], fontsize=6.5, family="monospace")
    ax.tick_params(axis="x", length=0)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("log₂(Δ / layer mean)", color=INK_2)
    ax.set_title("What the trained Mamba chooses to write into its state: step size Δ per byte "
                 "(darker = larger Δ = more of this token enters the state)", loc="left")
    fig.tight_layout()
    save(fig, "delta_heatmap.png")

    if d.get("selective"):
        s = d["selective"]
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        kinds = ["noise", "data", "marker"]
        x = np.arange(len(kinds))
        w = 0.36
        for i, (arch, lab) in enumerate([("mamba", "Mamba (selective)"), ("mamba_lti", "Mamba w/o selection")]):
            if arch in s:
                spec = ARCH[arch]
                vals = [s[arch][k] for k in kinds]
                bars = ax.bar(x + (i - 0.5) * w, vals, w, color=spec["color"],
                              hatch="//" if spec.get("ls") == "--" else "", edgecolor=style.SURFACE, lw=2, label=lab)
                for b, v in zip(bars, vals):
                    ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                                textcoords="offset points", ha="center", fontsize=8, color=INK)
        ax.set_xticks(x, ["noise tokens", "data tokens", "copy markers"])
        ax.set_ylabel("mean Δ in the first Mamba layer")
        ax.legend(loc="upper right")
        ax.set_title("Selective copying: how strongly each token type is written into the state", loc="left")
        ax.grid(axis="x", visible=False)
        fig.tight_layout()
        save(fig, "delta_selective.png")


def fig_hero():
    m = load_json(f"{R}/bench/models.json")
    dec = load_json(f"{R}/bench/decode.json")
    if not (m and dec):
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (src, key, metric, field, title, ylab, xl) in zip(axes, [
            (m, "L", "fwd", "ms", "Forward pass time", "ms", "sequence length"),
            (m, "L", "fwd", "peak_mb", "Forward pass memory", "MB", "sequence length"),
            (dec, "context", None, "ms_graph", "GPU time per generated token", "ms / token", "tokens in context")]):
        data = series(src["runs"], key, metric, field, filt=ARCHS)
        labels = plot_lines(ax, data, ARCH, ARCHS)
        log_len_axis(ax)
        ax.set_yscale("log")
        ax.set_title(title, loc="left")
        ax.set_xlabel(xl)
        ax.set_ylabel(ylab)
        direct_labels(ax, labels)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("Transformer vs. Mamba vs. RetNet vs. hybrid, all written from scratch (RTX 3050 Laptop, 4 GB)",
                 x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout()
    save(fig, "hero.png")


def main():
    style.apply()
    for f in (fig_hero, fig_scan, fig_models, fig_decode, fig_enwik8, fig_ablation, fig_mqar, fig_selective,
              fig_delta):
        try:
            f()
        except Exception as e:      # one broken figure should not block the others
            print(f"[warn] {f.__name__} failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
