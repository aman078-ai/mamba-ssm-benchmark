"""Shared matplotlib style: one fixed color per architecture, recessive axes, direct labels."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"
REF = "#a3a29c"   # reference slope lines

# Categorical slots in fixed order (validated palette: blue, orange, aqua, yellow, magenta, green).
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
ARCH = {
    "transformer": dict(color=SLOTS[0], marker="o", label="Transformer (naive attention)"),
    "mamba": dict(color=SLOTS[1], marker="s", label="Mamba (fused Triton scan)"),
    "retnet": dict(color=SLOTS[2], marker="^", label="RetNet (linear attn.)"),
    "hybrid": dict(color=SLOTS[3], marker="D", label="Hybrid (Mamba + 1 attn.)"),
    # variants keep their parent's color and switch to a dashed line
    "transformer_sdpa": dict(color=SLOTS[0], marker="o", label="Transformer (fused attention, PyTorch SDPA)", ls="--", mfc=SURFACE),
    "mamba_torchscan": dict(color=SLOTS[1], marker="s", label="Mamba (naive PyTorch scan)", ls="--", mfc=SURFACE),
    "mamba_lti": dict(color=SLOTS[1], marker="s", label="Mamba w/o selection (LTI)", ls="--", mfc=SURFACE),
}
SCAN = {
    "reference": dict(color=SLOTS[4], marker="v", label="autograd loop (original)"),
    "sequential": dict(color=SLOTS[5], marker="x", label="sequential + recompute"),
    "hillis_steele": dict(color=SLOTS[2], marker="^", label="Hillis-Steele (log depth)"),
    "blelloch": dict(color=SLOTS[3], marker="D", label="Blelloch (work-efficient)"),
    "chunked": dict(color=SLOTS[0], marker="o", label="chunked two-level"),
    "triton": dict(color=SLOTS[1], marker="s", label="fused Triton kernel"),
}


def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9.5,
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 8.5, "legend.labelcolor": INK,
        "lines.linewidth": 1.8, "lines.markersize": 5.5, "font.size": 9.5,
    })


def line(ax, x, y, spec, label=None, **kw):
    style = dict(color=spec["color"], marker=spec["marker"], ls=spec.get("ls", "-"),
                 mfc=spec.get("mfc", spec["color"]), mec=spec["color"], mew=1.4,
                 label=label or spec["label"])
    style.update(kw)
    return ax.plot(x, y, **style)


def direct_labels(ax, items, dx=6, min_gap_px=11):
    """items: list of (x, y, text).  Places text right of each end point, nudged apart vertically."""
    if not items:
        return
    fig = ax.figure
    fig.canvas.draw()
    lo, hi = sorted(ax.get_ylim())
    items = [(x, min(max(y, lo), hi), text) for x, y, text in items]     # keep labels inside the axes
    trans = ax.transData
    pts = [(trans.transform((x, y)), text) for x, y, text in items]
    pts.sort(key=lambda p: p[0][1])
    ys = [p[0][1] for p in pts]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_gap_px)
    inv = ax.transData.inverted()
    for (xy, text), y in zip(pts, ys):
        xd, yd = inv.transform((xy[0] + dx, y))
        ax.annotate(text, (xd, yd), fontsize=8, color=INK, va="center", ha="left", annotation_clip=False)


def ref_slope(ax, x0, y0, x1, power, text):
    """Gray guide line y ∝ x^power through (x0, y0), labelled at its far end."""
    xs = np.array([x0, x1], dtype=float)
    ys = y0 * (xs / x0) ** power
    ax.plot(xs, ys, color=REF, lw=1.0, ls=":", zorder=0)
    if text:
        ax.annotate(text, (xs[1], ys[1]), xytext=(2, 2), textcoords="offset points", fontsize=7.5, color=INK_2)


def oom_marker(ax, x, y, color, text=True):
    ax.plot([x], [y], marker="x", color=color, ms=9, mew=2.2, ls="none")
    if text:
        ax.annotate("OOM", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7.5, color=INK_2)


def fmt_len(v, _=None):
    v = int(v)
    return f"{v // 1024}K" if v >= 1024 and v % 1024 == 0 else str(v)
