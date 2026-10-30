"""Figures for docs/paper/paper.md that are not produced elsewhere (all from saved results).

    python -m analysis.paper_figures

plots/research/paper_mechanism.png  memory horizon by state index (3 seeds) and state RMS vs. position
plots/research/paper_passkey.png    P(RECALL marker) on plain text by position, context effect and recall by gap for
                                    the three passkey protocols
(plots/research/final_context_effect.png comes from analysis.aggregate_seeds.)
"""

import glob
import json
import os
import statistics as st

import matplotlib.pyplot as plt

from analysis import style
from analysis.research_plots import COLORS

R = os.path.join("results", "research")
BANDS = ["1-2x", "2-4x", "4-8x", "8-16x", "16-32x"]


def lg_runs(pattern):
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(R, pattern, "length_generalization.json")))]


def mechanism():
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for N, key in ((4, "Mamba (N=4)"), (16, "Mamba (N=16)"), (64, "Mamba (N=64)")):
        for L, ls in ((256, "-"), (1024, "--")):
            runs = lg_runs(f"stage4_pilot/mamba_n{N}_L{L}") + lg_runs(f"stage5a_seeds/mamba_n{N}_L{L}_seed*")
            curves = [[st.median(h["median_by_state_index"][n] for h in r["mamba"]["horizons"]) for n in range(N)]
                      for r in runs]
            mean = [st.mean(c[n] for c in curves) for n in range(N)]
            a.plot(range(1, N + 1), mean, ls, color=COLORS[key], marker="o", ms=2.5, lw=1.4,
                   label=f"N = {N}, trained at L = {L} ({len(runs)} seeds)")
    for L in (256, 1024):
        a.axhline(L, color="#a3a29c", ls=":", lw=1)
        a.text(1.05, L * 1.08, f"L = {L}", color="#7a7974", fontsize=8)
    a.set_xscale("log", base=2)
    a.set_yscale("log")
    a.set_xlabel("state index n (S4D decay, A_n initialized to −(n+1))")
    a.set_ylabel("memory horizon 1 / (|A| · mean Δ), bytes")
    a.set_title("(a) Median horizon per state index (median over channels and layers)", loc="left", fontsize=10)
    a.legend(fontsize=7)

    def rms_curve(pattern):
        runs = lg_runs(pattern)
        pos = runs[0]["mamba"]["state_rms"]["positions"]
        rel = [st.mean(l[i] / l[0] for l in r["mamba"]["state_rms"]["by_layer"]) for r in runs for i in range(len(pos))]
        k = len(pos)
        return pos, [st.mean(rel[j * k + i] for j in range(len(runs))) for i in range(k)], len(runs)
    for pattern, lab, c, ls in (("stage4_pilot/mamba_n16_L256", "plain text, L = 256", COLORS["Mamba (N=16)"], "-"),
                                ("stage5a_seeds/mamba_n16_L256_seed*", None, COLORS["Mamba (N=16)"], "-"),
                                ("stage5b_passkey/mamba_s4d_n16", "passkey, fixed end (5b)", "#7a2e0e", "-"),
                                ("stage5c_passkey_random/mamba_s4d_n16", "passkey, random end (5c)", "#b5532a", "--"),
                                ("stage5d_passkey_stream_mamba/mamba_s4d_n16_seed*", "passkey, position-free (5d)", "#4a3aa7", "-.")):
        if not lg_runs(pattern) or lab is None:
            continue
        if pattern.startswith("stage4_pilot"):
            pos, y0, n0 = rms_curve(pattern)
            _, y1, n1 = rms_curve("stage5a_seeds/mamba_n16_L256_seed*")
            y = [(a0 * n0 + a1 * n1) / (n0 + n1) for a0, a1 in zip(y0, y1)]
            n = n0 + n1
        else:
            pos, y, n = rms_curve(pattern)
        b.plot(pos, y, ls, color=c, marker="o", ms=3, label=f"Mamba N = 16, {lab} ({n} seed{'s' if n > 1 else ''})")
    b.set_xscale("log", base=2)
    b.set_xticks([1024, 2048, 4096, 8192, 16384], ["1K", "2K", "4K", "8K", "16K"])
    b.set_ylabel("SSM state RMS / value at 1K (mean over layers)")
    b.set_xlabel("position in one long validation sequence (bytes)")
    b.set_title("(b) State norm growth", loc="left", fontsize=10)
    b.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join("plots", "research", "paper_mechanism.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def passkey():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    a, b, c = axes
    sets = [("stage5b_passkey/mamba_s4d_n16", "Mamba N=16, fixed end", "#7a2e0e", "-"),
            ("stage5b_passkey/transformer_alibi", "ALiBi, fixed end", "#4a3aa7", "-"),
            ("stage5c_passkey_random/mamba_s4d_n16", "Mamba N=16, random end", "#b5532a", "--"),
            ("stage5d_passkey_stream_mamba/mamba_s4d_n16_seed*", "Mamba N=16, position-free", COLORS["Mamba (N=16)"], ":"),
            ("stage5d_passkey_stream_alibi/transformer_alibi_seed*", "ALiBi, position-free", "#8f84d8", ":")]
    for pattern, lab, col, ls in sets:
        dirs = sorted(glob.glob(os.path.join(R, pattern)))
        ctl = [json.load(open(os.path.join(d, "passkey_text_control.json"))) for d in dirs
               if os.path.exists(os.path.join(d, "passkey_text_control.json"))]
        ctl = [x for x in ctl if "p_by_position" in x["recall_marker_prediction"]]
        if ctl:
            curves = [x["recall_marker_prediction"]["p_by_position"] for x in ctl]
            k = min(len(cv) for cv in curves)
            y = [max(st.mean(cv[i] for cv in curves), 1e-6) for i in range(k)]
            a.plot(range(k), y, ls, color=col, lw=1.2, label=f"{lab} ({len(ctl)})")
        lgs = [json.load(open(os.path.join(d, "length_generalization.json"))) for d in dirs
               if os.path.exists(os.path.join(d, "length_generalization.json"))]
        if lgs:
            b.plot(range(len(BANDS)), [st.mean(lg["context_effect"][k] for lg in lgs) for k in BANDS], ls,
                   color=col, marker="o", ms=3.5, label=f"{lab} ({len(lgs)})")
        pks = [json.load(open(os.path.join(d, "passkey.json"))) for d in dirs if os.path.exists(os.path.join(d, "passkey.json"))]
        if pks:
            gaps = sorted({int(g) for p in pks for g, v in p["by_gap"].items() if v})
            y = [100 * st.mean(p["by_gap"][str(g)]["key_exact"] for p in pks if p["by_gap"].get(str(g))) for g in gaps]
            c.plot(gaps, y, ls, color=col, marker="o", ms=3.5, label=f"{lab} ({len(pks)})")
    a.axvline(512, color="#a3a29c", lw=1)
    a.set_yscale("log")
    a.set_ylim(1e-6, 2)
    a.set_xlabel("position in plain held-out text (bytes; line = training length 512)")
    a.set_ylabel("P(next byte = RECALL marker)")
    a.set_title("(a) Predicting the marker from position alone", loc="left", fontsize=10)
    a.legend(fontsize=7, loc="lower right")
    b.set_xticks(range(len(BANDS)), [x.replace("x", "×").replace("-", "–") for x in BANDS])
    b.set_yscale("symlog", linthresh=0.01)
    b.axhline(0, color="#52514e", lw=0.8)
    b.set_xlabel("position band (multiples of L = 512)")
    b.set_ylabel("context effect on plain text, bits per byte (symlog)")
    b.set_title("(b) Same-bytes context effect after passkey training", loc="left", fontsize=10)
    b.legend(fontsize=7)
    c.axvline(512, color="#a3a29c", lw=1)
    c.set_xscale("log", base=2)
    c.set_xlabel("gap between STORE and RECALL (bytes)")
    c.set_ylabel("keys recalled exactly (%)")
    c.set_title("(c) Passkey recall by gap", loc="left", fontsize=10)
    c.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join("plots", "research", "paper_passkey.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    style.apply()
    mechanism()
    passkey()
    print("saved plots/research/paper_mechanism.png, plots/research/paper_passkey.png")


if __name__ == "__main__":
    main()
