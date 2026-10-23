"""Figure for the passkey experiments (stage 5b fixed RECALL position, stage 5c random position).

    python -m analysis.passkey_plots

Left: same-bytes context effect beyond the training length on plain text.  Right: passkey recall by gap.
Writes plots/research/passkey.png.
"""

import glob
import json
import os

import matplotlib.pyplot as plt

from analysis import style
from analysis.research_plots import COLORS, family

BANDS = ["1-2x", "2-4x", "4-8x", "8-16x", "16-32x"]


def runs(exp):
    out = []
    for d in sorted(glob.glob(os.path.join("results", "research", exp, "*", "length_generalization.json"))):
        d = os.path.dirname(d)
        cfg = json.load(open(os.path.join(d, "config.json")))
        out.append({"family": family(cfg), "lg": json.load(open(os.path.join(d, "length_generalization.json"))),
                    "pk": json.load(open(os.path.join(d, "passkey.json")))})
    return out


def main():
    style.apply()
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8))
    for exp, ls, tag in (("stage5b_passkey", "-", "fixed"), ("stage5c_passkey_random", "--", "random")):
        for r in runs(exp):
            c = COLORS.get(r["family"], "#888")
            ce = r["lg"]["context_effect"]
            a.plot(range(len(BANDS)), [ce[k] for k in BANDS], ls, color=c, marker="o", ms=4,
                   label=f"{r['family']}, RECALL {tag}")
            gaps = [(int(g), v["key_exact"]) for g, v in r["pk"]["by_gap"].items() if v]
            if tag == "fixed":
                b.plot([g for g, _ in gaps], [100 * v for _, v in gaps], "-", color=c, marker="o", ms=4,
                       label=r["family"])
    a.set_xticks(range(len(BANDS)), [k.replace("x", "×").replace("-", "–") for k in BANDS])
    a.set_yscale("log")                                               # every value is > 0 here
    a.set_xlabel("position band (multiples of the 512-byte training length)")
    a.set_ylabel("context effect, bits per byte (> 0 hurts)")
    a.set_title("Plain text after passkey training: collapse", loc="left")
    a.legend(fontsize=7, loc="lower right", ncol=2)
    a.set_ylim(0.1, 100)
    b.set_xscale("log", base=2)
    b.axvline(512, color="#a3a29c", ls=":", lw=1)
    b.set_xlabel("gap between STORE and RECALL (bytes; dotted = training length)")
    b.set_ylabel("keys recalled exactly (%)")
    b.set_title("Passkey recall (fixed-position training)", loc="left")
    b.legend(fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.join("plots", "research"), exist_ok=True)
    fig.savefig(os.path.join("plots", "research", "passkey.png"), dpi=150, bbox_inches="tight")
    print("saved plots/research/passkey.png")


if __name__ == "__main__":
    main()
