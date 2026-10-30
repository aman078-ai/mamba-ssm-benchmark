"""Tables for docs/paper/paper.md, generated from saved results (no number in a table is typed by hand).

    python -m analysis.paper_tables

Writes docs/paper/generated/*.md; docs/paper/build.py splices them in with "!include generated/<name>.md".
"""

import glob
import json
import os
import re
import statistics as st
from collections import defaultdict

from analysis.aggregate_seeds import EXPERIMENTS, stats
from analysis.research_plots import family

OUT = os.path.join("docs", "paper", "generated")
PASSKEY = [("stage5b_passkey", "fixed end"), ("stage5c_passkey_random", "random end"),
           ("stage5d_passkey_stream_mamba", "position-free"), ("stage5d_passkey_stream_alibi", "position-free")]


def ms(s, d=3, signed=False):
    """'mean ± CI' (CI = 95% t-interval over seeds); a single seed prints the value alone."""
    if s is None:
        return "–"
    f = f"{{:{'+' if signed else ''}.{d}f}}"
    m = f.format(s["mean"]).replace("-", "−")
    return m + (f" ± {s['ci95']:.{d}f}" if s.get("ci95") is not None else "")


def runs_of(exps, merge=False):
    """Run directories grouped by configuration (seed suffix removed); merge=True pools seeds across experiments."""
    groups = defaultdict(list)
    for exp in exps:
        for p in sorted(glob.glob(os.path.join("results", "research", exp, "*", "length_generalization.json"))):
            d = os.path.dirname(p)
            name = re.sub(r"_seed\d+$", "", os.path.basename(d))
            groups[(None if merge else exp, name)].append(d)
    return groups


def load(d, name):
    p = os.path.join(d, name)
    return json.load(open(p)) if os.path.exists(p) else None


def label(cfg):
    return family(cfg).replace("Transformer (none)", "Transformer (NoPE)").replace("(rope)", "(RoPE)") \
        .replace("(alibi)", "(ALiBi)").replace("Mamba LTI", "Mamba, non-selective (LTI)") \
        .replace("Mamba scalar decay", "Mamba, scalar decay").replace("Mamba (N=", "Mamba, S4D decay (N=")


ORDER = ["Mamba, S4D decay (N=4)", "Mamba, S4D decay (N=16)", "Mamba, S4D decay (N=64)", "Mamba, scalar decay (N=16)",
         "Mamba, scalar decay (N=64)", "Mamba, non-selective (LTI) (N=16)", "Transformer (RoPE)", "Transformer (NoPE)",
         "Transformer (ALiBi)"]


def main_rows():
    rows = []
    for (_, name), dirs in runs_of(EXPERIMENTS, merge=True).items():
        cfg = load(dirs[0], "config.json")
        lgs = [load(d, "length_generalization.json") for d in dirs]
        L = lgs[0]["train_len"]
        rows.append({"label": label(cfg), "L": L, "params": cfg["non_embedding_params"], "n": len(dirs),
                     "bpc": stats([lg["bpc_by_length"][str(L)] for lg in lgs]),
                     "ce": {b: stats([lg["context_effect"].get(b) for lg in lgs])
                            for b in ("1-2x", "2-4x", "4-8x", "8-16x", "16-32x")},
                     "lgs": lgs})
    rows.sort(key=lambda r: (r["L"], ORDER.index(r["label"])))
    return rows


def table_main(rows):
    lines = ["| Model | $L$ | Params | bpc at $L$ | CE 1–2× | CE 2–4× | CE 4–8× | CE 8–16× | CE 16–32× |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ce = r["ce"]
        lines.append(f"| {r['label']} | {r['L']} | {r['params'] / 1e6:.2f}M | {ms(r['bpc'])} | {ms(ce['1-2x'], signed=True)} | "
                     f"{ms(ce['2-4x'], signed=True)} | {ms(ce['4-8x'], signed=True)} | {ms(ce['8-16x'], signed=True)} | "
                     f"{ms(ce['16-32x'], signed=True)} |")
    return "\n".join(lines)


def table_mechanism(rows):
    lines = ["| Model | $L$ | Median horizon, all modes | Median horizon, slowest state index | 99th pct. | Modes with horizon $> L$ | "
             "Longest mode | State RMS, 16K / 1K |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if not r["lgs"][0].get("mamba"):
            continue
        med, slow, p99, frac, mx, rms = [], [], [], [], [], []
        for lg in r["lgs"]:
            H = lg["mamba"]["horizons"]
            n_idx = len(H[0]["median_by_state_index"])
            slow.append(max(st.median(h["median_by_state_index"][n] for h in H) for n in range(n_idx)))
            med.append(st.median(h["median"] for h in H))
            p99.append(st.median(h["p99"] for h in H))
            frac.append(100 * st.mean(h["frac_above_train_len"] for h in H))
            mx.append(max(h["max"] for h in H))
            rms.append(max(l[-1] / l[0] for l in lg["mamba"]["state_rms"]["by_layer"]))
        lines.append(f"| {r['label']} | {r['L']} | {ms(stats(med), 1)} | {ms(stats(slow), 1)} | {ms(stats(p99), 0)} | "
                     f"{ms(stats(frac), 1)}% | {min(mx):.1e}–{max(mx):.1e} | {ms(stats(rms), 2)} |")
    return "\n".join(lines).replace("e+0", "e").replace("e+", "e")


SHORT = {"Mamba, S4D decay (N=16)": "Mamba S4D, N=16", "Mamba, S4D decay (N=64)": "Mamba S4D, N=64",
         "Mamba, scalar decay (N=16)": "Mamba scalar, N=16", "Mamba, scalar decay (N=64)": "Mamba scalar, N=64",
         "Mamba, non-selective (LTI) (N=16)": "Mamba LTI, N=16", "Transformer (RoPE)": "RoPE", "Transformer (ALiBi)": "ALiBi"}


def table_passkey():
    lines = ["| Protocol (seeds) | Model | Keys exact (%), gap 64 / 256 / 512 / 1024 | bpc at $L$ | CE 8–16× | "
             "P(RECALL) $> 0.5$ from | Mean P(RECALL) beyond $L$ | State RMS 16K/1K |",
             "|---|---|---|---|---|---|---|---|"]
    for exp, proto in PASSKEY:
        for (_, name), dirs in sorted(runs_of([exp]).items(), key=lambda kv: ORDER.index(label(load(kv[1][0], "config.json")))):
            cfg = load(dirs[0], "config.json")
            pks = [load(d, "passkey.json") for d in dirs]
            lgs = [load(d, "length_generalization.json") for d in dirs]
            ctl = [load(d, "passkey_text_control.json") for d in dirs]
            if any(p is None for p in pks):
                continue

            def acc(g):
                v = [p["by_gap"].get(str(g)) for p in pks]
                return stats([100 * x["key_exact"] for x in v if x]) if any(v) else None
            keys = " / ".join(ms(acc(g), 0) for g in (64, 256, 512, 1024))
            L = lgs[0]["train_len"]
            first = sorted({c["recall_marker_prediction"]["first_position_p_gt_half"] for c in ctl if c},
                           key=lambda f: (f is None, f or 0))
            after = stats([c["recall_marker_prediction"]["mean_p_after_L"] for c in ctl if c])
            rms = ms(stats([max(l[-1] / l[0] for l in lg["mamba"]["state_rms"]["by_layer"]) for lg in lgs]), 1) \
                if lgs[0].get("mamba") else "–"
            lines.append(f"| {proto} ({len(dirs)}) | {SHORT.get(label(cfg), label(cfg))} | {keys} | "
                         f"{ms(stats([lg['bpc_by_length'][str(L)] for lg in lgs]), 2)} | "
                         f"{ms(stats([lg['context_effect']['8-16x'] for lg in lgs]), 3, signed=True)} | "
                         f"{', '.join('never' if f is None else str(f) for f in first) or '–'} | {ms(after, 4)} | {rms} |")
    return "\n".join(lines)


def table_lr():
    res = defaultdict(dict)
    for d in glob.glob(os.path.join("results", "research", "stage3_lr_sweep", "*", "")):
        cfg = load(d, "config.json")
        val = [json.loads(l) for l in open(os.path.join(d, "log.jsonl")) if "val_bpc" in l][-1]["val_bpc"]
        res[label(cfg)][cfg["args"]["lr"]] = val
    lrs = sorted({lr for r in res.values() for lr in r})
    lines = ["| Model | " + " | ".join(f"{lr:g}" for lr in lrs) + " |", "|---|" + "---|" * len(lrs)]
    for m, r in sorted(res.items()):
        best = min(r.values())
        lines.append(f"| {m} | " + " | ".join(("**%.3f**" if r[lr] == best else "%.3f") % r[lr] for lr in lrs) + " |")
    return "\n".join(lines)


def table_test():
    p = os.path.join("results", "research", "test_split.json")
    if not os.path.exists(p):
        return "| (test-split check not run) |\n|---|"
    runs = json.load(open(p))["runs"]
    groups = defaultdict(list)
    for r in runs.values():
        groups[(r["train_len"], r["name"])].append(r)
    valid = {(r["L"], r["label"]): r for r in main_rows()}
    lines = ["| Model | $L$ | Seeds | bpc at $L$, test | CE 2–4×, test | CE 8–16×, test | CE 8–16×, validation |",
             "|---|---|---|---|---|---|---|"]
    keyed = [(L, rs) for (L, _), rs in groups.items()]
    for L, rs in sorted(keyed, key=lambda kv: (kv[0], ORDER.index(label_from_family(kv[1][0]["family"])))):
        lab = label_from_family(rs[0]["family"])
        v = valid.get((L, lab))
        lines.append(f"| {lab} | {L} | {len(rs)} | {ms(stats([r['bpc_at_train_len'] for r in rs]))} | "
                     f"{ms(stats([r['context_effect'].get('2-4x') for r in rs]), signed=True)} | "
                     f"{ms(stats([r['context_effect'].get('8-16x') for r in rs]), signed=True)} | "
                     f"{ms(v['ce']['8-16x'], signed=True) if v else '–'} |")
    return "\n".join(lines)


def label_from_family(f):
    return f.replace("Transformer (none)", "Transformer (NoPE)").replace("(rope)", "(RoPE)").replace("(alibi)", "(ALiBi)") \
        .replace("Mamba LTI", "Mamba, non-selective (LTI)").replace("Mamba scalar decay", "Mamba, scalar decay") \
        .replace("Mamba (N=", "Mamba, S4D decay (N=")


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = main_rows()
    tables = {"table_main": table_main(rows), "table_mechanism": table_mechanism(rows), "table_passkey": table_passkey(),
              "table_lr": table_lr(), "table_test": table_test()}
    for k, v in tables.items():
        with open(os.path.join(OUT, k + ".md"), "w", encoding="utf-8") as f:
            f.write(v + "\n")
        print(f"--- {k}\n{v}\n")


if __name__ == "__main__":
    main()
