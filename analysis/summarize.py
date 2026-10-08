"""Print every headline number as Markdown tables (pasted into the README).

    python -m analysis.summarize > results/summary.md
"""

import json
import os
import sys

R = "results"
ARCHS = ["transformer", "mamba", "retnet", "hybrid"]
NAMES = {"transformer": "Transformer", "mamba": "Mamba", "retnet": "RetNet", "hybrid": "Hybrid",
         "transformer_sdpa": "Transformer (PyTorch fused attn.)", "mamba_torchscan": "Mamba (pure-PyTorch scan)",
         "mamba_lti": "Mamba w/o selection (LTI)"}


def j(path):
    return json.load(open(path)) if os.path.exists(path) else None


def jl(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def fmt_ms(r):
    return "OOM" if r is None or r.get("oom") else (f"{r['ms']:.1f}" if r["ms"] >= 1 else f"{r['ms']:.2f}")


def fmt_len(L):
    return f"{L // 1024}K" if L >= 1024 else str(L)


def models_table():
    rows = []
    bench = j(f"{R}/bench/models.json") or {}
    tp = bench.get("train_b16_l512", {})
    print("| model | layers | non-embedding params | training throughput (tokens/s) | training memory |")
    print("|---|---|---|---|---|")
    for a in ARCHS:
        s = j(f"{R}/enwik8/{a}/summary.json")
        if not s:
            continue
        t = tp.get(a, {})
        thr = f"{t['tokens_per_s'] / 1e3:.1f}k" if "tokens_per_s" in t else "n/a"
        mem = f"{t['peak_mb'] / 1024:.2f} GB" if "peak_mb" in t else "n/a"
        print(f"| {NAMES[a]} | `{s['layers']}` | {s['non_embedding_params'] / 1e6:.2f}M | {thr} | {mem} |")
    print()


def enwik8_table():
    ev = j(f"{R}/enwik8/eval.json") or {}
    lengths = ["512", "1024", "2048", "4096", "8192"]
    print("| model | val bpc | **test bpc** (ctx 512) | " + " | ".join(f"test @ {fmt_len(int(L))}" for L in lengths[1:]) + " |")
    print("|---|---|---|" + "---|" * (len(lengths) - 1))
    for a in ARCHS:
        s = j(f"{R}/enwik8/{a}/summary.json")
        e = ev.get(a)
        if not (s and e):
            continue
        by = e["bpc_by_length"]
        cells = [f"{by[L]:.3f}" if by.get(L) is not None else "OOM" for L in lengths[1:]]
        print(f"| {NAMES[a]} | {s['final_val_bpc']:.3f} | **{e['test_bpc']:.3f}** | " + " | ".join(cells) + " |")
    print()


def speed_table():
    d = j(f"{R}/bench/models.json")
    if not d:
        return
    runs = d["runs"]
    lengths = sorted({r["L"] for r in runs})
    show = [L for L in lengths if L in (1024, 4096, 16384, 65536, 131072)]
    variants = list(dict.fromkeys(r["variant"] for r in runs))
    for key, title in (("fwd", "Inference forward pass, batch 1 (ms)"), ("train", "Training step fwd+bwd, batch 1 (ms)")):
        print(f"**{title}**\n")
        print("| model | " + " | ".join(fmt_len(L) for L in show) + " | longest length that fits |")
        print("|---|" + "---|" * (len(show) + 1))
        for v in variants:
            rr = {r["L"]: r[key] for r in runs if r["variant"] == v}
            fits = [L for L, m in rr.items() if not m.get("oom")]
            print(f"| {NAMES[v]} | " + " | ".join(fmt_ms(rr.get(L)) for L in show) +
                  f" | {fmt_len(max(fits)) if fits else '-'} |")
        print()
    # crossover: first length where mamba forward is faster than transformer
    for key in ("fwd", "train"):
        t = {r["L"]: r[key] for r in runs if r["variant"] == "transformer"}
        m = {r["L"]: r[key] for r in runs if r["variant"] == "mamba"}
        cross = None
        for L in lengths:
            if L in t and L in m and not m[L].get("oom"):
                if t[L].get("oom") or m[L]["ms"] < t[L]["ms"]:
                    cross = L
                    break
        print(f"- crossover ({key}): Mamba faster than Transformer from L = {fmt_len(cross) if cross else 'never'}")
    print()


def decode_table():
    d = j(f"{R}/bench/decode.json")
    if not d:
        return
    runs = d["runs"]
    ctxs = sorted({r["context"] for r in runs})
    print("GPU time per generated token (CUDA graph replay) · cache/state size")
    print()
    print("| model | " + " | ".join(f"{fmt_len(c)} ctx" for c in ctxs) + " |")
    print("|---|" + "---|" * len(ctxs))
    for a in ARCHS:
        rr = {r["context"]: r for r in runs if r["model"] == a}
        cells = []
        for c in ctxs:
            r = rr.get(c)
            cells.append("OOM" if not r or r.get("oom") else f"{r['ms_graph']:.2f} ms · {r['state_mb']:.4g} MB")
        print(f"| {NAMES[a]} | " + " | ".join(cells) + " |")
    print()


def scan_table():
    d = j(f"{R}/bench/scan.json")
    if not d:
        return
    runs = d["runs"]
    lengths = sorted({r["L"] for r in runs})
    print("| scan | " + " | ".join(f"L={fmt_len(L)} fwd / fwd+bwd (ms)" for L in lengths) + " | peak mem fwd+bwd @ max L |")
    print("|---|" + "---|" * (len(lengths) + 1))
    for s in dict.fromkeys(r["scan"] for r in runs):
        rr = {r["L"]: r for r in runs if r["scan"] == s}
        cells = [f"{fmt_ms(rr[L]['fwd'])} / {fmt_ms(rr[L]['fwd_bwd'])}" if L in rr else "-" for L in lengths]
        top = rr[max(rr)]["fwd_bwd"]
        mem = "OOM" if top.get("oom") else f"{top['peak_mb']:.0f} MB @ {fmt_len(max(rr))}"
        print(f"| `{s}` | " + " | ".join(cells) + f" | {mem} |")
    print()


def synthetic_tables():
    rows = jl(f"{R}/synthetic/mqar.jsonl")
    if rows:
        kvs = sorted({r["n_kv"] for r in rows})
        print("| model (d_model 64) | " + " | ".join(f"{k} pairs" for k in kvs) + " |")
        print("|---|" + "---|" * len(kvs))
        for a in ARCHS:
            acc = {r["n_kv"]: r["acc"] for r in rows if r["arch"] == a}
            if acc:
                print(f"| {NAMES[a]} | " + " | ".join(f"{acc[k]:.1%}" if k in acc else "-" for k in kvs) + " |")
        print()
    rows = jl(f"{R}/synthetic/selective.jsonl")
    if rows:
        lengths = sorted({int(L) for r in rows for L in r.get("acc_by_length", {})})
        print("| model | params | accuracy (trained length) | " + " | ".join(f"@ {fmt_len(L)}" for L in lengths if L != 128) + " |")
        print("|---|---|---|" + "---|" * (len(lengths) - 1))
        for r in rows:
            by = r.get("acc_by_length", {})
            cells = [f"{by[str(L)]:.1%}" if by.get(str(L)) is not None else "OOM" for L in lengths if L != 128]
            print(f"| {NAMES[r['arch']]} | {r['params'] / 1e3:.0f}K | **{r['acc']:.1%}** | " + " | ".join(cells) + " |")
        print()


ABLATIONS = [("mamba", "results/enwik8/mamba", "selective Δ,B,C · exact ZOH (default)"),
             ("mamba_lti", "results/enwik8_ablation/mamba_lti", "**no selection** (LTI: Δ,B,C constant)"),
             ("mamba_euler", "results/enwik8_ablation/mamba_euler", "Euler B̄ = ΔB (Mamba's simplification)"),
             ("mamba_bilinear", "results/enwik8_ablation/mamba_bilinear", "bilinear / Tustin")]


def ablation_curve(run_dir, max_step=1000):
    return [(r["step"], r["val_bpc"]) for r in jl(f"{run_dir}/log.jsonl") if "val_bpc" in r and r["step"] <= max_step]


def ablation_table():
    print("| Mamba variant | val bpc @ step 500 | val bpc @ step 1000 | Δ vs default |")
    print("|---|---|---|---|")
    base = None
    for name, d, desc in ABLATIONS:
        c = dict(ablation_curve(d))
        if 1000 not in c:
            continue
        base = c[1000] if base is None else base
        diff = "—" if name == "mamba" else f"{c[1000] - base:+.3f}"
        print(f"| {desc} | {c.get(500, float('nan')):.3f} | {c[1000]:.3f} | {diff} |")
    print()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    for title, fn in (("Models", models_table), ("enwik8", enwik8_table), ("Speed", speed_table),
                      ("Ablation", ablation_table), ("Decode", decode_table), ("Scans", scan_table),
                      ("Synthetic", synthetic_tables)):
        print(f"### {title}\n")
        fn()


if __name__ == "__main__":
    main()
