"""Regenerate every number and figure in the README from a clean clone.

    python reproduce.py                    # everything (~4-5 h on an RTX 3050 Laptop GPU)
    python reproduce.py --only bench,plots # selected stages
    python reproduce.py --quick            # minutes-long smoke run; writes only to results/_quick/

Stages, in order:
    tests      pytest (scans vs NumPy brute force, gradients, cache/state decoding, Triton kernels)
    train      enwik8 training of the four matched-size models
    ablation   Mamba without selection / with Euler / bilinear discretization (first 1000 steps)
    eval       test bpc, length extrapolation, per-position loss
    bench      scan algorithms, full-model speed/memory, generation latency
    synthetic  MQAR recall sweep, selective copying
    delta      what the selection mechanism learned
    plots      figures in plots/, tables in results/summary.md, text samples, docs/index.html
"""

import argparse
import subprocess
import sys

STAGES = ["tests", "train", "ablation", "eval", "bench", "synthetic", "delta", "plots"]
QUICK_STAGES = ["train", "ablation", "eval", "bench", "synthetic"]   # these can be redirected to results/_quick
MODELS = ["transformer", "mamba", "retnet", "hybrid"]


def run(*args):
    cmd = [sys.executable, *args]
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(STAGES))
    ap.add_argument("--quick", action="store_true", help="tiny settings; outputs go to results/_quick/")
    args = ap.parse_args()
    stages = args.only.split(",")
    q = args.quick
    R = "results/_quick" if q else "results"
    if q:
        stages = [s for s in stages if s in QUICK_STAGES]

    if "tests" in stages:
        run("-m", "pytest", "-q")
    if "train" in stages:
        for m in MODELS:
            run("-m", "training.train_lm", "--model", m, "--steps", "20" if q else "4000", "--out", f"{R}/enwik8",
                *(["--eval_every", "10", "--eval_bytes", "20000"] if q else []))
    if "ablation" in stages:
        for name, ov in (("mamba_lti", '{"selective": false}'), ("mamba_euler", '{"discretization": "euler"}'),
                         ("mamba_bilinear", '{"discretization": "bilinear"}')):
            run("-m", "training.train_lm", "--model", "mamba", "--run", name, "--override", ov,
                "--steps", "4000", "--stop_step", "10" if q else "1000", "--out", f"{R}/enwik8_ablation",
                *(["--eval_every", "10", "--eval_bytes", "20000"] if q else []))
    if "eval" in stages:
        run("-m", "analysis.evaluate_lm", "--root", f"{R}/enwik8",
            *(["--lengths", "512,1024", "--extrap_bytes", "20000", "--position_len", "1024",
               "--position_bytes", "20000"] if q else []))
    if "bench" in stages:
        run("-m", "benchmarks.bench_scan", "--out", f"{R}/bench/scan.json", *(["--lengths", "256,1024"] if q else []))
        run("-m", "benchmarks.bench_models", "--out", f"{R}/bench/models.json", *(["--lengths", "256,1024"] if q else []))
        run("-m", "benchmarks.bench_decode", "--out", f"{R}/bench/decode.json", *(["--contexts", "128,1024"] if q else []))
    if "synthetic" in stages:
        for suite in ("mqar", "selective"):
            run("-m", "tasks.run_synthetic", "--suite", suite, "--out", f"{R}/synthetic", *(["--steps", "20"] if q else []))
    if "delta" in stages:
        run("-m", "analysis.delta_analysis")
    if "plots" in stages:
        run("-m", "analysis.make_plots")
        with open("results/summary.md", "w", encoding="utf-8") as f:
            subprocess.run([sys.executable, "-m", "analysis.summarize"], stdout=f, check=True)
        run("generate.py", "--n", "300", "--out", "results/samples.md")
        run("-m", "analysis.build_dashboard")
    if q:
        print(f"\nquick smoke run finished; outputs in {R}/ (delete it when done)")


if __name__ == "__main__":
    main()
