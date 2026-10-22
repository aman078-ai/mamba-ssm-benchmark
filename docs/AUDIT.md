# Research audit and upgrade plan

Audit of commit `2b31138` (October 2026). Every claim in the README was treated as a hypothesis and checked against the
code, the raw results and new diagnostic runs. Checks that produced evidence live in
[`analysis/audit_checks.py`](../analysis/audit_checks.py) and write to [`results/audit/`](../results/audit).

Status legend: **open**, **fixed (stage N)**, **mitigated** (documented, not removed), **scheduled (stage N)**.

## Findings

| # | Priority | Category | Finding | Evidence | Status |
|---|---|---|---|---|---|
| 1 | critical | unfair comparison | Headline efficiency claims compared naive attention with a fused Mamba kernel. | `results/bench/models.json`: fused attention runs a 128K forward pass and trains at 105.8k tok/s vs. Mamba's 42.8k. | fixed (stage 0): README now compares naive-naive and fused-fused |
| 2 | critical | experimental design | Every quality number is a single seed (seed 0). | `results/enwik8/*/config.json` | mitigated (stage 0: labelled preliminary); scheduled (stage 3: seed variance) |
| 3 | critical | experimental design | Synthetic-task conclusion was a training-budget artifact. | `results/audit/mqar_budget.json`: pure Mamba, 4-pair MQAR, 7.8% at 3,000 steps vs. 52.9% at 9,000 steps (re-run: 52.6% and 52.9%; the spread comes from the non-deterministic kernel, #13) | fixed (stage 0): claim withdrawn |
| 4 | critical | experimental design | One learning rate for all architectures; no tuning. | `training/train_lm.py` defaults | scheduled (stage 3: per-architecture LR sweep) |
| 5 | high | reporting | "4 GB" memory claims; PyTorch was capped at 2.97 GiB. | `results/logs/bench.log` | fixed (stage 0); environment capture scheduled (stage 1) |
| 6 | high | benchmarking | Benchmarks ran with other applications on the GPU (game launcher + overlay, voice app, chat app). | `nvidia-smi` process list | mitigated (stage 0); clean re-run scheduled (stage 2/3) |
| 7 | high | confound | No-selection ablation has 7.5% fewer parameters, one seed, compared at step 1,000 of 4,000. | `results/enwik8_ablation/*/config.json` | mitigated (stage 0); param-matched rerun scheduled (stage 5) |
| 8 | high | missing baseline | Length extrapolation compared RoPE without any extension method; no NoPE/ALiBi baseline; 8K points missing (OOM). | `models/transformer.py`, `results/enwik8/eval.json` | scheduled (stage 4) |
| 9 | high | experimental design | Models not converged; ranking is budget-dependent. | val bpc still falling 0.006–0.01 per 250 steps at the end | mitigated (stage 0) |
| 10 | high | baseline strength | Transformer uses a GELU MLP, weaker than the "Transformer++" (SwiGLU) baseline the Mamba paper uses. Impact unmeasured. | `models/common.py` | open |
| 11 | high | reproducibility | Dependencies unpinned (`torch>=2.4`, `numpy`). | `requirements.txt` | scheduled (stage 1) |
| 12 | high | testing | Triton kernels never run in CI (tests skip without a GPU). | `.github/workflows/tests.yml` | mitigated: GPU tests documented as a manual step (stage 2) |
| 13 | medium | reproducibility | Triton backward uses atomics, so dB and dC differ run-to-run (up to 9.2e-5). | `results/audit/determinism.json` | scheduled (stage 2: deterministic mode) |
| 14 | medium | numerics | bf16 full forward vs. bf16 token-by-token decode disagree on 0.33% of argmax predictions (fp32: max logit diff 5.5e-3). Expected for bf16, undocumented. | `results/audit/consistency.json` | scheduled (stage 2: GPU equivalence test with tolerances) |
| 15 | medium | benchmarking | "Training throughput" excluded the optimizer step. | `benchmarks/bench_models.py` | scheduled (stage 2) |
| 16 | medium | missing metrics | No prefill / time-to-first-token split, reserved or process memory, FLOPs, batch>1 decoding, official-kernel comparison. | — | partly scheduled (stages 1–2) |
| 17 | medium | reproducibility | No experiment configuration files; no raw CSV. | — | scheduled (stage 1) |
| 18 | medium | repository | No LICENSE or CITATION file. | — | scheduled (stage 1) |
| 19 | medium | reporting | "bpc" undefined (it is bits per byte). | README | fixed (stage 0) |
| 20 | low | repository | GitHub repository description still says "64–1024 tokens". | GitHub page | open (owner action) |

Re-verified as holding: the test suite design (NumPy brute force, finite-difference gradients, ZOH vs. RK4, cache and
recurrent-state equivalence, causality); a 64K-token forward pass of the trained Mamba stays finite; parameter matching
within 2.1% across the four main models; identical batch order across models; the test split was never used for
model selection.

## Literature position (abstract-level reading; full papers still to be read)

| Result in this repository | Already established by |
|---|---|
| Linear vs. quadratic scaling, constant generation state | Gu & Dao 2023; consumer-GPU measurements in arXiv 2507.12442 |
| Attention beats SSMs at recall and copying | Arora et al. 2023 (Zoology); Jelassi et al. 2024 |
| Hybrids beat both | Waleffe et al. 2024 (8B, 3.5T tokens); Lieber et al. 2024 (Jamba) |
| Byte-level Mamba is competitive | Wang et al. 2024 (MambaByte) |
| Parallel scans for linear recurrences | Martin & Cundy 2018; Dao & Gu 2024 |
| **When Mamba generalizes beyond its training length** | **contested**: DeciMamba (limited, receptive field tied to training length), Stuffed Mamba (state collapse when the state is oversized), Mamba Modulation (open area) |

## Research question

In small selective SSMs trained on byte-level text, how do **state size** and **training context length** decide
whether the model generalizes to longer contexts, and does the learned **step size Δ** (its effective receptive field)
predict where it fails?

- **H1** (Stuffed Mamba): at a fixed training length, a larger state starts failing sooner beyond it.
- **H2**: a longer training length delays that failure.
- **H3** (DeciMamba): the receptive field implied by Δ, measured on held-out text, predicts the failure length.
- Controls: Mamba without selection; Transformers with RoPE, NoPE and ALiBi at the same training lengths.

## Stage checklist

| Stage | Goal | Status |
|---|---|---|
| 0 | Correct the record: README claims, units, memory cap, withdrawn findings | done |
| 1 | Reproducibility: pinned environment, experiment configs, CSV export, environment capture, LICENSE, CITATION | pending |
| 2 | Tests and tooling: deterministic kernel mode, GPU equivalence, long-sequence stability, optimizer-inclusive throughput | pending |
| 3 | Seed variance and per-architecture learning-rate tuning on the research models | pending |
| 4 | Pilot of H1–H3 (one seed) with baselines; go/no-go | pending |
| 5 | Full grid with 3 seeds and parameter-matched controls | pending |
| 6 | Analysis, statistics, figures from CSV | pending |
| 7 | Manuscript, release tag | pending |

Before/after records for each stage are in [`CHANGELOG.md`](../CHANGELOG.md).
