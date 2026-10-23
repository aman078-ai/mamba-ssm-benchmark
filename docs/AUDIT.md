# Research audit and upgrade plan

Audit of commit `2b31138` (October 2026). Every claim in the README was treated as a hypothesis and checked against the
code, the raw results and new diagnostic runs. Checks that produced evidence live in
[`analysis/audit_checks.py`](../analysis/audit_checks.py) and write to [`results/audit/`](../results/audit).

Status legend: **open**, **fixed (stage N)**, **mitigated** (documented, not removed), **scheduled (stage N)**.

## Findings

| # | Priority | Category | Finding | Evidence | Status |
|---|---|---|---|---|---|
| 1 | critical | unfair comparison | Headline efficiency claims compared naive attention with a fused Mamba kernel. | `results/bench/models.json`: fused attention runs a 128K forward pass and trains at 105.8k tok/s vs. Mamba's 42.8k. | fixed (stage 0): README now compares naive-naive and fused-fused |
| 2 | critical | experimental design | Every quality number is a single seed (seed 0). | `results/enwik8/*/config.json` | mitigated (stage 0); seed variance measured (stage 3: sd ≈ 0.005 bpc); multi-seed final runs in stage 5 |
| 3 | critical | experimental design | Synthetic-task conclusion was a training-budget artifact. | `results/audit/mqar_budget.json`: pure Mamba, 4-pair MQAR, 7.8% at 3,000 steps vs. 52.9% at 9,000 steps (re-run: 52.6% and 52.9%; the spread comes from the non-deterministic kernel, #13) | fixed (stage 0): claim withdrawn |
| 4 | critical | experimental design | One learning rate for all architectures; no tuning. | `training/train_lm.py` defaults | fixed for the research models (stage 3); original 4.7M runs stay labelled untuned |
| 5 | high | reporting | "4 GB" memory claims; PyTorch was capped at 2.97 GiB. | `results/logs/bench.log` | fixed (stage 0); cap now recorded with every result (stage 1) |
| 6 | high | benchmarking | Benchmarks ran with other applications on the GPU (game launcher + overlay, voice app, chat app). | `nvidia-smi` process list | fixed (2026-10-23): re-run with the voice app closed; changes < 5%, no conclusion changed (`results/bench/run1_shared_gpu/` kept) |
| 7 | high | confound | No-selection ablation has 7.5% fewer parameters, one seed, compared at step 1,000 of 4,000. | `results/enwik8_ablation/*/config.json` | mitigated (stage 0); param-matched rerun scheduled (stage 5) |
| 8 | high | missing baseline | Length extrapolation compared RoPE without any extension method; no NoPE/ALiBi baseline; 8K points missing (OOM). | `models/transformer.py`, `results/enwik8/eval.json` | scheduled (stage 4) |
| 9 | high | experimental design | Models not converged; ranking is budget-dependent. | val bpc still falling 0.006–0.01 per 250 steps at the end | mitigated (stage 0) |
| 10 | high | baseline strength | Transformer uses a GELU MLP, weaker than the "Transformer++" (SwiGLU) baseline the Mamba paper uses. Impact unmeasured. | `models/common.py` | open |
| 11 | high | reproducibility | Dependencies unpinned (`torch>=2.4`, `numpy`). | `requirements.txt` | fixed (stage 1): `requirements-lock.txt` |
| 12 | high | testing | Triton kernels never run in CI (tests skip without a GPU). | `.github/workflows/tests.yml` | mitigated (stage 2): `tests/test_gpu.py` + `tests/test_triton.py` must be run locally on a GPU before each release |
| 13 | medium | reproducibility | Triton backward uses atomics, so dB and dC differ run-to-run (up to 9.2e-5). | `results/audit/determinism.json` | fixed (stage 2): bitwise-deterministic mode under `torch.use_deterministic_algorithms(True)`, tested |
| 14 | medium | numerics | bf16 full forward vs. bf16 token-by-token decode disagree on 0.33% of argmax predictions (fp32: max logit diff 5.5e-3). Expected for bf16, undocumented. | `results/audit/consistency.json` | fixed (stage 2): GPU equivalence test (fp32, 1e-4) for Mamba, hybrid and RetNet stacks; bf16 gap documented |
| 15 | medium | benchmarking | "Training throughput" excluded the optimizer step. | `benchmarks/bench_models.py` | fixed (stage 2): `train_b16_l512_opt` includes clipping + AdamW; README now quotes it |
| 16 | medium | missing metrics | No prefill / time-to-first-token split, reserved or process memory, FLOPs, batch>1 decoding, official-kernel comparison. | — | partly scheduled (stages 1–2) |
| 17 | medium | reproducibility | No experiment configuration files; no raw CSV. | — | fixed (stage 1): `configs/` + `experiments/run.py`, `results/csv/` |
| 18 | medium | repository | No LICENSE or CITATION file. | — | fixed (stage 1): MIT `LICENSE`, `CITATION.cff` |
| 19 | medium | reporting | "bpc" undefined (it is bits per byte). | README | fixed (stage 0) |
| 20 | low | repository | GitHub repository description still says "64–1024 tokens". | GitHub page | open (owner action) |
| 21 | high | analysis error | Window-level bpc improving with window length was read as "Mamba keeps using more context". It reflects fewer cold-start bytes per window; position bands within a window are also confounded by which text sits where. Same-bytes measurement (full vs. ~512 context) at 8–16× the training length: Transformer +2.96, Mamba −0.001, RetNet −0.001, hybrid +0.047 bits/byte. | `results/enwik8/*/length_generalization.json` | fixed (stage 3): README finding 2 rewritten on the same-bytes measurement |

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
- Measurement (from audit finding 21): window-level bpc mixes context use with the number of cold-start bytes per
  window, and comparing positions inside one window mixes context use with how hard the text at each position is.
  Each run is therefore scored on the **same bytes under two contexts**: the loss with the full context from the start
  of a 16K window minus the loss with about one training length of context (sliding windows). Reported per band of
  positions, fixed in advance: 1–2×, 2–4×, 4–8×, 8–16× (and 16–32×) the training length. Negative = extra context helps
  (utilization); positive = it hurts (degradation or collapse).

## Stage 3 results (validation split; research models, 1,500 steps of 16 × 512 bytes)

Learning-rate sweep, seed 0 (`results/research/stage3_lr_sweep/`):

| learning rate | 1e-3 | 2e-3 | 4e-3 | 8e-3 | 1.6e-2 | 3.2e-2 |
|---|---|---|---|---|---|---|
| Mamba (8 layers, d 128, N 16; 0.93M) | 2.034 | 1.974 | 1.949 | 1.940 | **1.936** | 1.958 |
| Transformer, RoPE (5 layers, d 128; 0.98M) | 2.325 | 2.192 | 2.112 | **2.092** | 2.104 | 2.965 |

Seed variance at the chosen rates (`results/research/stage3_seeds/`): Mamba 1.932 / 1.940 / 1.931 (mean 1.934, sd 0.005);
Transformer 2.093 / 2.099 / 2.099 (mean 2.097, sd 0.004). The 0.163 bpc gap is about 36 pooled standard deviations.

What this changes:
- Single-seed differences below about 0.015 bpc (3 sd) are treated as noise in the pilot.
- Re-running Mamba with the same seed moved it by 0.004 bpc (atomics in the kernel; Transformer re-runs are
  bit-identical), so kernel nondeterminism is about as large as seed variance at this scale.
- The best learning rates are 5–10× higher than the 1.5e-3 used for the original 4.7M comparison, which therefore ran
  both architectures far from their optimum (supports finding 4).

## Stage 4 pilot results (one seed; validation split; `results/research/stage4_pilot/`, `stage4b_scalar_decay/`)

Context effect = loss on the same bytes with full context minus loss with ~one training length of context
(bits per byte; < 0 = extra context helps, > 0 = it hurts). Models ~0.8–1.2M parameters, 24.6M training bytes.

| model | trained at 256: bpc at 256 | effect 2–4× | effect 8–16× | effect 16–32× | trained at 1024: effect 8–16× |
|---|---|---|---|---|---|
| Mamba, N = 4 | 1.845 | −0.001 | +0.001 | +0.000 | +0.000 |
| Mamba, N = 16 | 1.838 | −0.001 | −0.000 | −0.001 | −0.000 |
| Mamba, N = 64 | 1.839 | −0.000 | −0.002 | −0.001 | −0.000 |
| Mamba, scalar decay, N = 16 | 1.841 | +0.001 | −0.001 | −0.001 | — |
| Mamba, scalar decay, N = 64 | 1.847 | −0.001 | −0.001 | −0.001 | — |
| Mamba without selection (LTI), N = 16 | 1.874 | +0.002 | +0.002 | +0.003 | +0.000 |
| Transformer, ALiBi | 1.900 | **−0.058** | **−0.051** | **−0.051** | −0.007 |
| Transformer, NoPE | 2.330 | +2.84 | +3.36 | +3.54 | +0.97 |
| Transformer, RoPE | 1.928 | +3.20 | +2.80 | +2.79 | +3.23 |

Mechanism measurements (Mamba runs):
- Memory horizons are short: median over channels per state index is ~50–100 bytes for the slowest state index and
  1–25 bytes for the fastest (S4D decay); ~8–9 bytes for every index with scalar decay. Only a thin tail of modes
  reaches a few hundred bytes (99th percentile 240–1,350 bytes).
- Horizons do not scale with training length (99th percentile, N = 16: 784 bytes trained at 256 vs. 679 at 1,024) and
  shrink as N grows (S4D decay adds faster modes).
- State RMS grows at most 1.5× from 1K to 16K bytes: no sign of state explosion.

**Go/no-go.** H1 (a larger state collapses sooner) and H3 (horizons tied to training length) are not supported in this
regime, and the follow-up rules out decay parameterization as the explanation: neither per-state (Mamba-1) nor scalar
(Mamba-2-style) decay collapses at N = 64. Decision: **go, with the question reframed** as a boundary result —
"small byte-level selective SSMs neither collapse nor use context beyond their training length, because the memory
horizons they learn are short and independent of training length" — confirmed with seeds (stage 5a), plus a test of
whether collapse appears once long memory is forced (stage 5b).

## Stage 5 results

**5a, seeds.** Seeds 1–2 for every pilot configuration ([`results/research/aggregate.md`](../results/research/aggregate.md),
`plots/research/final_context_effect.png`). The pilot holds: every Mamba configuration has a context effect within
±0.003 bits per byte up to 32×; RoPE and NoPE collapse; ALiBi −0.041 ± 0.021 (95% CI). Two runs at L = 1024 died at step
2,400 without a traceback and were re-run.

**5b, passkey (fixed RECALL position).** An 8-letter key stored 128–494 bytes before a RECALL block that ends every
512-byte training window; 6,000 steps, seed 0.

| model | keys recalled at gap 256 / 512 / 1024 | context effect 8–16× on plain text | P(RECALL) > 0.5 from position |
|---|---|---|---|
| Mamba, N = 16 | 0 / 0 / 0% | +32.1 | 508 |
| Mamba, N = 64 | 0 / 0 / 0% | +30.7 | 503 |
| Mamba, scalar decay, N = 16 | 0 / 0 / 0% | +24.1 | 406 |
| Mamba, scalar decay, N = 64 | 0 / 0 / 0% | +41.4 | 460 |
| Mamba LTI, N = 16 | 0 / 0 / 0% | +4.9 | 485 |
| Transformer, ALiBi | 100 / 99 / 14% | +6.3 | 485 |
| Transformer, RoPE | 0 / 0 / 0% | +2.9 | 503 (mean 0.16 after 512) |

The RECALL marker always sat at position 503. The models learned to predict it from position, and beyond the training
length predict it on ordinary text (~30 bits per byte each time). State RMS grows 9–16× from 1K to 16K (≤ 1.5× after
plain training). Control: adding the STORE marker every training window contained does not change the collapse
(`analysis/passkey_text_control.py`).

**5c, random RECALL position.** Same as 5b with the RECALL block placed at random. Mamba N = 16: +32.1 → +15.2 at 8–16×,
onset 508 → 753; scalar decay N = 64: +41.4 → +17.9, onset 460 → 569. Recall still at chance. Every window still has
exactly one RECALL, so the probability of meeting it still rises to 1 by the window's end.

**Conclusion for RQ3.** Not answered: no Mamba variant learned the recall in this budget, so long memory was never
forced. What the experiment does show is that a training signal tied to absolute position within the window makes
Mamba build a non-forgetting counter, and that counter produces collapse beyond the training length.

## Stage checklist

| Stage | Goal | Status |
|---|---|---|
| 0 | Correct the record: README claims, units, memory cap, withdrawn findings | done |
| 1 | Reproducibility: pinned environment, experiment configs, CSV export, environment capture, LICENSE, CITATION | done |
| 2 | Tests and tooling: deterministic kernel mode, GPU equivalence, long-sequence stability, optimizer-inclusive throughput | done |
| 3 | Seed variance and per-architecture learning-rate tuning on the research models | done |
| 4 | Pilot of H1–H3 (one seed) with baselines; go/no-go | done: go, reframed (see above) |
| 5 | 5a: seeds 1–2 for every pilot configuration; 5b: forced-long-memory boundary test; 5c: position control | done |
| 6 | Analysis, statistics, figures from CSV | done (`analysis/aggregate_seeds.py`, `analysis/passkey_plots.py`, `results/csv/`) |
| 7 | Manuscript, release tag | draft in `docs/paper/paper.md`; release tag not created |

Before/after records for each stage are in [`CHANGELOG.md`](../CHANGELOG.md).
