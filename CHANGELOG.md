# Changelog

Each entry records what changed, which results it affects, and the numbers before and after.

## Unreleased

### Stage 3: tuning, seed variance, measurement fix

- Learning-rate sweep (6 rates) and 3-seed variance for the research models; see docs/AUDIT.md for the table.
  Chosen: Mamba 1.6e-2, Transformer 8e-3. Seed sd ≈ 0.005 bpc; tuned gap 0.163 bpc.
- Attention gained NoPE / ALiBi position encodings and a chunked implementation for long-context evaluation.
- New measurement, `context_effect`: the same bytes scored with full vs. ~training-length context. The earlier
  position-band comparison was confounded by which text sits at which position (smoke test: -0.199 bpc by bands,
  ~0.000 by the same-bytes measure).
- README finding 2 corrected (audit #21).

### Stage 2: tests and tooling

- Triton backward: under `torch.use_deterministic_algorithms(True)` dB/dC are reduced through per-tile partial buffers
  instead of atomics. Before: run-to-run gradient differences up to 9.2e-5; after (deterministic mode): bitwise
  identical (`results/audit/determinism.json`). Default mode is unchanged, so earlier results are unaffected.
- New tests: deterministic mode, GPU training path vs. token-by-token decoding (fp32, 1e-4) for Mamba / hybrid /
  RetNet stacks, 16K-token forward+backward finiteness, Triton vs. chunked scan at L = 8192, scans at L = 2048 with
  extreme decays, experiment-runner expansion.
- `bench_models` also measures `train_b16_l512_opt` (gradient clipping + AdamW step). Fused-attention Transformer:
  76.3 ms without the optimizer step, 78.3 ms with it (smoke run).

### Stage 1: reproducibility infrastructure

No model, kernel or training logic changed, so no result is affected.

- `requirements-lock.txt` with the exact package versions behind `results/` (torch 2.7.1+cu126, triton-windows
  3.3.1.post21, numpy 2.4.6, ...).
- `runtime.environment()`: git commit, versions, GPU, driver, memory cap, clocks and other GPU processes, recorded in
  every benchmark JSON (`env`) and every training run's `config.json` (`environment`).
- `experiments/run.py` + `configs/`: resumable experiment grids from JSON (tested in `tests/test_experiments.py`).
- `analysis/export_csv.py`: all results as CSV in `results/csv/`.
- `docs/BENCHMARK_PROTOCOL.md`, MIT `LICENSE`, `CITATION.cff`.

### Stage 0: correct the record

No code paths or experiments changed; only claims, labels and figures. Results files are unchanged.

| Claim | Before (commit 2b31138) | After |
|---|---|---|
| Memory limit | "runs out of the 4 GB" | PyTorch was capped at 2.97 GiB (other applications held VRAM); all OOM points are relative to that cap |
| Quadratic wall | "Hand-written attention runs out at 8K; Mamba runs 128K" | Compared fused vs. fused: attention faster up to ~32K and 2.5× faster to train at 512; Mamba 3.2× faster at 128K. Old claim withdrawn |
| Quality ranking | "Quality is a draw, hybrid wins" | Same numbers, labelled preliminary (single seed, untuned, not converged) |
| Recall finding | "Exact recall is attention's strength" (Mamba 7.8%) | Withdrawn: Mamba reached 52.9% with 9,000 steps (`results/audit/mqar_budget.json`) |
| Selection ablation | "Selection matters (+0.047 bpc)" | Preliminary: the no-selection model has 7.5% fewer parameters and one seed |
| Metric name | "bpc" | Defined as bits per byte |

Added: `docs/AUDIT.md`, `analysis/audit_checks.py`, `results/audit/`.

## 0.2.0 (commit 2b31138)

Rebuild of the original prototype: from-scratch Mamba, Transformer, RetNet and hybrid, scan algorithms, Triton kernels,
enwik8 experiments, benchmarks, synthetic tasks, 136 tests.
