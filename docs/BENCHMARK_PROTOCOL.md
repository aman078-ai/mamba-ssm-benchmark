# Benchmark and training protocol

Rules for producing numbers that are comparable across runs and machines. Results that break a rule must say so.

## Environment
- Record the environment with every result (`runtime.environment()`: git commit and dirty flag, Python, torch, CUDA,
  Triton, GPU, driver, PyTorch memory cap, SM/memory clocks, other GPU processes). Benchmarks store it under `env`,
  training runs under `environment` in `config.json`.
- Install from `requirements-lock.txt` to reproduce the exact package set.

## Timing (benchmarks/common.py)
- Close other GPU applications first. If any remain, they appear in `env.other_gpu_processes`; say so when reporting.
- Laptop: plugged in, high-performance power mode, no other heavy work. Let the GPU idle for a minute between suites.
- Warm-up calls before timing (≥2; ≥20 for CUDA-graph decode), `torch.cuda.synchronize()` around every timed call,
  report the **median** and the interquartile range of repeated calls.
- Timing variability (repeated calls in one process) is not seed variability (different trained models). Report them
  separately and never mix them in one error bar.
- Report kernel class with every speed number: naive (materializes intermediates) or fused. Compare naive with naive and
  fused with fused; a comparison across classes measures kernels, not architectures.

## Memory
- `peak_mb` = `torch.cuda.max_memory_allocated` above what was allocated before the call (activations, gradients and
  workspaces; weights excluded). Reserved memory and process memory are not yet reported (docs/AUDIT.md #16).
- Out-of-memory is relative to PyTorch's memory cap (`env.memory_cap_gib`), not the card's total memory.

## Training comparisons
- Same data order for all models (same sampler seed), same token budget, bf16 autocast, gradient clipping 1.0.
- Tune the learning rate per architecture on the validation split with the same small budget for each, then fix it.
- At least 3 seeds per configuration for any quality claim; report mean ± standard deviation and every seed.
- The test split is evaluated once per final model and never used to choose anything.
- State the control: matched parameters, matched training tokens, or matched compute. They are not interchangeable.

## Determinism
- The Triton backward pass accumulates dB and dC with atomic adds, so gradients differ in the last bits between runs.
  Call `torch.use_deterministic_algorithms(True)` to switch it to a deterministic reduction (slower).
