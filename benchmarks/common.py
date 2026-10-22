"""Timing and memory measurement helpers (CUDA-synchronized, warmed up, median of repeats)."""

import json
import os
import statistics
import time

import torch


def cuda_time(fn, warmup=2, min_repeats=3, max_repeats=20, budget_s=1.0):
    """Median wall-clock ms of fn() with CUDA synchronization; also returns (q25, q75)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    start = time.perf_counter()
    while len(times) < max_repeats and (len(times) < min_repeats or time.perf_counter() - start < budget_s):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    q = statistics.quantiles(times, n=4) if len(times) >= 2 else [times[0]] * 3
    return statistics.median(times), q[0], q[2]


def peak_memory_mb(fn):
    """Peak CUDA memory allocated during fn(), above what was allocated before it."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20


def measure(fn, **kw):
    """Returns dict(ms, q25, q75, peak_mb) or dict(oom=True)."""
    try:
        mem = peak_memory_mb(fn)
        ms, q25, q75 = cuda_time(fn, **kw)
        return {"ms": ms, "q25": q25, "q75": q75, "peak_mb": mem}
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"oom": True}


def save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"saved {path}")


def env_info():
    from runtime import environment
    return environment()
