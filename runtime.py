"""Device setup and environment capture shared by all scripts."""

import datetime
import os
import platform
import subprocess
import sys

import torch

_MEMORY_CAP_GIB = None


def setup_device(headroom_mb=256, verbose=True):
    """Pick CUDA if available, enable TF32, and cap PyTorch at the currently free VRAM.

    On Windows the NVIDIA driver can silently spill GPU allocations into system RAM
    ("sysmem fallback"), which turns an out-of-memory condition into a 100x slowdown.
    Capping the caching allocator makes PyTorch raise a clean OutOfMemoryError instead,
    which the benchmarks record as the memory wall.  GPU_MEM_CAP_MB lowers the cap further
    (useful when several jobs share the GPU).  The cap is recorded by environment().
    """
    global _MEMORY_CAP_GIB
    if not torch.cuda.is_available():
        return "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    free, total = torch.cuda.mem_get_info()
    cap = max(0.1, (free - headroom_mb * 2**20) / total)
    if os.environ.get("GPU_MEM_CAP_MB"):
        cap = min(cap, int(os.environ["GPU_MEM_CAP_MB"]) * 2**20 / total)
    cap = min(cap, 0.98)
    torch.cuda.set_per_process_memory_fraction(cap)
    _MEMORY_CAP_GIB = cap * total / 2**30
    if verbose:
        print(f"[device] {torch.cuda.get_device_name(0)}: {total / 2**30:.2f} GiB total, "
              f"PyTorch capped at {_MEMORY_CAP_GIB:.2f} GiB")
    return "cuda"


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def environment():
    """Everything needed to judge whether a measurement is comparable to another one."""
    env = {
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_run(["git", "status", "--porcelain", "--untracked-files=no"])),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    try:
        import triton
        env["triton"] = triton.__version__
    except Exception:
        env["triton"] = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        env.update(gpu=props.name, vram_gib=props.total_memory / 2**30, memory_cap_gib=_MEMORY_CAP_GIB,
                   driver=_run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
                   gpu_clocks_mhz=_run(["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem", "--format=csv,noheader"]),
                   other_gpu_processes=[line.split(",")[-1].strip().split("\\")[-1] for line in
                                        _run(["nvidia-smi", "--query-compute-apps=pid,process_name",
                                              "--format=csv,noheader"]).splitlines()
                                        if line.strip() and str(os.getpid()) not in line.split(",")[0]])
    return env
