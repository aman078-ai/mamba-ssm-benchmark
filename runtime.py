"""Device setup shared by all scripts."""

import os

import torch


def setup_device(headroom_mb=256, verbose=True):
    """Pick CUDA if available, enable TF32, and cap PyTorch at the currently free VRAM.

    On Windows the NVIDIA driver can silently spill GPU allocations into system RAM
    ("sysmem fallback"), which turns an out-of-memory condition into a 100x slowdown.
    Capping the caching allocator makes PyTorch raise a clean OutOfMemoryError instead,
    which the benchmarks record as the memory wall.  GPU_MEM_CAP_MB lowers the cap further
    (useful when several jobs share the GPU).
    """
    if not torch.cuda.is_available():
        return "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    free, total = torch.cuda.mem_get_info()
    cap = max(0.1, (free - headroom_mb * 2**20) / total)
    if os.environ.get("GPU_MEM_CAP_MB"):
        cap = min(cap, int(os.environ["GPU_MEM_CAP_MB"]) * 2**20 / total)
    torch.cuda.set_per_process_memory_fraction(min(cap, 0.98))
    if verbose:
        print(f"[device] {torch.cuda.get_device_name(0)}: {total / 2**30:.2f} GiB total, "
              f"PyTorch capped at {cap * total / 2**30:.2f} GiB")
    return "cuda"
