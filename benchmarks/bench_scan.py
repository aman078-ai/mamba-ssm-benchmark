"""Benchmark the selective-scan implementations in isolation.

For one Mamba layer's scan (B=1, D=512 channels, N=16 states) and growing L, measure forward and
forward+backward time and peak memory of:

    reference      Python loop over time, differentiated by autograd (the original project's approach)
    sequential     same loop, custom backward with recomputation
    hillis_steele  log-depth parallel scan, O(L log L) work
    blelloch       work-efficient parallel scan
    chunked        two-level scan (sequential inside chunks, parallel across)
    triton         fused kernels: state kept in registers, never written to DRAM
"""

import argparse

import torch

from benchmarks.common import env_info, measure, save
from models.scan import selective_scan
from runtime import setup_device

LIMITS = {"reference": 4096, "sequential": 4096}   # slow Python loops: cap their lengths


def make_inputs(L, B=1, D=512, N=16, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(B, L, D, generator=g).to(device).requires_grad_()
    delta = (torch.rand(B, L, D, generator=g) * 0.1 + 1e-3).to(device).requires_grad_()
    A = (-torch.arange(1, N + 1, dtype=torch.float32).repeat(D, 1)).to(device).requires_grad_()
    Bm = torch.randn(B, L, N, generator=g).to(device).requires_grad_()
    Cm = torch.randn(B, L, N, generator=g).to(device).requires_grad_()
    return x, delta, A, Bm, Cm


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="256,1024,4096,16384,65536")
    ap.add_argument("--scans", default="reference,sequential,hillis_steele,blelloch,chunked,triton")
    ap.add_argument("--out", default="results/bench/scan.json")
    args = ap.parse_args(argv)
    setup_device()
    lengths = [int(v) for v in args.lengths.split(",")]
    results = {"env": env_info(), "config": {"B": 1, "D": 512, "N": 16}, "runs": []}
    for scan in args.scans.split(","):
        for L in lengths:
            if L > LIMITS.get(scan, 1 << 30):
                continue
            ins = make_inputs(L)
            f = lambda: selective_scan(*ins, scan=scan)[0]

            def fwd():
                with torch.no_grad():
                    f()

            def fwd_bwd():
                f().sum().backward()

            rec = {"scan": scan, "L": L, "fwd": measure(fwd, budget_s=0.5),
                   "fwd_bwd": measure(fwd_bwd, budget_s=0.5)}
            results["runs"].append(rec)
            fm, bm = rec["fwd"], rec["fwd_bwd"]
            print(f"{scan:14s} L={L:6d}  fwd {fm.get('ms', float('nan')):9.2f} ms {fm.get('peak_mb', 0):8.1f} MB"
                  f"   fwd+bwd {bm.get('ms', float('nan')):9.2f} ms {bm.get('peak_mb', 0):8.1f} MB", flush=True)
            del ins
            torch.cuda.empty_cache()
    save(results, args.out)


if __name__ == "__main__":
    main()
