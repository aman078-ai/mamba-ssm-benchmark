"""Checks run during the research audit (docs/AUDIT.md).  Each writes results/audit/<check>.json.

    python -m analysis.audit_checks --check mqar_budget     # is the MQAR result a training-budget artifact?
    python -m analysis.audit_checks --check determinism     # run-to-run gradient differences per scan
    python -m analysis.audit_checks --check consistency     # GPU full forward vs. token-by-token decoding
"""

import argparse
import json
import os

import torch

from runtime import setup_device

OUT = "results/audit"


def save(name, obj):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{name}.json")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"saved {path}")


def mqar_budget(device):
    """Pure Mamba on 4-pair MQAR (the setting where the README reported 7.8%), with other
    learning rates and a 3x longer budget."""
    from tasks.run_synthetic import accuracy, train
    kw = dict(seq_len=128, n_kv=4)
    rows = []
    for lr, steps in [(1e-3, 3000), (3e-4, 3000), (3e-3, 9000)]:
        model, secs, loss = train("mamba", "mqar", kw, 64, steps, 64, lr, device)
        rows.append(dict(arch="mamba", d_model=64, n_kv=4, seq_len=128, lr=lr, steps=steps,
                         final_loss=loss, acc=accuracy(model, "mqar", kw, device), train_s=secs))
        print(rows[-1], flush=True)
    save("mqar_budget", {"reference": "README (before audit): mamba, lr 3e-3, 3000 steps -> 7.8% accuracy",
                         "runs": rows})


def determinism(device):
    from models.scan import selective_scan
    torch.manual_seed(0)
    B, L, D, N = 4, 512, 256, 16
    x = torch.randn(B, L, D, device=device, requires_grad=True)
    dt = (torch.rand(B, L, D, device=device) * 0.1).requires_grad_()
    A = (-torch.rand(D, N, device=device) * 4).requires_grad_()
    Bm = torch.randn(B, L, N, device=device, requires_grad=True)
    Cm = torch.randn(B, L, N, device=device, requires_grad=True)

    def grads(scan):
        y, _ = selective_scan(x, dt, A, Bm, Cm, scan=scan)
        return torch.autograd.grad(y.square().sum(), (x, dt, A, Bm, Cm))

    out = {}
    for det in (False, True):
        torch.use_deterministic_algorithms(det, warn_only=True)
        for scan in ("triton", "chunked"):
            g1, g2 = grads(scan), grads(scan)
            out[f"{scan}{'_deterministic' if det else ''}"] = {
                n: (a - b).abs().max().item() for n, a, b in zip(("x", "delta", "A", "B", "C"), g1, g2)}
    torch.use_deterministic_algorithms(False)
    print(json.dumps(out, indent=1))
    save("determinism", out)


def consistency(device):
    from analysis.evaluate_lm import load
    from data.enwik8 import Enwik8
    ds = Enwik8()
    out = {}
    for run in ("mamba", "hybrid"):
        if not os.path.exists(f"results/enwik8/{run}/model.pt"):
            continue
        m = load(f"results/enwik8/{run}", device)
        x = torch.tensor(list(bytes(ds.splits["test"][:300])), device=device)[None]
        for amp in (False, True):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                full = m(x).float()
                st = m.init_state(1, 300, device, torch.float32)
                steps = torch.cat([m(x[:, t:t + 1], st, t).float() for t in range(300)], 1)
            out[f"{run}_{'bf16' if amp else 'fp32'}"] = {
                "max_abs_logit_diff": (full - steps).abs().max().item(),
                "argmax_agreement": (full.argmax(-1) == steps.argmax(-1)).float().mean().item()}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            xl = torch.tensor(list(bytes(ds.splits["test"][:65536])), device=device)[None]
            out[f"{run}_64k_forward_finite"] = bool(torch.isfinite(m(xl)).all().item()) if run == "mamba" else None
    print(json.dumps(out, indent=1))
    save("consistency", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", required=True, choices=["mqar_budget", "determinism", "consistency"])
    args = ap.parse_args()
    device = setup_device(verbose=False)
    {"mqar_budget": mqar_budget, "determinism": determinism, "consistency": consistency}[args.check](device)


if __name__ == "__main__":
    main()
