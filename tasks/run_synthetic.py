"""Train/evaluate small models on the synthetic tasks.

    python -m tasks.run_synthetic --suite mqar            # recall capacity sweep
    python -m tasks.run_synthetic --suite selective       # selection ablation

Each run appends one JSON line to results/synthetic/<suite>.jsonl.
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from models.lm import LM, LMConfig
from runtime import setup_device
from tasks import synthetic

# Two-layer-transformer-sized models (Mamba blocks are ~half the params of an attention+MLP layer).
ARCHS = {
    "transformer": dict(layers="AA"),
    "mamba": dict(layers="MMMM"),
    "retnet": dict(layers="RR", mlp_ratio=3.5),
    "hybrid": dict(layers="MMA"),
    "mamba_lti": dict(layers="MMMM", selective=False),
}


def make_batch(task, batch, kw, gen, device):
    x, y = getattr(synthetic, task)(batch, generator=gen, **kw)
    return x.to(device), y.to(device)


@torch.no_grad()
def accuracy(model, task, kw, device, n=1024, batch=128, seed=1234):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    correct = total = 0
    for _ in range(math.ceil(n / batch)):
        x, y = make_batch(task, batch, kw, gen, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            pred = model(x).argmax(-1)
        m = y != synthetic.IGNORE
        correct += (pred[m] == y[m]).sum().item()
        total += m.sum().item()
    model.train()
    return correct / total


def train(arch, task, task_kw, d_model, steps, batch, lr, device, seed=0, log_every=0):
    torch.manual_seed(seed)
    cfg = LMConfig(vocab_size=synthetic.TASK_VOCAB[task], d_model=d_model, n_head=max(1, d_model // 32),
                   **ARCHS[arch])
    model = LM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.98))
    gen = torch.Generator().manual_seed(seed)
    warm = max(1, steps // 20)
    t0 = time.perf_counter()
    for step in range(steps):
        lr_t = lr * min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / steps)))
        for gp in opt.param_groups:
            gp["lr"] = lr_t
        x, y = make_batch(task, batch, task_kw, gen, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(x)
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1),
                               ignore_index=synthetic.IGNORE)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if log_every and step % log_every == 0:
            print(f"    step {step} loss {loss.item():.3f}", flush=True)
    return model, time.perf_counter() - t0, loss.item()


def append(path, rec):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def eval_lengths(model, task, base_kw, lengths, device):
    """Accuracy at other sequence lengths; halves the batch on OOM, None if even batch 1 fails."""
    out = {}
    for L in lengths:
        n = 256 if L <= 1024 else 64
        bs = max(1, min(128, 2 ** 15 // L))
        out[L] = None
        while bs >= 1:
            try:
                out[L] = accuracy(model, task, {**base_kw, "seq_len": L}, device, n=n, batch=bs)
                break
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                bs //= 2
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["mqar", "selective"], required=True)
    ap.add_argument("--archs", default="")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--out", default="results/synthetic")
    args = ap.parse_args(argv)
    device = setup_device()
    path = os.path.join(args.out, f"{args.suite}.jsonl")

    if args.suite == "mqar":
        archs = args.archs.split(",") if args.archs else ["transformer", "mamba", "retnet", "hybrid"]
        steps = args.steps or 3000
        for n_kv in (4, 8, 16, 32):
            for arch in archs:
                kw = dict(seq_len=128, n_kv=n_kv)
                model, secs, loss = train(arch, "mqar", kw, 64, steps, 64, 3e-3, device)
                rec = dict(task="mqar", arch=arch, d_model=64, seq_len=128, n_kv=n_kv, steps=steps,
                           params=model.num_params(), final_loss=loss, train_s=secs,
                           acc=accuracy(model, "mqar", kw, device))
                if n_kv == 8:   # recall across longer gaps than seen in training
                    rec["acc_by_length"] = eval_lengths(model, "mqar", {"n_kv": n_kv},
                                                        [128, 256, 512, 1024, 2048, 4096], device)
                append(path, rec)

    elif args.suite == "selective":
        archs = args.archs.split(",") if args.archs else ["mamba", "mamba_lti", "transformer", "retnet", "hybrid"]
        steps = args.steps or 3000
        kw = dict(seq_len=128, n_tokens=8)
        for arch in archs:
            model, secs, loss = train(arch, "selective_copying", kw, 64, steps, 64, 3e-3, device)
            os.makedirs(os.path.join(args.out, "models"), exist_ok=True)
            torch.save({"cfg": model.cfg.to_dict(), "state_dict": model.state_dict()},
                       os.path.join(args.out, "models", f"selective_{arch}.pt"))
            append(path, dict(task="selective_copying", arch=arch, d_model=64, seq_len=128, n_tokens=8,
                              steps=steps, params=model.num_params(), final_loss=loss, train_s=secs,
                              acc=accuracy(model, "selective_copying", kw, device),
                              acc_by_length=eval_lengths(model, "selective_copying", {"n_tokens": 8},
                                                         [128, 256, 512, 1024, 2048, 4096], device)))


if __name__ == "__main__":
    main()
