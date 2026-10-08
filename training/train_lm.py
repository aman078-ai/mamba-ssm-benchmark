"""Train a byte-level LM on enwik8 and log bits-per-character.

    python -m training.train_lm --model mamba --steps 3000
    python -m training.train_lm --model transformer --seq_len 512 --batch_size 16

Writes results/enwik8/<run>/{config.json, log.jsonl, summary.json, model.pt}.
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from data.enwik8 import Enwik8
from models.lm import PRESETS, build
from runtime import setup_device

LN2 = math.log(2)


@torch.no_grad()
def evaluate_bpc(model, ds, split, seq_len, batch_size, device, max_bytes=None, amp=True):
    model.eval()
    total, count = 0.0, 0
    for x, y in ds.eval_batches(split, batch_size, seq_len, device, max_bytes):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device == "cuda"):
            logits = model(x)
        total += F.cross_entropy(logits.float().reshape(-1, 256), y.reshape(-1), reduction="sum").item()
        count += y.numel()
    model.train()
    return total / count / LN2


def lr_at(step, total, peak, warmup, floor):
    if step < warmup:
        return peak * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * p))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mamba", choices=list(PRESETS))
    ap.add_argument("--run", default=None, help="run name (default: model name)")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--stop_step", type=int, default=0,
                    help="stop early at this step but keep the LR schedule of --steps (for ablations)")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--min_lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--eval_bytes", type=int, default=1_000_000)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--override", default="{}", help='JSON LMConfig overrides, e.g. {"d_block":128}')
    ap.add_argument("--out", default="results/enwik8")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    device = setup_device()
    amp = not args.no_amp and device == "cuda"

    run = args.run or args.model
    out_dir = os.path.join(args.out, run)
    os.makedirs(out_dir, exist_ok=True)
    ds = Enwik8()
    model = build(args.model, **json.loads(args.override)).to(device)
    n_params = model.num_params()
    print(f"[{run}] layers={model.cfg.layers} non-embedding params={n_params:,} device={device}")

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "A_log" not in n]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2 or "A_log" in n]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=device == "cuda")
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"args": vars(args), "model_cfg": model.cfg.to_dict(), "non_embedding_params": n_params}, f, indent=2)

    log = open(os.path.join(out_dir, "log.jsonl"), "w", buffering=1)
    gen = torch.Generator().manual_seed(args.seed)
    tokens_per_step = args.batch_size * args.seq_len
    train_time, best_val = 0.0, float("inf")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps or (args.stop_step and step == args.stop_step):
            val = evaluate_bpc(model, ds, "valid", args.seq_len, 2 * args.batch_size, device,
                               args.eval_bytes, amp)
            best_val = min(best_val, val)
            rec = {"step": step, "tokens": step * tokens_per_step, "train_time_s": train_time, "val_bpc": val}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"[{run}] step {step:5d}  val_bpc {val:.4f}  ({train_time / 60:.1f} min)")
        if step == args.steps or (args.stop_step and step == args.stop_step):
            break

        t0 = time.perf_counter()
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr, args.warmup, args.min_lr)
        x, y = ds.random_batch("train", args.batch_size, args.seq_len, device, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(x)
        loss = F.cross_entropy(logits.float().reshape(-1, 256), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        if device == "cuda":
            torch.cuda.synchronize()
        train_time += time.perf_counter() - t0
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"non-finite loss at step {step}")

        if step % args.log_every == 0:
            rec = {"step": step, "tokens": (step + 1) * tokens_per_step, "train_time_s": train_time,
                   "train_bpc": loss.item() / LN2, "grad_norm": gnorm.item(), "lr": opt.param_groups[0]["lr"]}
            log.write(json.dumps(rec) + "\n")
            if step % (args.log_every * 4) == 0:
                tps = (step + 1) * tokens_per_step / max(train_time, 1e-9)
                print(f"[{run}] step {step:5d}  train_bpc {rec['train_bpc']:.3f}  {tps / 1e3:.1f}k tok/s")

    peak_mem = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0
    summary = {"run": run, "model": args.model, "layers": model.cfg.layers, "non_embedding_params": n_params,
               "steps": step, "tokens": step * tokens_per_step, "seq_len": args.seq_len,
               "final_val_bpc": val, "best_val_bpc": best_val, "train_time_min": train_time / 60,
               "tokens_per_sec": step * tokens_per_step / train_time, "peak_train_mem_mb": peak_mem}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    torch.save({"cfg": model.cfg.to_dict(), "state_dict": model.state_dict()}, os.path.join(out_dir, "model.pt"))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
