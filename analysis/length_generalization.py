"""Length-generalization measurements for one trained run (research stages 4-6).

    python -m analysis.length_generalization --run_dir results/research/<experiment>/<run>

Writes <run_dir>/length_generalization.json with, on the VALIDATION split (the test split is reserved for final runs):
  bpc_by_length     bits per byte with non-overlapping windows of each length (training length x 1, 2, 4, ...)
  position_loss     bits per byte at each position of a long window, log-spaced bins
  context_effect    for the SAME bytes: loss with the full long context minus loss with ~one training length of
                    context, per band of positions (1-2x, 2-4x, ... the training length); < 0 = extra context helps,
                    > 0 = it hurts.  Unlike position_loss, this is not confounded by which text sits at which position.
  mamba (Mamba layers only):
    horizons        per layer: distribution of each (channel, state) mode's memory horizon
                    h = 1 / (|A_n| * mean_t Δ_t[d])  in bytes (the e-folding time of exp(Δ A); a stationary
                    approximation that ignores how Δ varies with the input)
    state_rms       per layer: RMS of the SSM state after 1K, 2K, 4K, ... bytes of one long sequence, computed by
                    chunked prefill with exact state passing (tests the "state collapse" explanation)
"""

import argparse
import json
import math
import os

import torch
import torch.nn.functional as F

from data.enwik8 import Enwik8
from models.lm import LM, LMConfig
from models.mamba import Mamba
from runtime import environment, setup_device

LN2 = math.log(2)


def load(run_dir, device, attn_impl="chunked"):
    ck = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu")
    cfg = dict(ck["cfg"])
    cfg["attn_impl"] = attn_impl          # same math as training; bounded memory for long contexts
    model = LM(LMConfig(**cfg))
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def bpc(model, ds, L, n_bytes, device):
    total, count = 0.0, 0
    bs = max(1, 32768 // L)
    for x, y in ds.eval_batches("valid", bs, L, device, n_bytes):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(x)
        total += F.cross_entropy(logits.float().reshape(-1, 256), y.reshape(-1), reduction="sum").item()
        count += y.numel()
    return total / count / LN2


@torch.no_grad()
def position_loss(model, ds, L, n_windows, device, n_bins=48):
    sums = torch.zeros(L, device=device)
    n = 0
    for x, y in ds.eval_batches("valid", 1, L, device, n_windows * L):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(x)
        sums += F.cross_entropy(logits.float()[0], y[0], reduction="none")
        n += 1
    per_pos = (sums / n / LN2).cpu()
    edges = torch.logspace(0, math.log10(L), n_bins + 1).long().unique()
    return [{"start": int(a), "end": int(b), "bpc": per_pos[a:b].mean().item()}
            for a, b in zip(edges[:-1], edges[1:]) if b > a]


@torch.no_grad()
def context_effect(model, ds, L_train, W, n_windows, device):
    """Same bytes, two contexts.  For bytes at positions in each band of a W-byte window, compare the loss with the
    full context from the window start against the loss with only ~L_train bytes of context (sliding windows of
    length L_train, stride L_train/2, keeping the second half so every byte sees L_train/2..L_train bytes).
    Returns mean(full - short) in bits per byte per band: < 0 = extra context helps, > 0 = it hurts."""
    half = L_train // 2
    bands = {"1-2x": (L_train, 2 * L_train), "2-4x": (2 * L_train, 4 * L_train), "4-8x": (4 * L_train, 8 * L_train),
             "8-16x": (8 * L_train, 16 * L_train), "16-32x": (16 * L_train, 32 * L_train)}
    bands = {k: v for k, v in bands.items() if v[1] <= W}
    diffs = {k: [] for k in bands}
    for x, y in ds.eval_batches("valid", 1, W, device, n_windows * W):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            full = F.cross_entropy(model(x).float()[0], y[0], reduction="none")
            starts = torch.arange(0, W - L_train + 1, half, device=device)
            idx = starts[:, None] + torch.arange(L_train, device=device)[None]
            xs, ys = x[0][idx], y[0][idx]
            short_all = torch.cat([F.cross_entropy(model(xs[i:i + 64]).float().reshape(-1, 256), ys[i:i + 64].reshape(-1),
                                                   reduction="none").view(-1, L_train) for i in range(0, len(xs), 64)])
        short = torch.full((W,), float("nan"), device=device)
        keep = idx[:, half:]                                    # positions with half..L_train bytes of context
        short[keep.reshape(-1)] = short_all[:, half:].reshape(-1)
        for k, (lo, hi) in bands.items():
            d = (full[lo:hi] - short[lo:hi])
            d = d[~torch.isnan(d)]
            diffs[k].append(d.mean().item() / LN2)
    return {k: sum(v) / len(v) for k, v in diffs.items() if v}


@torch.no_grad()
def mamba_stats(model, ds, L_long, train_len, device, chunk=1024):
    layers = [b.mixer for b in model.blocks if isinstance(b.mixer, Mamba)]
    if not layers:
        return None
    x = torch.tensor(list(bytes(ds.splits["valid"][:L_long])), device=device)[None]
    # horizons from the step sizes seen on held-out text
    for m in layers:
        m.record_delta = True
        m.delta_sum, m.delta_n = 0, 0
    states = model.init_state(1, L_long, device, torch.float32)
    rms = {i: [] for i in range(len(layers))}
    checkpoints = []
    for p0 in range(0, L_long, chunk):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            model(x[:, p0:p0 + chunk], states, p0)
        for m in layers:
            m.delta_sum = m.delta_sum + m.last_delta.float().sum(dim=(0, 1))
            m.delta_n += m.last_delta.shape[1]
        end = min(p0 + chunk, L_long)
        if end & (end - 1) == 0 or end == L_long:          # powers of two
            checkpoints.append(end)
            mamba_states = [s for s, b in zip(states, model.blocks) if isinstance(b.mixer, Mamba)]
            for i, s in enumerate(mamba_states):
                rms[i].append(s.ssm.float().pow(2).mean().sqrt().item())
    horizons = []
    for m in layers:
        mean_delta = (m.delta_sum / m.delta_n).clamp_min(1e-8)             # (d_inner,)
        A = torch.exp(m.A_log.float()).expand(-1, m.d_state)                # |A|, (d_inner, N)
        h = 1.0 / (A * mean_delta[:, None])                                 # (d_inner, N) in bytes
        q = torch.quantile(h.flatten(), torch.tensor([0.1, 0.5, 0.9, 0.99], device=h.device))
        horizons.append({"p10": q[0].item(), "median": q[1].item(), "p90": q[2].item(), "p99": q[3].item(),
                         "max": h.max().item(), "frac_above_train_len": (h > train_len).float().mean().item(),
                         # median over channels for each state index n (S4D-real init: A_n = -(n+1))
                         "median_by_state_index": h.median(dim=0).values.tolist()})
        m.record_delta = False
        for attr in ("last_delta", "delta_sum", "delta_n"):
            if hasattr(m, attr):
                delattr(m, attr)
    return {"horizons": horizons, "state_rms": {"positions": checkpoints, "by_layer": [rms[i] for i in rms]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--max_len", type=int, default=16384)
    ap.add_argument("--n_bytes", type=int, default=1_000_000, help="validation bytes per evaluation length")
    ap.add_argument("--position_windows", type=int, default=16)
    args = ap.parse_args()
    device = setup_device(verbose=False)
    ds = Enwik8()
    cfg = json.load(open(os.path.join(args.run_dir, "config.json")))
    train_len = cfg["args"]["seq_len"]
    model = load(args.run_dir, device)
    lengths = [train_len * 2 ** k for k in range(20) if train_len * 2 ** k <= args.max_len]
    out = {"split": "valid", "train_len": train_len, "n_bytes_per_length": args.n_bytes, "bpc_by_length": {}}
    for L in lengths:
        try:
            out["bpc_by_length"][L] = bpc(model, ds, L, max(args.n_bytes, L + 1), device)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            out["bpc_by_length"][L] = None
        print(f"  L={L:6d}  bpc={out['bpc_by_length'][L]}", flush=True)
    out["position_loss"] = position_loss(model, ds, args.max_len, args.position_windows, device)
    out["context_effect"] = context_effect(model, ds, train_len, args.max_len, args.position_windows, device)
    print("  context effect (full - short context, bits/byte):", {k: round(v, 4) for k, v in out["context_effect"].items()})
    out["mamba"] = mamba_stats(model, ds, args.max_len, train_len, device)
    out["environment"] = environment()
    with open(os.path.join(args.run_dir, "length_generalization.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {args.run_dir}/length_generalization.json")


if __name__ == "__main__":
    main()
