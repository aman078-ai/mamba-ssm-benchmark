"""Control for stage 5b: is the beyond-L collapse a length failure, or a reaction to text without passkey markers?

    python -m analysis.passkey_text_control --run_dir results/research/<experiment>/<run>

Passkey-trained models are evaluated on long plain text by length_generalization.py, but every training window they
saw contained a STORE block.  Here the same held-out text is scored twice, with and without a STORE block near the
start (as in training), and the loss on ordinary text bytes is reported per band of positions relative to the training
length L.  It also records P(next byte = RECALL marker) on plain text by position: a model that has learned where
the RECALL block sits in a training window predicts the marker from position alone.
Writes <run_dir>/passkey_text_control.json.
"""

import argparse
import json
import math
import os

import torch
import torch.nn.functional as F

from analysis.length_generalization import load
from data.enwik8 import Enwik8
from data.passkey import RECALL, passkey_batch
from runtime import environment, setup_device

LN2 = math.log(2)


@torch.no_grad()
def band_loss(model, x, y, ignore, L, device):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        logits = model(x.to(device)).float()
    nll = F.cross_entropy(logits.transpose(1, 2), y.to(device), reduction="none").cpu()
    keep = ~ignore
    T = y.shape[1]
    out, a = {}, 0
    for b in [L // 2, L] + [L * 2 ** k for k in range(1, 6)]:
        b = min(b, T)
        if b <= a:
            break
        m = keep[:, a:b]
        out[f"{a}-{b}"] = (nll[:, a:b][m].sum() / m.sum() / LN2).item()
        a = b
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--gap", type=int, default=8192)
    ap.add_argument("--n", type=int, default=16)
    args = ap.parse_args()
    device = setup_device(verbose=False)
    cfg = json.load(open(os.path.join(args.run_dir, "config.json")))
    L, key_len = cfg["args"]["seq_len"], cfg["args"].get("passkey_len") or 8
    model = load(args.run_dir, device)
    text = Enwik8().splits["valid"]
    g = torch.Generator().manual_seed(0)
    x, y, m = passkey_batch(text, args.n, args.gap, key_len, g)
    block = key_len + 1
    s0 = 64                                                           # STORE position (prefix = 64)
    ignore = m.clone()
    ignore[:, s0 - 1:s0 + block] = True                               # targets that are STORE / key bytes
    ignore[:, -block - 1:] = True                                     # RECALL block
    with_marker = band_loss(model, x, y, ignore, L, device)
    g2 = torch.Generator().manual_seed(0)                             # same window starts, original text restored
    T1 = x.shape[1] + 1
    starts = torch.randint(0, len(text) - T1 - 1, (args.n,), generator=g2)
    buf = torch.stack([torch.as_tensor(text[int(s):int(s) + T1].astype("int64")) for s in starts])
    plain_loss = band_loss(model, buf[:, :-1], buf[:, 1:], ignore, L, device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        p = model(buf[:, :4 * L].to(device)).float().softmax(-1)[..., RECALL].mean(0).cpu()
    on = (p > 0.5).nonzero()
    recall_pred = {"first_position_p_gt_half": int(on[0]) if len(on) else None,
                   "mean_p_before_L_minus_32": p[:L - 32].mean().item(), "mean_p_after_L": p[L:].mean().item(),
                   "p_by_position": [round(v, 5) for v in p.tolist()]}
    print(f"  P(RECALL) on plain text: { {k: v for k, v in recall_pred.items() if k != 'p_by_position'} }")
    out = {"train_len": L, "recall_marker_prediction": recall_pred, "gap": args.gap, "n": args.n, "with_store_marker": with_marker,
           "plain_text": plain_loss, "environment": environment()}
    for k in with_marker:
        print(f"  {k:>12}: with marker {with_marker[k]:7.3f}   plain {plain_loss[k]:7.3f}")
    with open(os.path.join(args.run_dir, "passkey_text_control.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
