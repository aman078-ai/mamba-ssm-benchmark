"""What does selection learn?  Inspect the input-dependent step size Δ of trained Mamba models.

Large Δ_t means exp(Δ_t A) ≈ 0: the state is overwritten by the current token.  Small Δ_t means
the state is carried through and the token is mostly ignored.

1. enwik8 Mamba: Δ per byte on a prose snippet (heatmap) and averaged by character class.
2. Selective-copying models: Δ on noise vs data vs marker tokens, selective vs LTI.

    python -m analysis.delta_analysis
"""

import json
import os
import re

import torch

from analysis.evaluate_lm import load
from data.enwik8 import Enwik8
from models.lm import LM, LMConfig
from runtime import setup_device
from tasks import synthetic


def mamba_layers(model):
    return [b.mixer for b in model.blocks if b.kind == "M"]


@torch.no_grad()
def record(model, x):
    layers = mamba_layers(model)
    for m in layers:
        m.record_delta = True
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
        model(x)
    out = [m.last_delta.float().mean(-1) for m in layers]       # each (B, T): mean over channels
    for m in layers:
        m.record_delta = False
        del m.last_delta
    return torch.stack(out, 1)                                   # (B, layers, T)


CLASSES = {
    "space": lambda c, p: c == " ",
    "newline": lambda c, p: c == "\n",
    "word start": lambda c, p: c.isalpha() and not p.isalpha(),
    "inside word": lambda c, p: c.isalpha() and p.isalpha(),
    "digit": lambda c, p: c.isdigit(),
    "punctuation": lambda c, p: c in ".,;:!?'\"()",
    "markup": lambda c, p: c in "<>[]{}|=/&#*",
}


def main():
    device = setup_device()
    out = {}
    ds = Enwik8()
    model = load("results/enwik8/mamba", device)

    # pick a prose snippet from the test split and warm the state up on the preceding 1 KB
    test = ds.text("test", 0, 2_000_000)
    m = re.search(r"\n\n'''[A-Z][^\n]{200,}", test)
    start = m.start() + 2 if m else 100_000
    snippet_len, warm = 150, 1024
    raw = bytes(ds.splits["test"][start - warm:start + snippet_len])
    x = torch.tensor(list(raw), device=device)[None]
    d = record(model, x)[0, :, warm:]                            # (layers, snippet)
    out["text"] = raw[warm:].decode("utf-8", errors="replace")[:snippet_len]
    out["delta_rel"] = (d / d.mean(1, keepdim=True)).tolist()

    # character-class statistics over 64 KB of test text
    raw = bytes(ds.splits["test"][start:start + 65536])
    x = torch.tensor(list(raw), device=device)[None]
    d = record(model, x)[0]
    d = d / d.mean(1, keepdim=True)
    text = raw.decode("latin-1")
    stats = {}
    for name, fn in CLASSES.items():
        idx = [i for i in range(1, len(text)) if fn(text[i], text[i - 1])]
        if idx:
            stats[name] = {"count": len(idx), "rel_delta_by_layer": d[:, idx].mean(1).tolist()}
    out["class_stats"] = stats
    print(json.dumps({k: [round(v, 2) for v in s["rel_delta_by_layer"]] for k, s in stats.items()}, indent=1))

    # selective copying: does Δ separate noise from data?
    sel = {}
    for arch in ("mamba", "mamba_lti"):
        path = f"results/synthetic/models/selective_{arch}.pt"
        if not os.path.exists(path):
            continue
        ck = torch.load(path, map_location="cpu")
        sm = LM(LMConfig(**ck["cfg"]))
        sm.load_state_dict(ck["state_dict"])
        sm = sm.to(device).eval()
        xs, _ = synthetic.selective_copying(256, seq_len=128, n_tokens=8, generator=torch.Generator().manual_seed(0))
        xs = xs.to(device)
        dl = record(sm, xs)[:, 0]                                 # first Mamba layer, (B, T)
        marker = 17   # n_colors + 1
        sel[arch] = {"noise": dl[xs == 0].mean().item(),
                     "data": dl[(xs > 0) & (xs < marker)].mean().item(),
                     "marker": dl[xs == marker].mean().item()}
        print(arch, sel[arch])
    out["selective"] = sel
    with open("results/delta.json", "w") as f:
        json.dump(out, f)
    print("saved results/delta.json")


if __name__ == "__main__":
    main()
