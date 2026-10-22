"""Full-model speed and memory vs sequence length (batch 1, bf16 autocast, matched ~4.7M params).

Measures inference forward and one training step (forward + backward) for each architecture until
it runs out of GPU memory, which marks the memory wall.

    python -m benchmarks.bench_models
"""

import argparse

import torch
import torch.nn.functional as F

from benchmarks.common import env_info, measure, save
from models.lm import build
from runtime import setup_device

VARIANTS = {
    "transformer": dict(name="transformer"),
    "transformer_sdpa": dict(name="transformer", attn_impl="sdpa"),   # PyTorch fused attention, reference
    "mamba": dict(name="mamba"),                                       # Triton scan
    "mamba_torchscan": dict(name="mamba", scan="chunked"),             # pure-PyTorch scan
    "retnet": dict(name="retnet"),
    "hybrid": dict(name="hybrid"),
}


# The pure-PyTorch scan runs a Python loop per channel block; cap it to keep the benchmark short.
MAX_LEN = {"mamba_torchscan": 16384}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="256,512,1024,2048,4096,8192,16384,32768,65536,131072")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--out", default="results/bench/models.json")
    args = ap.parse_args(argv)
    device = setup_device()
    lengths = [int(v) for v in args.lengths.split(",")]
    results = {"env": env_info(), "runs": []}
    for vname in args.variants.split(","):
        spec = dict(VARIANTS[vname])
        model = build(spec.pop("name"), **spec).to(device)
        n_params = model.num_params()
        # training throughput at the enwik8 training shape (batch 16 x 512)
        xb = torch.randint(0, 256, (16, 512), device=device)

        def train_b16():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(xb).float().view(-1, 256), xb.view(-1))
            loss.backward()
            model.zero_grad(set_to_none=True)

        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)

        def train_b16_opt():                    # a full optimizer step, as in real training
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(xb).float().view(-1, 256), xb.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

        # "train_b16_l512" (fwd + bwd only) is kept for comparison with earlier results;
        # "train_b16_l512_opt" adds gradient clipping and the AdamW step.
        for key, fn in (("train_b16_l512", train_b16), ("train_b16_l512_opt", train_b16_opt)):
            tb = measure(fn, budget_s=2.0, max_repeats=10)
            if not tb.get("oom"):
                tb["tokens_per_s"] = 16 * 512 / tb["ms"] * 1e3
                print(f"{vname:17s} {key}: {tb['ms']:.1f} ms = {tb['tokens_per_s'] / 1e3:.1f}k tokens/s, "
                      f"{tb['peak_mb']:.0f} MB", flush=True)
            results.setdefault(key, {})[vname] = tb
        del opt
        del xb
        stop = {"fwd": False, "train": False}
        for L in lengths:
            if L > MAX_LEN.get(vname, 1 << 30):
                break
            x = torch.randint(0, 256, (1, L), device=device)

            def fwd():
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    model(x)

            def train_step():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = F.cross_entropy(model(x).float().view(-1, 256), x.view(-1))
                loss.backward()
                model.zero_grad(set_to_none=True)

            rec = {"variant": vname, "L": L, "params": n_params}
            for key, fn in (("fwd", fwd), ("train", train_step)):
                rec[key] = {"oom": True, "skipped": True} if stop[key] else measure(fn, budget_s=0.5, max_repeats=10)
                stop[key] = stop[key] or rec[key].get("oom", False)
            results["runs"].append(rec)
            show = lambda r: "OOM" if r.get("oom") else f"{r['ms']:9.2f} ms {r['peak_mb']:8.1f} MB"
            print(f"{vname:17s} L={L:6d}  fwd {show(rec['fwd'])}   train {show(rec['train'])}", flush=True)
            if stop["fwd"] and stop["train"]:
                break
        del model
        torch.cuda.empty_cache()
    save(results, args.out)


if __name__ == "__main__":
    main()
