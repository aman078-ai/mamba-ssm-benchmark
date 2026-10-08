"""Sample text from the trained enwik8 models, side by side, with speed and cache/state size.

    python generate.py --prompt "The history of " --n 400
    python generate.py --models mamba --temperature 0.8
"""

import argparse
import sys
import time

import torch

from analysis.evaluate_lm import load
from models.lm import LM
from runtime import setup_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="transformer,mamba,retnet,hybrid")
    ap.add_argument("--prompt", default="The history of the ")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    device = setup_device(verbose=False)
    prompt = torch.tensor([list(args.prompt.encode("utf-8"))], device=device)
    report = []
    for name in args.models.split(","):
        model = load(f"results/enwik8/{name}", device)
        torch.manual_seed(args.seed)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            model.generate(prompt, 8, args.temperature, args.top_k)          # warm-up
            torch.cuda.synchronize() if device == "cuda" else None
            t0 = time.perf_counter()
            out = model.generate(prompt, args.n, args.temperature, args.top_k)
            torch.cuda.synchronize() if device == "cuda" else None
        secs = time.perf_counter() - t0
        state_kb = LM.state_nbytes(model.init_state(1, prompt.shape[1] + args.n, device, torch.bfloat16),
                                   prompt.shape[1] + args.n) / 1024
        text = bytes(out[0].tolist()).decode("utf-8", errors="replace")
        header = f"-- {name}  ({args.n / secs:.0f} tokens/s, cache/state after {out.shape[1]} tokens: {state_kb:.0f} KB)"
        print(header + "\n" + text + "\n")
        report.append((name, args.n / secs, state_kb, text))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(f"Prompt: `{args.prompt}` · temperature {args.temperature} · top-k {args.top_k} · seed {args.seed}\n\n")
            for name, tps, kb, text in report:
                f.write(f"**{name}** ({tps:.0f} tokens/s, cache/state {kb:.0f} KB)\n\n```\n{text}\n```\n\n")


if __name__ == "__main__":
    main()
