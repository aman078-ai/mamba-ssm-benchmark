"""Autoregressive generation: per-token latency and cache/state memory vs context length.

For context length P, the model holds P tokens of history (KV cache for attention, fixed-size
state for Mamba/retention) and we time one decode step.  The cache is filled with random values:
the cost of a decode step depends on the cache's size, not its contents.

    python -m benchmarks.bench_decode
"""

import argparse

import torch

from benchmarks.common import cuda_time, env_info, save
from models.lm import LM, build
from models.transformer import KVCache
from runtime import setup_device


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", default="128,512,2048,8192,32768,131072,262144")
    ap.add_argument("--models", default="transformer,mamba,retnet,hybrid")
    ap.add_argument("--out", default="results/bench/decode.json")
    args = ap.parse_args(argv)
    device = setup_device()
    contexts = [int(v) for v in args.contexts.split(",")]
    results = {"env": env_info(), "runs": []}
    for name in args.models.split(","):
        model = build(name).to(device).eval()
        for P in contexts:
            rec = {"model": name, "context": P}
            try:
                states = model.init_state(1, P + 1, device, dtype=torch.bfloat16)
                for s in states:
                    if isinstance(s, KVCache):
                        s.k.normal_()
                        s.v.normal_()
                    else:
                        for t in vars(s).values():
                            t.normal_()
                tok = torch.randint(0, 256, (1, 1), device=device)

                def step():
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                        model(tok, states, P)

                ms, q25, q75 = cuda_time(step, warmup=3, min_repeats=10, max_repeats=50, budget_s=0.5)
                rec.update(ms=ms, q25=q25, q75=q75, state_mb=LM.state_nbytes(states, P) / 2**20,
                           tokens_per_s=1e3 / ms)
                try:
                    side = torch.cuda.Stream()
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        for _ in range(3):
                            step()
                    torch.cuda.current_stream().wait_stream(side)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        step()
                    rec["ms_graph"] = cuda_time(graph.replay, warmup=20, min_repeats=100, max_repeats=400, budget_s=2.0)[0]
                    del graph
                except Exception as e:      # capture is best-effort
                    rec["graph_error"] = f"{type(e).__name__}: {e}"[:200]
                del states
            except torch.OutOfMemoryError:
                rec["oom"] = True
            torch.cuda.empty_cache()
            results["runs"].append(rec)
            if rec.get("oom"):
                print(f"{name:12s} P={P:7d}  OOM", flush=True)
            else:
                print(f"{name:12s} P={P:7d}  {rec['ms']:7.3f} ms/token (graph {rec.get('ms_graph', float('nan')):7.3f})"
                      f"   state {rec['state_mb']:10.3f} MB", flush=True)
        del model
    save(results, args.out)


if __name__ == "__main__":
    main()
