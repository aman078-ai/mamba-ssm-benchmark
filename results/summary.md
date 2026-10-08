### Models

| model | layers | non-embedding params | training throughput (tokens/s) | training memory |
|---|---|---|---|---|
| Transformer | `AAAAAA` | 4.72M | 64.9k | 1.22 GB |
| Mamba | `MMMMMMMMMMM` | 4.82M | 42.8k | 1.81 GB |
| RetNet | `RRRRRR` | 4.72M | 71.2k | 0.91 GB |
| Hybrid | `MMMMMAMMMM` | 4.73M | 46.0k | 1.71 GB |

### enwik8

| model | val bpc | **test bpc** (ctx 512) | test @ 1K | test @ 2K | test @ 4K | test @ 8K |
|---|---|---|---|---|---|---|
| Transformer | 1.696 | **1.651** | 2.248 | 3.213 | 3.905 | OOM |
| Mamba | 1.685 | **1.673** | 1.641 | 1.634 | 1.631 | 1.629 |
| RetNet | 1.822 | **1.802** | 1.775 | 1.769 | 1.765 | 1.763 |
| Hybrid | 1.646 | **1.608** | 1.548 | 1.541 | 1.554 | OOM |

### Speed

**Inference forward pass, batch 1 (ms)**

| model | 1K | 4K | 16K | 64K | 128K | longest length that fits |
|---|---|---|---|---|---|---|
| Transformer | 17.0 | 104.3 | OOM | OOM | OOM | 4K |
| Transformer (PyTorch fused attn.) | 14.8 | 19.0 | 126.1 | 1519.1 | 5763.1 | 128K |
| Mamba | 24.2 | 44.5 | 187.4 | 848.9 | 1800.3 | 128K |
| Mamba (pure-PyTorch scan) | 73.4 | 235.1 | 1373.0 | OOM | OOM | 16K |
| RetNet | 30.9 | 31.2 | 76.6 | 305.7 | 648.1 | 128K |
| Hybrid | 22.9 | 54.2 | OOM | OOM | OOM | 4K |

**Training step fwd+bwd, batch 1 (ms)**

| model | 1K | 4K | 16K | 64K | 128K | longest length that fits |
|---|---|---|---|---|---|---|
| Transformer | 46.1 | OOM | OOM | OOM | OOM | 2K |
| Transformer (PyTorch fused attn.) | 40.7 | 64.4 | 545.2 | OOM | OOM | 32K |
| Mamba | 63.8 | 170.1 | OOM | OOM | OOM | 8K |
| Mamba (pure-PyTorch scan) | 301.9 | 1104.3 | OOM | OOM | OOM | 8K |
| RetNet | 72.1 | 72.3 | 220.9 | OOM | OOM | 16K |
| Hybrid | 60.0 | 185.5 | OOM | OOM | OOM | 4K |

- crossover (fwd): Mamba faster than Transformer from L = 2K
- crossover (train): Mamba faster than Transformer from L = 4K

### Ablation

| Mamba variant | val bpc @ step 500 | val bpc @ step 1000 | Δ vs default |
|---|---|---|---|
| selective Δ,B,C · exact ZOH (default) | 2.172 | 1.998 | — |
| **no selection** (LTI: Δ,B,C constant) | 2.226 | 2.045 | +0.047 |
| Euler B̄ = ΔB (Mamba's simplification) | 2.171 | 1.998 | +0.000 |
| bilinear / Tustin | 2.170 | 1.995 | -0.003 |

### Decode

GPU time per generated token (CUDA graph replay) · cache/state size

| model | 128 ctx | 512 ctx | 2K ctx | 8K ctx | 32K ctx | 128K ctx | 256K ctx |
|---|---|---|---|---|---|---|---|
| Transformer | 0.80 ms · 0.75 MB | 0.71 ms · 3 MB | 0.77 ms · 12 MB | 1.02 ms · 48 MB | 1.95 ms · 192 MB | 6.05 ms · 768 MB | 11.46 ms · 1536 MB |
| Mamba | 0.93 ms · 0.376 MB | 0.93 ms · 0.376 MB | 0.93 ms · 0.376 MB | 0.93 ms · 0.376 MB | 0.93 ms · 0.376 MB | 0.93 ms · 0.376 MB | 0.94 ms · 0.376 MB |
| RetNet | 0.86 ms · 0.375 MB | 0.85 ms · 0.375 MB | 0.85 ms · 0.375 MB | 0.85 ms · 0.375 MB | 0.85 ms · 0.375 MB | 0.86 ms · 0.375 MB | 0.87 ms · 0.375 MB |
| Hybrid | 0.88 ms · 0.4326 MB | 0.88 ms · 0.8076 MB | 0.89 ms · 2.308 MB | 0.94 ms · 8.308 MB | 1.09 ms · 32.31 MB | 1.76 ms · 128.3 MB | 2.64 ms · 256.3 MB |

### Scans

| scan | L=256 fwd / fwd+bwd (ms) | L=1K fwd / fwd+bwd (ms) | L=4K fwd / fwd+bwd (ms) | L=16K fwd / fwd+bwd (ms) | L=64K fwd / fwd+bwd (ms) | peak mem fwd+bwd @ max L |
|---|---|---|---|---|---|---|
| `reference` | 145.1 / 730.9 | 382.8 / 2515.8 | 1659.2 / 10232.2 | - | - | 672 MB @ 4K |
| `sequential` | 28.1 / 93.4 | 115.7 / 365.2 | 1010.6 / 3248.5 | - | - | 729 MB @ 4K |
| `hillis_steele` | 4.4 / 14.8 | 20.2 / 68.4 | 94.0 / 315.9 | 429.6 / 1428.7 | 1937.7 / 6385.0 | 1044 MB @ 64K |
| `blelloch` | 4.1 / 14.1 | 9.8 / 37.3 | 36.8 / 144.7 | 148.3 / 582.5 | 597.2 / 2360.9 | 1172 MB @ 64K |
| `chunked` | 4.0 / 12.9 | 6.4 / 24.2 | 19.7 / 93.0 | 119.7 / 444.2 | 912.2 / 2908.2 | 1045 MB @ 64K |
| `triton` | 0.27 / 1.3 | 0.69 / 3.3 | 2.5 / 11.0 | 8.4 / 42.0 | 32.2 / 526.5 | 904 MB @ 64K |

### Synthetic

| model (d_model 64) | 4 pairs | 8 pairs | 16 pairs | 32 pairs |
|---|---|---|---|---|
| Transformer | 52.8% | 21.5% | 11.3% | 8.8% |
| Mamba | 7.8% | 1.4% | 1.5% | 1.5% |
| RetNet | 27.9% | 14.1% | 2.4% | 2.0% |
| Hybrid | 99.0% | 24.3% | 17.6% | 6.6% |

| model | params | accuracy (trained length) | @ 256 | @ 512 | @ 1K | @ 2K | @ 4K |
|---|---|---|---|---|---|---|---|
| Mamba | 131K | **12.8%** | 8.6% | 6.4% | 6.9% | 6.6% | 6.2% |
| Mamba w/o selection (LTI) | 111K | **13.9%** | 10.0% | 7.0% | 6.4% | 6.6% | 6.8% |
| Transformer | 99K | **72.9%** | 20.7% | 16.3% | 10.3% | 9.4% | 6.8% |
| RetNet | 99K | **71.7%** | 32.1% | 17.2% | 12.5% | 10.5% | 8.0% |
| Hybrid | 115K | **100.0%** | 18.4% | 14.3% | 12.5% | 11.7% | 7.0% |

