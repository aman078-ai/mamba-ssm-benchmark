### Models

| model | layers | non-embedding params | training throughput incl. optimizer (tokens/s) | training memory |
|---|---|---|---|---|
| Transformer | `AAAAAA` | 4.72M | 60.2k | 1.20 GB |
| Mamba | `MMMMMMMMMMM` | 4.82M | 42.4k | 1.81 GB |
| RetNet | `RRRRRR` | 4.72M | 70.7k | 0.91 GB |
| Hybrid | `MMMMMAMMMM` | 4.73M | 45.3k | 1.71 GB |

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
| Transformer | 13.1 | 128.6 | OOM | OOM | OOM | 4K |
| Transformer (PyTorch fused attn.) | 10.4 | 18.2 | 124.8 | 1505.8 | 5704.4 | 128K |
| Mamba | 15.4 | 43.6 | 186.4 | 845.3 | 1795.9 | 128K |
| Mamba (pure-PyTorch scan) | 61.2 | 231.8 | 1080.5 | OOM | OOM | 16K |
| RetNet | 21.0 | 21.4 | 76.2 | 305.0 | 645.7 | 128K |
| Hybrid | 14.6 | 57.1 | OOM | OOM | OOM | 4K |

**Training step fwd+bwd, batch 1 (ms)**

| model | 1K | 4K | 16K | 64K | 128K | longest length that fits |
|---|---|---|---|---|---|---|
| Transformer | 37.7 | OOM | OOM | OOM | OOM | 2K |
| Transformer (PyTorch fused attn.) | 32.7 | 63.4 | 540.8 | OOM | OOM | 32K |
| Mamba | 52.5 | 168.8 | OOM | OOM | OOM | 8K |
| Mamba (pure-PyTorch scan) | 283.4 | 1095.1 | OOM | OOM | OOM | 8K |
| RetNet | 67.5 | 81.3 | 219.8 | OOM | OOM | 16K |
| Hybrid | 51.9 | 190.8 | OOM | OOM | OOM | 4K |

- crossover (fwd): Mamba faster than Transformer from L = 2K
- crossover (train): Mamba faster than Transformer from L = 2K

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
| Transformer | 0.88 ms · 0.75 MB | 0.75 ms · 3 MB | 0.82 ms · 12 MB | 1.08 ms · 48 MB | 2.05 ms · 192 MB | 6.55 ms · 768 MB | 12.41 ms · 1536 MB |
| Mamba | 0.92 ms · 0.376 MB | 0.92 ms · 0.376 MB | 0.92 ms · 0.376 MB | 0.92 ms · 0.376 MB | 0.92 ms · 0.376 MB | 0.92 ms · 0.376 MB | 0.93 ms · 0.376 MB |
| RetNet | 0.85 ms · 0.375 MB | 0.85 ms · 0.375 MB | 0.85 ms · 0.375 MB | 0.85 ms · 0.375 MB | 0.84 ms · 0.375 MB | 0.85 ms · 0.375 MB | 0.85 ms · 0.375 MB |
| Hybrid | 0.88 ms · 0.4326 MB | 0.89 ms · 0.8076 MB | 0.90 ms · 2.308 MB | 0.94 ms · 8.308 MB | 1.09 ms · 32.31 MB | 1.82 ms · 128.3 MB | 2.79 ms · 256.3 MB |

### Scans

| scan | L=256 fwd / fwd+bwd (ms) | L=1K fwd / fwd+bwd (ms) | L=4K fwd / fwd+bwd (ms) | L=16K fwd / fwd+bwd (ms) | L=64K fwd / fwd+bwd (ms) | peak mem fwd+bwd @ max L |
|---|---|---|---|---|---|---|
| `reference` | 79.9 / 501.7 | 344.5 / 1857.3 | 1261.3 / 7420.0 | - | - | 672 MB @ 4K |
| `sequential` | 22.9 / 96.0 | 91.1 / 288.5 | 831.4 / 2278.5 | - | - | 729 MB @ 4K |
| `hillis_steele` | 4.9 / 14.7 | 20.0 / 68.3 | 93.9 / 315.7 | 429.4 / 1428.2 | 1937.0 / 6382.5 | 1044 MB @ 64K |
| `blelloch` | 2.9 / 14.3 | 9.4 / 36.8 | 36.5 / 144.0 | 147.2 / 580.2 | 593.4 / 2353.6 | 1172 MB @ 64K |
| `chunked` | 2.5 / 10.9 | 5.0 / 25.2 | 19.3 / 92.4 | 96.5 / 420.8 | 708.6 / 2256.2 | 1045 MB @ 64K |
| `triton` | 0.33 / 1.2 | 0.63 / 2.8 | 2.1 / 10.7 | 8.2 / 41.9 | 32.0 / 525.2 | 904 MB @ 64K |

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

