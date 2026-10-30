| Protocol (seeds) | Model | Keys exact (%), gap 64 / 256 / 512 / 1024 | bpc at $L$ | CE 8–16× | P(RECALL) $> 0.5$ from | Mean P(RECALL) beyond $L$ | State RMS 16K/1K |
|---|---|---|---|---|---|---|---|
| fixed end (1) | Mamba S4D, N=16 | – / 0 / 0 / 0 | 1.79 | +32.062 | 508 | 0.9984 | 13.6 |
| fixed end (1) | Mamba S4D, N=64 | – / 0 / 0 / 0 | 1.82 | +30.665 | 503 | 0.9991 | 10.6 |
| fixed end (1) | Mamba scalar, N=16 | – / 0 / 0 / 0 | 3.72 | +24.072 | 406 | 0.7303 | 13.7 |
| fixed end (1) | Mamba scalar, N=64 | – / 0 / 0 / 0 | 2.26 | +41.446 | 460 | 0.9986 | 15.7 |
| fixed end (1) | Mamba LTI, N=16 | – / 0 / 0 / 0 | 2.56 | +4.899 | 485 | 0.9220 | 9.4 |
| fixed end (1) | RoPE | – / 0 / 0 / 0 | 1.87 | +2.897 | 503 | 0.1574 | – |
| fixed end (1) | ALiBi | – / 100 / 99 / 14 | 1.82 | +6.265 | 485 | 0.8399 | – |
| random end (1) | Mamba S4D, N=16 | – / 0 / 0 / 0 | 1.81 | +15.192 | 753 | 0.7081 | 2.3 |
| random end (1) | Mamba scalar, N=64 | – / 0 / 0 / 0 | 1.86 | +17.879 | 569 | 0.8911 | 6.4 |
| position-free (3) | Mamba S4D, N=16 | 0 ± 0 / 0 ± 0 / 0 ± 0 / 0 ± 0 | 1.72 ± 0.01 | −0.002 ± 0.001 | never | 0.0006 ± 0.0000 | 1.3 ± 0.2 |
| position-free (3) | ALiBi | 99 ± 1 / 99 ± 2 / 98 ± 1 / 47 ± 32 | 1.69 ± 0.00 | −0.025 ± 0.004 | never | 0.0005 ± 0.0001 | – |
