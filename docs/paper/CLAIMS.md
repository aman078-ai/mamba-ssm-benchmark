# Claim-to-evidence audit for `docs/paper/paper.md`

Checked on 2026-10-29 against the code and the stored results (not the README). "Verified" means re-derived from the
raw result files or the code during this audit. Status: **safe** = may be stated as written; **qualified** = may be
stated with the qualification given; **removed** = taken out of the manuscript.

## Method and protocol

| # | Claim | Evidence | Source | Verified | Status / qualification |
|---|---|---|---|---|---|
| M1 | Same-bytes context effect compares identical bytes under full vs. L/2..L bytes of context | `context_effect()`: sliding windows of length L, stride L/2, second half kept; full-context loss on the same positions | `analysis/length_generalization.py` | yes (code read) | safe. Short context is *between L/2 and L*, not exactly L; the paper says so |
| M2 | The method is not new | Khandelwal et al. 2018; Sun et al. 2021; "effective remembrance" in Buitrago Ruiz & Gu 2025 | literature | yes | safe; the paper claims no novelty for it |
| M3 | Validation split for all analysis; test split untouched | `bpc`, `context_effect`, `position_loss`, horizons all use `"valid"`; test used only by `analysis/test_split_check.py` | code | yes | safe. LR selection also used validation (minor selection effect, disclosed) |
| M4 | Split 90M / 5M / 5M | `N_TRAIN, N_VALID, N_TEST` | `data/enwik8.py` | yes | safe |
| M5 | Marker bytes 1 and 2 never occur in enwik8 | counted 0 in train, valid and test | ad-hoc check, 2026-10-29 | yes | safe |
| M6 | Budget 24.6M bytes per main run | 3000 × 32 × 256 = 3000 × 8 × 1024 = 24,576,000 for all 48 runs | `config.json` of every run | yes | safe |
| M7 | 48 runs, seeds 0/1/2 | 9 configs at L=256 + 7 at L=1024, 3 seeds each, all present | `results/research/{stage4_pilot,stage4b_scalar_decay,stage5a_seeds}` | yes | safe. Two seed-1/2 runs crashed at step 2,400 and were re-run from scratch (same config) |
| M8 | Parameter counts 0.83–1.23M (non-embedding) | `non_embedding_params` in config | `config.json` | yes | safe; not parameter-matched (disclosed) |
| M9 | LR 1.6e-2 (Mamba), 8e-3 (Transformer) from a sweep | sweep table; Mamba 0.008 and 0.016 within 0.004 bpc | `results/research/stage3_lr_sweep` | yes | qualified: swept only for Mamba N=16 S4D and RoPE; transferred to other variants |
| M10 | 95% CI from t with df = 2 | `T95[2] = 4.303`, `ci95 = t * sd / sqrt(n)` | `analysis/aggregate_seeds.py` | yes | safe |
| M11 | Eq. (1) matches the code (exact ZOH, softplus Δ, A = −exp(a), D skip, SiLU gate) | `discretize(method="zoh")`, `Mamba.ssm_params`, `Mamba.forward` | `models/scan.py`, `models/mamba.py` | yes | safe |
| M12 | Horizon τ = 1/(|A| mean Δ) is the average e-folding time | product of exp(Δ_u A) = exp(A ΣΔ_u); mean Δ over the first 16,384 validation bytes | `mamba_stats()` | yes | qualified: mean-field; ignores B/C and input-dependent bursts (stated) |
| M13 | State RMS by chunked prefill with exact state passing | `mamba_stats()`, chunk 1024 | code; state-passing equivalence tested in `tests/test_models.py` | yes | safe |

## Results (main study, 3 seeds)

| # | Claim | Evidence | Verified | Status / qualification |
|---|---|---|---|---|
| R1 | All Mamba seed means within 0.004 bpc of zero in every band | max |mean| = 0.0040 (LTI, 16–32×, L=256); generated Table 1 | yes | safe. The draft said 0.003, which was wrong; corrected |
| R2 | Small consistent gain ~0.003 bpc in the 1–2× band at L=256 | −0.002 to −0.004 for all six variants; CI excludes 0 only for some | yes | qualified: "about 0.003", not individually significant for every variant |
| R3 | RoPE +2.8, NoPE +3.1 at 8–16× (L=256) | +2.804 ± 0.029, +3.120 ± 0.948 | yes | safe; NoPE CI is wide |
| R4 | ALiBi −0.041 ± 0.021 (L=256), −0.009 ± 0.004 (L=1024) | Table 1 | yes | safe |
| R5 | Abstract: Transformers "degrade by up to 3.4 bits per byte" | max seed mean: RoPE L=1024 8–16× +3.359 | yes | safe |
| R6 | Selective Mamba 1.836–1.847 bpc at L=256; LTI 1.874; Transformers 1.906/1.930/2.325 | Table 1 | yes | safe |
| R7 | NoPE undertrained at L=1024 (3.14 bpc) | Table 1 | yes | safe |
| R8 | Median horizon 2–38 bytes; slowest index 40–76; fastest 1–24 (S4D) | Table 2 / ad-hoc recomputation | yes | safe |
| R9 | Horizons do not follow training length | N=16 slowest-index 52.5 vs 56.3; p99 806 vs 685 | yes | safe (for 256 vs 1024 only) |
| R10 | 0.1–19% of modes exceed L; slowest mode 3e3–8e6 bytes | Table 2 | yes | safe; **replaces** the earlier claim that "the state never fills" |
| R11 | State RMS 16K/1K is 1.2–1.8 (seed means, max over layers) | Table 2 | yes | safe. Earlier claim "≤ 1.5×" was wrong (individual seeds up to 2.45); corrected |
| R12 | "Window-level smoke test showed 0.199 bpc spurious gain" | number appears only in CHANGELOG; no stored result | — | **removed** from the paper |
| R13 | Mamba MQAR: 52.9% after 9,000 steps vs < 8% after 3,000 | `results/audit/mqar_budget.json` | yes | safe (used only as context in Discussion) |

## Passkey (exploratory)

| # | Claim | Evidence | Verified | Status / qualification |
|---|---|---|---|---|
| P1 | Fixed end: RECALL marker at position 503 | `insert_passkey`: r0 = T+1 − 9 = 504 in the buffer, i.e. target index 503 | yes | safe |
| P2 | Fixed end: only ALiBi learned recall (100/99/14% at 256/512/1024) | `passkey.json` | yes | safe; one seed |
| P3 | Fixed end: Mamba failure +4.9 to +41.5 bpc at 8–16× on plain text | `length_generalization.json` | yes | safe; one seed per model |
| P4 | Mechanism: models predict RECALL from position (onset 406–508; mean P 0.73–1.00 beyond 512) | `passkey_text_control.json` | yes | safe. Direct behavioral evidence |
| P5 | STORE-presence control does not change the loss | 29.3 vs 29.4 bpc at 512–1024 (N=16) | yes | safe |
| P6 | Same clock in ALiBi (onset 485, mean 0.84) | `passkey_text_control.json` (2 sequences, gap 4096) | yes | qualified: small evaluation sample for Transformers (memory) |
| P7 | Random end halves the failure (+15.2, +17.9) and moves onset to 569–753 | stage 5c files | yes | safe; one seed each |
| P8 | "One layer-6 mode with a 4e5-byte horizon shows non-forgetting" (earlier draft) | plain-trained models also have modes of 1e4–8e6 bytes | — | **removed**: not distinctive |
| P9 | "State RMS growth 9–16× is the signature of the failure" (earlier draft) | 5c N=16 fails by 15 bpc with 2.3× growth | — | **qualified**: growth is a symptom of the counter, not a necessary signature |
| P10 | Position-free protocol: hazard ≤ 0.0013 per byte | simulation with 20,000 windows (gap 16–448, p = 0.5) | yes | safe |
| P11 | Position-free (5d): Mamba recall at chance at every gap (64–16,384), 3 seeds; training recall accuracy ≈ 4% throughout | `passkey.json`, `log.jsonl` (`train_passkey_acc`) | yes | safe |
| P12 | Position-free: no clock (P(RECALL) ≈ 0.0005 at every position = training base rate 0.28/512), Mamba context effect within 0.003 of zero | `passkey_text_control.json`, `length_generalization.json` | yes | safe |
| P13 | Position-free: ALiBi recall 98–100% at gaps 64–512, 33–58% at 1,024, 0% from 2,048; context effect −0.020 to −0.026 | ALiBi files (re-evaluated by `experiments/post_stage5d.sh`; seed 0 re-run after a one-off CUDA error while tests ran concurrently) | yes | safe |
| P14 | "The 5b/5c failure was caused by the training format" | 5d differs from 5b/5c only in passkey placement/frequency (also 10,000 vs. 6,000 steps and a wider gap range) | yes | qualified: 5d also differs in budget and gap range; the paper says "differ only in where and how often the passkey occurs" — **adjusted** to mention the other differences |

## Test split

| # | Claim | Evidence | Verified | Status |
|---|---|---|---|---|
| T1 | Selective Mamba context effect within 0.004 of zero on the test split (all bands); LTI within 0.007 | `results/research/test_split.json` (8 windows) | yes | safe |
| T2 | RoPE/NoPE degrade (+3.2/+3.3 at 8–16×, L=256); ALiBi −0.047 ± 0.023 / −0.009 ± 0.002 | same | yes | safe |
| T3 | Test bpc 0.02–0.06 lower than validation for every model | same vs. Table 1 | yes | safe |

## Literature claims

Every citation was checked against arXiv metadata or the official proceedings page; see `docs/paper/LITERATURE.md`.
The draft's characterization of DeciMamba ("receptive field limited by training length"), Stuffed Mamba (inability to
forget when the state is large relative to the training length) and Mamba Modulation (spectrum of A) was checked
against the full text (arXiv HTML). Earlier wording "Stuffed Mamba reports state collapse" was adjusted: the
current version of that paper speaks of "inability to forget" and over-retention.
