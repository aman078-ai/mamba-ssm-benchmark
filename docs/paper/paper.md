---
title: "Stable but Short-Sighted: Length Generalization of Small Selective State-Space Models on Byte-Level Text"
author: "Aman Shaik"
status: "DRAFT"
---

# Abstract

Recurrent sequence models such as Mamba are often expected to handle inputs longer than those seen in training, yet
published evidence disagrees: some work reports that Mamba's effective receptive field is limited by the training
length, other work reports "state collapse" beyond it. We study the question at a scale where it can be controlled
precisely: ~1M-parameter models trained on 24.6M bytes of enwik8, with Mamba variants spanning state sizes 4–64,
per-state (Mamba-1) and shared (Mamba-2-style) decay, and a non-selective ablation, against Transformers with RoPE,
NoPE and ALiBi. We first show that two common ways of measuring length generalization are confounded, by the number of
cold-start bytes per window and by which text sits at which position, and replace them with a *same-bytes* measure: the
loss on identical bytes with full versus training-length context. With three seeds per configuration, every Mamba
variant is stable up to 32× its training length but gains nothing from the extra context (|effect| ≤ 0.003 bits per
byte), RoPE and NoPE Transformers collapse (+2.8 to +3.1 bits per byte), and only ALiBi benefits (−0.04). Mechanistic
measurements explain the Mamba result: learned memory horizons are short (median 1–100 bytes per state index), do not
grow with training length, and shrink as the state grows, so the state never fills. A passkey task meant to force long
memory instead exposes a different failure: every Mamba variant learns to predict, from position alone, where the
passkey's RECALL marker ends each training window, and beyond the training length this learned clock fires on
ordinary text (+5 to +41 bits per byte). Randomizing the marker's position halves the effect but does not remove it.
No Mamba variant learned the recall itself in 49M training bytes; an ALiBi Transformer did.
All code, configurations, raw results and an audit trail of corrected claims are released.

# 1. Introduction

Linear-time recurrent models compress the past into a fixed-size state, so in principle they can read arbitrarily long
inputs. Whether they *generalize* to lengths beyond training is a separate question with conflicting answers:
DeciMamba reports that Mamba's receptive field is restricted by the training length; Stuffed Mamba reports that models
whose state is large relative to the training length "collapse" beyond it; Mamba Modulation describes the question as
open. Most of this evidence comes from pretrained models at hundreds of millions of parameters, where training data,
length and tuning cannot be varied freely.

We ask the question in a regime small enough to control every variable on a single 4 GB laptop GPU:

- **RQ1.** Do small selective SSMs trained on byte-level text degrade, stay stable, or improve beyond their training
  length, and how does that depend on state size, decay parameterization, selectivity and training length?
- **RQ2.** What mechanism explains the behavior?
- **RQ3.** Does the picture change when training forces the model to hold information over long gaps?

Contributions: (i) a demonstration that two common length-generalization measures are confounded, and a same-bytes
measure that is not; (ii) a controlled, multi-seed comparison of 9 Mamba configurations and 3 Transformer position
encodings at two training lengths; (iii) mechanistic measurements (memory horizons per state index, state-norm growth)
that account for the results; (iv) evidence that selective SSMs learn an absolute-position clock from regularities
tied to the training window, which produces collapse beyond it; (v) a fully reproducible release, including an
audit of claims an earlier version of this project made and later withdrew.

# 2. Related work

**Selective SSMs.** Mamba (Gu & Dao, 2023) makes the step size Δ and projections B, C input-dependent; Mamba-2
(Dao & Gu, 2024) restricts the decay to a scalar per head. MambaByte (Wang et al., 2024) applies Mamba to bytes.
**Length generalization of SSMs.** DeciMamba (Ben-Kish et al., 2025); Stuffed Mamba (Chen et al., 2024); Mamba
Modulation (2025). **Position encodings and extrapolation.** RoPE (Su et al., 2021) is known not to extrapolate
without extension methods; ALiBi (Press et al., 2022) was designed to; NoPE results vary by setting.
**Recall and hybrids.** Zoology (Arora et al., 2023), Jelassi et al. (2024), Waleffe et al. (2024).

# 3. Setup

**Data.** enwik8, standard 90M/5M/5M split; all analysis on the validation split. Bits per byte (bpc).
**Models.** Byte-level LMs, d_model 128. Mamba: 8 layers (expand 2, conv 4, d_state N ∈ {4, 16, 64}); decay
parameterization per state (S4D-real init, A_n = −(n+1)) or scalar per channel (init U[1, 16]); non-selective (LTI)
ablation with Δ, B, C learned constants. Transformer: 5 layers, 4 heads, RoPE / NoPE / ALiBi. 0.83–1.23M non-embedding
parameters (Table 1).
**Training.** AdamW (β = 0.9, 0.95, wd 0.1), 3,000 steps × 8,192 bytes (24.6M bytes), warmup 150, cosine to 1e-4,
bf16 autocast, clipping 1.0; training length L ∈ {256, 1024} with the batch adjusted to keep bytes per step fixed.
**Tuning.** Learning rate swept over {1, 2, 4, 8, 16, 32}×1e-3 per family at L = 512 (1,500 steps): Mamba 1.6e-2,
Transformer 8e-3. Seed standard deviation of validation bpc ≈ 0.005.
**Seeds.** Three per configuration (0, 1, 2).

# 4. Measuring length generalization without confounds

*Window-level bpc* (evaluate non-overlapping windows of length L' > L) falls with L' even for a model that uses no
extra context, because longer windows contain fewer cold-start bytes. *Position bands* inside one long window compare
different text at different positions. On the same trained model, the second measure reported a 0.199 bits-per-byte
"improvement" beyond the training length that vanished (−0.001 to +0.000) under the same-bytes measure.

**Same-bytes context effect.** For bytes at positions in band [aL, bL) of a W = 16K window, we compute the loss with
the full context from the window start and the loss with L/2–L bytes of context (sliding windows of length L, stride
L/2), and report their mean difference in bits per byte. Negative values mean the extra context helps; positive values
mean it hurts.

# 5. Results

**5.1 Quality at the training length.** [Table from results/research/aggregate.md: bpc at L, mean ± 95% CI.]
Selective Mamba variants are indistinguishable from one another (1.836–1.847 at L = 256) and better than the
non-selective ablation (1.874) and every Transformer (ALiBi 1.906, RoPE 1.930, NoPE 2.325).

**5.2 Beyond the training length.** [Figure: plots/research/final_context_effect.png.] Every selective and
non-selective Mamba configuration has a context effect within ±0.003 bits per byte at every band up to 32× L, at both
training lengths. RoPE and NoPE Transformers collapse (+2.8 and +3.1 at 8–16× for L = 256); ALiBi is the only model
that improves (−0.041 ± 0.021).

**5.3 Mechanism.** Memory horizons h = 1/(|A_n| · mean Δ) are short: the median over channels is ~50–100 bytes for the
slowest state index and 1–25 bytes for the fastest (S4D decay), ~8–9 bytes for scalar decay. Horizons do not scale
with training length (99th percentile, N = 16: 784 bytes at L = 256 vs. 679 at L = 1024) and shrink with N. The SSM
state RMS grows at most 1.5× from 1K to 16K bytes. With memory this short, the state never fills, which is why neither
collapse nor long-range use appears.

**5.4 Forcing long memory (passkey).** We inserted an 8-letter random key into every 512-byte training window:
a STORE marker and the key, then 128–494 bytes of text, then a RECALL marker and the same key ending the window
(6,000 steps × 16 × 512 bytes, seed 0, the same models as above). The recalled bytes can only be predicted from
memory, so a model that lowers the training loss on them must hold information across the gap.
[Figure: plots/research/passkey.png.]

*Recall.* Only the ALiBi Transformer learned the task: exact recall of 100% of keys at gap 256, 99% at 512 and 14%
at 1,024 (57% of bytes). All five Mamba variants and the RoPE Transformer stayed at chance (≈4.70 bits per key byte
= log2 26) at every gap, inside and beyond the training length. RQ3 is therefore not answered at this scale: the
models were never forced into long memory because they never learned to use it.

*Length generalization.* The same models, scored on plain held-out text, collapse beyond the training length (context
effect at 8–16×, bits per byte): S4D decay N = 16 +32.1, N = 64 +30.7; scalar decay N = 16 +24.1, N = 64 +41.4; LTI
+4.9; ALiBi +6.3; RoPE +2.9 (about the same as plain training). Section 5.2's Mamba models, trained identically except
for the passkey, showed |effect| ≤ 0.003.

*Mechanism: a learned clock.* The RECALL marker (byte 2, which never occurs in enwik8) always sat at position 503.
On plain text, every passkey-trained Mamba assigns it probability above 0.5 from position 503–508 (S4D decay) or
406–485 (scalar decay, LTI), and a mean probability of 0.73–1.00 beyond position 512. The loss of a byte predicted as RECALL with
near-certainty is ~30 bits, which accounts for the collapse. The models count position in their state: the SSM state
RMS grows 9–16× from 1K to 16K bytes (≤1.5× for plain-text training), and in the S4D N = 16 model one layer-6 mode has
a memory horizon of ~4×10⁵ bytes, i.e. it does not forget. Two controls rule out simpler explanations. Adding the STORE
marker that every training window contained does not change the collapse (512–1,024 bytes: 29.3 vs. 29.4 bits per
byte for S4D N = 16). Placing the RECALL block at a random position in each window (stage 5c) lowers the S4D N = 16
effect from +32.1 to +15.2 at 8–16× and moves the onset from position 508 to 753 (scalar decay N = 64: +41.4 →
+17.9, onset 460 → 569, and validation bpc at the training length recovers from 2.26 to 1.86), but does not remove
it, because every window still contains exactly one RECALL, so the probability of seeing it rises to 1 by the window's end.
Recall stayed at chance.

The scalar-decay models also degrade *inside* the training length on plain text (validation bpc 3.72 and 2.26 vs. 1.79
for S4D decay), because they predict the marker from position ~406–460 onward; their training loss, measured on
windows that contain the marker, was normal (1.85–1.93).

The clock is not specific to recurrence. The ALiBi Transformer, which has no absolute position signal but can count
through causal attention, also predicts the marker from position 485 (mean probability 0.84 beyond 512), which accounts
for its +6.3 collapse after passkey training versus −0.04 after plain training. The RoPE Transformer crosses 0.5 at 503
but only briefly (mean 0.16 beyond 512); its collapse matches plain training.

# 6. Discussion

The small-scale picture separates *stability* from *use*: stability beyond the training length is cheap for selective
SSMs here, but it comes from forgetting quickly, not from exploiting long context. This is consistent with DeciMamba's
observation of a limited receptive field but not with its explanation via training length, and it does not reproduce
state collapse, which Stuffed Mamba attributes to states that are oversized for the training length.

The passkey experiments show what *does* produce collapse here: a training signal that depends on absolute position
within the window. Given one, every Mamba variant (including the non-selective one, and including N = 64, which
Stuffed Mamba would expect to be most prone) builds a non-forgetting mode that counts bytes (ALiBi attention learns the same clock), and the count leaves the
range seen in training once the input is longer than the window. This is the same symptom as state collapse, a state
norm that keeps growing and a loss that explodes after the training length, from a different cause than state
capacity. We do not claim it explains collapse in pretrained models, but it suggests a check: whether their training
data carries window-length regularities (fixed-length packing, document boundaries, end-of-sequence tokens) that a
recurrent model can learn to count. For benchmark design, tasks that put an event at a fixed or bounded position in
every training window teach recurrent models position, and the length generalization measured afterwards reflects that
lesson rather than the architecture. Learning the recall itself needed attention in our budget, consistent with
Zoology and Jelassi et al.

# 7. Limitations and threats to validity

Scale (~1M parameters, 24.6M bytes) and modality (bytes of Wikipedia markup) differ from the settings of the papers
we compare with. Learning rates were tuned at L = 512 and transferred. Memory horizons use a stationary approximation
(mean Δ). The Mamba scan kernel uses atomic adds, so runs are not bitwise reproducible (deterministic mode available).
One GPU, Windows, PyTorch 2.7.1. The passkey experiments use one seed and 6,000 steps; Mamba might learn recall
with a longer budget (an earlier associative-recall check needed 3× the steps), which would let RQ3 be answered. Our
randomized variant still has a rising hazard toward the window end; a design with no position information at all
(e.g. a marker that appears in only some windows, at a geometrically distributed position) was not run. ALiBi passkey
recall beyond gap 1,024 was not measured (out of memory under the 2.97 GiB cap).

# 8. Reproducibility

`configs/stage3_*.json`, `stage4_pilot.json`, `stage4b_scalar_decay.json`, `stage5a_seeds.json`,
`stage5b_passkey.json`, `stage5c_passkey_random.json`; run with `python -m experiments.run <config>`. Raw results under `results/research/`, CSV under
`results/csv/`, environment recorded per run. Audit trail: `docs/AUDIT.md`.

# References

Arora et al. 2023, Zoology. Ben-Kish et al. 2025, DeciMamba. Chen et al. 2024, Stuffed Mamba. Dao & Gu 2024,
Transformers are SSMs. Gu & Dao 2023, Mamba. Jelassi et al. 2024, Repeat After Me. Press et al. 2022, ALiBi.
Su et al. 2021, RoFormer. Waleffe et al. 2024, An Empirical Study of Mamba-based Language Models. Wang et al. 2024,
MambaByte. Mamba Modulation 2025 (arXiv 2509.19633).
