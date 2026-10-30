# Literature notes for the length-generalization paper

Verification: bibliographic data were taken from the arXiv API (export.arxiv.org) or the official proceedings page on
2026-10-29. Papers marked **full text** were read in their arXiv HTML version for the points listed; the others were
checked at the level of the abstract and metadata. No quotation in the paper comes from these notes.

## Closest work (full text)

**Stuffed Mamba: Oversized States Lead to the Inability to Forget.** Yingfa Chen, Xinrong Zhang, Shengding Hu, Xu Han,
Zhiyuan Liu, Maosong Sun. COLM 2025. arXiv:2410.07145 (v4, Jan 2026).
- Contribution: Mamba-2 models fail beyond the training length because they do not learn to forget when the state is
  large relative to the training length ("state overparameterization").
- Experiments: official Mamba-2 checkpoints 130M–1.3B (trained at 8K) and models of 36M–85M trained from scratch on
  RedPajama-V2; passkey retrieval up to 256K; memory-retention strength (cumulative decay of the first token);
  scaling laws for the training length needed to learn forgetting (linear in state size).
- Relevant limitation: no models below 36M parameters; tokenized text; no analysis of position-dependent training
  regularities.
- Difference: we train 1M-parameter byte models from scratch and do not observe failure with plain-text training even
  for N = 64 at L = 256; our failures arise only from a passkey format that leaks position.

**DeciMamba: Exploring the Length Extrapolation Potential of Mamba.** Assaf Ben-Kish, Itamar Zimerman, Shady
Abu-Hussein, Nadav Cohen, Amir Globerson, Lior Wolf, Raja Giryes. ICLR 2025. arXiv:2406.14528.
- Contribution: an effective receptive field (Mamba mean distance from implicit attention) limited by the training
  length; DeciMamba pools tokens by Δ to extend context without training.
- Experiments: pretrained Mamba 130M–2.8B; 80M models trained on WikiText-103 at several lengths; passkey; PG-19
  perplexity with sliding windows scoring the last 100 tokens per window (different tokens at each context length).
- Difference: we measure context use on identical bytes and find no use beyond tens of bytes even within L.

**Mamba Modulation: On the Length Generalization of Mamba.** Peng Lu, Jerry Huang, Qiuhao Zeng, Xinyu Wang, Boxing
Chen, Philippe Langlais, Yufei Cui. NeurIPS 2025. arXiv:2509.19633.
- Contribution: state growth beyond the training length depends on the spectrum of A (eigenvalues near 1 explode);
  per-layer spectrum scaling calibrated on ~20 samples.
- Experiments: Mamba 1.4B/2.8B, Mamba-2 1.3B/2.7B; PG19, ProofPile, GovReport up to 128K; perplexity explodes for base
  models (e.g. Mamba-2 1.3B: 9.5 at 2K → 1479 at 64K on PG19).
- Difference: our small models show neither explosion nor growth beyond 1.2–1.8× after plain training.

**Understanding and Improving Length Generalization in Recurrent Models.** Ricardo Buitrago Ruiz, Albert Gu. arXiv
preprint 2507.02782 (2025).
- Contribution: "unexplored states hypothesis"; state passing / TBTT / fitted noise fix length generalization with
  ~100–500 post-training steps; "effective remembrance" metric (distance between output distributions with and
  without a prefix).
- Notes: explicitly observes that a model using only the last T tokens generalizes trivially, which is our regime.
- Difference: we observe this trivial regime directly at small scale and show a different failure mode (a learned
  position clock) when training data leak position.

**Probing Length Generalization in Mamba via Image Reconstruction.** Jan Rathjens, Robin Schiewer, Laurenz Wiskott,
Anand Subramoney. arXiv preprint 2603.12499 (2026).
- Contribution: Mamba adapts its processing strategy to the training-length distribution in an image-reconstruction
  task; could not reproduce a state-size effect; a length-prepended variant.
- Difference: language modeling on bytes; we identify the specific learned mechanism (predicting the marker from
  position).

## Context (metadata / abstract level)

- Mamba: Albert Gu, Tri Dao. arXiv:2312.00752 (2023). Venue not verified here; cited as preprint.
- Transformers are SSMs (Mamba-2): Tri Dao, Albert Gu. ICML 2024. arXiv:2405.21060.
- S4: Albert Gu, Karan Goel, Christopher Ré. ICLR 2022. arXiv:2111.00396.
- S4D: Albert Gu, Ankit Gupta, Karan Goel, Christopher Ré. arXiv:2206.11893 (cited as preprint).
- LongMamba: Zhifan Ye, Kejing Xia, Yonggan Fu, Xin Dong, Jihoon Hong, Xiangchi Yuan, Shizhe Diao, Jan Kautz, Pavlo
  Molchanov, Yingyan Celine Lin. ICLR 2025. arXiv:2504.16053. Local vs. global channels by receptive field.
- MambaExtend: Seyedarmin Azizi, Souvik Kundu, Mohammad Erfan Sadeghi, Massoud Pedram. ICLR 2025 (proceedings).
  Calibrates Δ scaling factors without training.
- MambaByte: Junxiong Wang, Tushaar Gangavarapu, Jing Nathan Yan, Alexander M. Rush. COLM 2024. arXiv:2401.13660.
- Zoology: Simran Arora et al. ICLR 2024. arXiv:2312.04927. Associative recall explains most of the gap between
  gated convolutions and attention.
- Repeat After Me: Samy Jelassi, David Brandfonbrener, Sham M. Kakade, Eran Malach. ICML 2024, PMLR 235:21502–21521.
- An Empirical Study of Mamba-based Language Models: Roger Waleffe et al. (16 authors). arXiv:2406.07887.
- ALiBi: Ofir Press, Noah A. Smith, Mike Lewis. ICLR 2022 (iclr.cc poster page). arXiv:2108.12409.
- RoFormer: Jianlin Su et al. arXiv:2104.09864 (journal version not verified; cited as preprint).
- NoPE learns position: Adi Haviv, Ori Ram, Ofir Press, Peter Izsak, Omer Levy. Findings of EMNLP 2022.
  arXiv:2203.16634.
- Positional encoding and length generalization: Amirhossein Kazemnejad et al. NeurIPS 2023. arXiv:2305.19466.
- Sharp Nearby, Fuzzy Far Away: Urvashi Khandelwal, He He, Peng Qi, Dan Jurafsky. ACL 2018. arXiv:1805.04623.
  Same-target-tokens context ablation in LSTM language models.
- Do Long-Range Language Models Actually Use Long-Range Context?: Simeng Sun, Kalpesh Krishna, Andrew
  Mattarella-Micke, Mohit Iyyer. EMNLP 2021. arXiv:2109.09115.
- Shortformer: Ofir Press, Noah A. Smith, Mike Lewis. ACL 2021. arXiv:2012.15832.
- Landmark Attention (passkey retrieval): Amirkeivan Mohtashami, Martin Jaggi. NeurIPS 2023. arXiv:2305.16300.
- Character-level LM with deeper self-attention (enwik8 split): Rami Al-Rfou et al. arXiv:1808.04444.
- Finite-precision RNNs can count: Gail Weiss, Yoav Goldberg, Eran Yahav. ACL 2018. arXiv:1805.04908.
- AdamW: Ilya Loshchilov, Frank Hutter. arXiv:1711.05101.
- enwik8: Marcus Hutter, The Human Knowledge Compression Contest, http://prize.hutter1.net.

## Novelty assessment

- Same-bytes evaluation: **established** (Khandelwal et al., Sun et al.); we use it, we do not claim it.
- Short learned memory as the reason for trivial stability: **anticipated** in Buitrago Ruiz & Gu (2025) as a
  possibility; our contribution is direct measurement in a controlled small-scale sweep.
- Training-length-dependent strategies in Mamba: **related** finding in Rathjens et al. (2026) for images.
- Passkey formats leaking position and producing apparent length-generalization failure in SSMs: we found no prior
  report of this specific mechanism; NoPE Transformers learning position (Haviv et al.) and RNNs counting (Weiss et
  al.) are the closest established facts. Treat as a modest, exploratory contribution.
