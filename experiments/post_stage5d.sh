#!/usr/bin/env bash
# Follow-up evaluations after stage 5d, run one at a time with the default 2.97 GiB memory cap:
#  1. ALiBi (stage 5d): long-gap passkey recall, length generalization and the plain-text controls, which ran out of
#     memory during the stage-5d run because two training jobs shared the GPU;
#  2. the plain-text controls of stages 5b/5c again, to store P(RECALL) by position (added after those runs);
#  3. the test-split confirmation of the main study.
#     bash experiments/post_stage5d.sh
set -u
PY=.venv/Scripts/python.exe
for d in results/research/stage5d_passkey_stream_alibi/transformer_alibi_seed*; do
  echo "== $d"
  $PY -m analysis.passkey_eval --run_dir "$d" --batch_tokens 8192
  $PY -m analysis.length_generalization --run_dir "$d" --max_len 16384 --position_windows 16
  $PY -m analysis.passkey_text_control --run_dir "$d" --n 2 --gap 4096
done
for d in results/research/stage5b_passkey/mamba_* results/research/stage5c_passkey_random/mamba_*; do
  echo "== control $d"
  $PY -m analysis.passkey_text_control --run_dir "$d"
done
for d in results/research/stage5b_passkey/transformer_*; do
  echo "== control $d"
  $PY -m analysis.passkey_text_control --run_dir "$d" --n 2 --gap 4096
done
echo "== test split"
$PY -m analysis.test_split_check
