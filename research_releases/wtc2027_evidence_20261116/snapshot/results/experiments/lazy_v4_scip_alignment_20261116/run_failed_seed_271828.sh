#!/bin/zsh
set -euo pipefail

root="${0:A:h:h:h}"
exp="$root/experiments/lazy_v4_scip_alignment_20261116"
python_bin="/opt/anaconda3/bin/python3"
evaluator="$root/experiments/drl_canonical_audit_20261115/evaluate_lazy_v4_canonical.py"
checkpoint="$root/experiments/drl_canonical_audit_20261115/remote_evidence/lazy_transfer_formal_20261116/drl_lazy_v4_transfer_formal_20261116/seed_271828/outputs/pomo_n100_optimized/model_best.pt"
output="$exp/results/seed_271828_serial_retry1"

if [[ -e "$output" ]]; then
  print -u2 "Refusing to reuse existing output: $output"
  exit 2
fi

"$python_bin" "$evaluator" \
  --checkpoint "$checkpoint" \
  --graph-size 100 \
  --samples 30 \
  --seed 2026111600 \
  --output "$output" \
  --device cpu \
  --timeout 3600
