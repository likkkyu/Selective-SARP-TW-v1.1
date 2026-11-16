#!/bin/zsh
set -euo pipefail

root="${0:A:h:h:h}"
exp="$root/experiments/lazy_v4_scip_alignment_20261116"
python_bin="/opt/anaconda3/bin/python3"
evaluator="$root/experiments/drl_canonical_audit_20261115/evaluate_lazy_v4_canonical.py"

pids=()
for seed in 314159 271828 161803; do
  checkpoint="$root/experiments/drl_canonical_audit_20261115/remote_evidence/lazy_transfer_formal_20261116/drl_lazy_v4_transfer_formal_20261116/seed_${seed}/outputs/pomo_n100_optimized/model_best.pt"
  output="$exp/results/seed_${seed}"
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
    --timeout 3600 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if (( failed )); then
  print -u2 "ONE_OR_MORE_ALIGNMENT_EVALUATIONS_FAILED"
  exit 1
fi
print "ALL_ALIGNMENT_EVALUATIONS_COMPLETED"
