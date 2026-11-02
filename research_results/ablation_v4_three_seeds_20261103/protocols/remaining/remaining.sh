#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export TERM=xterm
for seed in 4321 8765; do
 for policy in learned service_first; do
  root="formal_${policy}_${seed}"
  echo "===== START seed=$seed policy=$policy $(date -u -Iseconds) ====="
  python -u experiments/sensitivity_20261030/corrected_entry.py --stage train --policy "$policy" --record "$root" --graph_sizes 50 --epochs 20 --epoch-size 2048 --val-size 500 --batch-size 16 --pomo-size 1 --seed "$seed" --num-workers 0 --reject-warmup-epochs 0 --baseline-mode batch_mean --output-root "$root/outputs" 2>&1 | tee "$root.log"
  python -u experiments/sensitivity_20261030/corrected_entry.py --stage eval --audit-output "$root/audit.json" --checkpoint "$root/outputs/pomo_n50_optimized/model_best.pt" --graph-size 50 --num-samples 500 --batch-size 16 --seed 917531 --max-concurrent-open-orders 6 --enable-delivery-viability --json-output "$root/eval.json" 2>&1 | tee "${root}_eval.log"
  python -c "import json;d=json.load(open('$root/audit.json'));assert d['hard_valid']==500 and d['samples']==500 and d['unfulfilled_total']==0"
  echo "===== DONE seed=$seed policy=$policy $(date -u -Iseconds) ====="
 done
done
