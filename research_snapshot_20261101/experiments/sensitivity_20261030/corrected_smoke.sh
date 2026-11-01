#!/usr/bin/env bash
set -euo pipefail
for policy in learned service_first; do
 root="smoke_${policy}"
 python -u experiments/sensitivity_20261030/corrected_entry.py --stage train --policy "$policy" --record "$root" --graph_sizes 50 --epochs 2 --epoch-size 64 --val-size 32 --batch-size 16 --pomo-size 1 --seed 1234 --num-workers 0 --reject-warmup-epochs 0 --baseline-mode batch_mean --output-root "$root/outputs" > "$root.log" 2>&1
 python -u experiments/sensitivity_20261030/corrected_entry.py --stage eval --audit-output "$root/audit.json" --checkpoint "$root/outputs/pomo_n50_optimized/model_final.pt" --graph-size 50 --num-samples 16 --batch-size 16 --seed 928632 --max-concurrent-open-orders 6 --enable-delivery-viability --json-output "$root/eval.json" > "${root}_eval.log" 2>&1
done
