#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python experiments/reject_ablation_20261029/test_policy.py > policy_test.log 2>&1
for policy in learned service_first; do
  timeout 600 python -u experiments/reject_ablation_20261029/run.py --policy "$policy" --record "pilot_${policy}" --graph_sizes 50 --epochs 2 --epoch-size 64 --val-size 32 --batch-size 16 --pomo-size 1 --seed 1234 --num-workers 0 --reject-warmup-epochs 0 --baseline-mode batch_mean --output-root "pilot_${policy}/outputs" --benchmark-mode --benchmark-warmup-epochs 0 --benchmark-json --benchmark-json-output "pilot_${policy}/timing.json" > "pilot_${policy}.log" 2>&1
done
