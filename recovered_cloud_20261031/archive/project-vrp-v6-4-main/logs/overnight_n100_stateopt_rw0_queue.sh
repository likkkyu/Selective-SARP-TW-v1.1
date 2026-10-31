#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/project-vrp-v6-4-main

echo "[queue] waiting for round12 training to finish..."
while pgrep -f "run_training_optimized.py.*outputs_n100_round12_stateopt_rw0_e5" >/dev/null; do
  sleep 60
done

echo "[queue] evaluating round12..."
if [ -f outputs_n100_round12_stateopt_rw0_e5/pomo_n100_optimized/model_best_service.pt ]; then
  python3 evaluate_model.py \
    --graph-size 100 \
    --checkpoint outputs_n100_round12_stateopt_rw0_e5/pomo_n100_optimized/model_best_service.pt \
    --num-samples 500 \
    --batch-size 32 \
    --seed 99999 \
    --decode greedy \
    --max-concurrent-open-orders 6 \
    --enable-delivery-viability \
    --min-orders-per-dispatch 4 \
    --json-output outputs_cmp/n100_round12_stateopt_rw0_e5_eval500.json \
    2>&1 | tee logs/eval_n100_round12_stateopt_rw0_e5.log
fi

echo "[queue] training round13 seed4321..."
python3 run_training_optimized.py \
  --graph_sizes 100 \
  --normalization-dir /root/autodl-tmp/project-vrp-v6-4-main/outputs/normalization_profiles \
  --epochs 5 \
  --epoch-size 1024 \
  --val-size 500 \
  --batch-size 3 \
  --pomo-size 1 \
  --seed 4321 \
  --num-workers 4 \
  --max-concurrent-open-orders 6 \
  --min-orders-per-dispatch 4 \
  --enable-delivery-viability \
  --amp \
  --amp-dtype bf16 \
  --reject-warmup-epochs 0 \
  --reject-init-bias -2.5 \
  --decode-pickup-urgency-bias 0.0 \
  --alpha-reject 575 \
  --output-root outputs_n100_round13_stateopt_rw0_e5_seed4321 \
  2>&1 | tee logs/train_n100_round13_stateopt_rw0_e5_seed4321.log

echo "[queue] evaluating round13..."
python3 evaluate_model.py \
  --graph-size 100 \
  --checkpoint outputs_n100_round13_stateopt_rw0_e5_seed4321/pomo_n100_optimized/model_best_service.pt \
  --num-samples 500 \
  --batch-size 32 \
  --seed 99999 \
  --decode greedy \
  --max-concurrent-open-orders 6 \
  --enable-delivery-viability \
  --min-orders-per-dispatch 4 \
  --json-output outputs_cmp/n100_round13_stateopt_rw0_e5_seed4321_eval500.json \
  2>&1 | tee logs/eval_n100_round13_stateopt_rw0_e5_seed4321.log

echo "[queue] training round14 seed8765..."
python3 run_training_optimized.py \
  --graph_sizes 100 \
  --normalization-dir /root/autodl-tmp/project-vrp-v6-4-main/outputs/normalization_profiles \
  --epochs 5 \
  --epoch-size 1024 \
  --val-size 500 \
  --batch-size 3 \
  --pomo-size 1 \
  --seed 8765 \
  --num-workers 4 \
  --max-concurrent-open-orders 6 \
  --min-orders-per-dispatch 4 \
  --enable-delivery-viability \
  --amp \
  --amp-dtype bf16 \
  --reject-warmup-epochs 0 \
  --reject-init-bias -2.5 \
  --decode-pickup-urgency-bias 0.0 \
  --alpha-reject 575 \
  --output-root outputs_n100_round14_stateopt_rw0_e5_seed8765 \
  2>&1 | tee logs/train_n100_round14_stateopt_rw0_e5_seed8765.log

echo "[queue] evaluating round14..."
python3 evaluate_model.py \
  --graph-size 100 \
  --checkpoint outputs_n100_round14_stateopt_rw0_e5_seed8765/pomo_n100_optimized/model_best_service.pt \
  --num-samples 500 \
  --batch-size 32 \
  --seed 99999 \
  --decode greedy \
  --max-concurrent-open-orders 6 \
  --enable-delivery-viability \
  --min-orders-per-dispatch 4 \
  --json-output outputs_cmp/n100_round14_stateopt_rw0_e5_seed8765_eval500.json \
  2>&1 | tee logs/eval_n100_round14_stateopt_rw0_e5_seed8765.log

echo "[queue] overnight queue completed."
