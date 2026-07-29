#!/usr/bin/env bash
set -u
set -o pipefail

SEED=99999
TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="outputs_cmp/gurobi_subset_${TS}"
LOG_DIR="logs/gurobi_subset_${TS}"
MASTER_LOG="${LOG_DIR}/master.log"

mkdir -p "$OUT_DIR" "$LOG_DIR"

# 只跑 distance 可改成: MODE_LIST=(distance)
MODE_LIST=(distance cost)

SIZES=(25 50 100 200)
SAMPLES=(20 10 1 1)

for mode in "${MODE_LIST[@]}"; do
  for i in "${!SIZES[@]}"; do
    n="${SIZES[$i]}"
    k="${SAMPLES[$i]}"
    LOG="${LOG_DIR}/gurobi_n${n}_${mode}.log"

    echo "[$(date '+%F %T')] START mode=${mode} n=${n} samples=${k} seed=${SEED}" | tee -a "$MASTER_LOG"

    python3 gurobi.py \
      --graph-sizes "$n" \
      --num-samples "$k" \
      --seed "$SEED" \
      --objective-mode "$mode" \
      --time-limit-25 600 \
      --time-limit-50 1200 \
      --time-limit-100 1800 \
      2>&1 | tee "$LOG"

    rc=${PIPESTATUS[0]:-1}
    echo "[$ %T')] EXIT_CODE mode=${mode} n=${n} rc=${rc}" | tee -a "$MASTER_LOG"

    OUT_FILE="gurobi_results_${n}_${mode}.json"
    if [[ -f "$OUT_FILE" ]]; then
      mv "$OUT_FILE" "$OUT_DIR/"
      echo "[$(date '+%F %T')] DONE mode=${mode} n=${n} -> ${OUT_DIR}/${OUT_FILE}" | tee -a "$MASTER_LOG"
    else
      echo "[$(date '+%F %T')] WARN missing ${OUT_FILE}" | tee -a "$MASTER_LOG"
    fi
  done
done

echo "[$(date '+%F %T')] ALL_DONE out_dir=${OUT_DIR}" | tee -a "$MASTER_LOG"
