#!/bin/bash
# Selective SARP-TW v1.1 一键训练脚本 (在 GPU 服务器上运行)
#
# 当前实现说明:
# 不再硬编码 --epochs 100 / --batch-size 256, 完全交由
# run_training_optimized.py 的 PHASE_CONFIGS 决定:
#   N=25 :  epochs=50, batch=8, pomo=2
#   N=50 :  epochs=60, batch=8, pomo=2
#   N=100:  epochs=80, batch=4, pomo=1
# 这样 N=100 不会因强制 batch=256 而显存爆掉。
#
# 用法:
#   bash train_gpu.sh                  # 默认跑 N=25 50 100 顺序训练
#   bash train_gpu.sh "50"             # 仅跑 N=50
#   bash train_gpu.sh "25 100"         # 跑 N=25 + N=100
#   CUDA_VISIBLE_DEVICES=0 bash train_gpu.sh 100
#
# 高级覆盖 (仅 ablation 用, 不建议常规使用):
#   EPOCHS=30 BATCH_SIZE=8 bash train_gpu.sh 100
#
# 依赖: 见 environment.yml

set -e

SIZES=${1:-"25 50 100"}

echo "====================================="
echo "Selective SARP-TW v1.1 训练脚本"
echo "Graph sizes: $SIZES"
echo "(epochs / batch / pomo 由 PHASE_CONFIGS 自动适配)"
echo "====================================="

# Step 1: 训练（内部按 graph_size 自动校准归一化常数）
echo ""
echo "[Step 1/1] 启动训练 (含自动校准)..."

# 可选 ENV 覆盖 (默认完全走 PHASE_CONFIGS)
EXTRA_ARGS=""
if [ -n "$EPOCHS" ];      then EXTRA_ARGS="$EXTRA_ARGS --epochs $EPOCHS"; fi
if [ -n "$BATCH_SIZE" ];  then EXTRA_ARGS="$EXTRA_ARGS --batch-size $BATCH_SIZE"; fi
if [ -n "$POMO_SIZE" ];   then EXTRA_ARGS="$EXTRA_ARGS --pomo-size $POMO_SIZE"; fi

python run_training_optimized.py \
    --graph_sizes $SIZES \
    --calibrate-before-train \
    $EXTRA_ARGS

echo ""
echo "训练完成。Checkpoint 在 outputs/pomo_n{N}_optimized/ 下。"
echo "评估示例: python evaluate_model.py --graph-size 50"
