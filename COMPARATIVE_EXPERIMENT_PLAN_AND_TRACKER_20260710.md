# 对比试验总控看板（N25 / N50 / N100 / N200）

- 更新时间：2026-07-16
- 负责人：bytedance + Claude
- 目标：在**硬约束语义不变**前提下，补齐论文级对比证据链（主对比 + 消融 + 效率）。

---

## 0. 当前冻结里程碑（pre-service-optimization baseline）

基线证据文件（冻结，不改口径）：
- `outputs_cmp/n100_ours_rw1_fixed_eval500.json`
- `outputs_cmp/n200_phaseC_rw1_e10_eval500.json`

冻结指标（用于后续服务率优化对照）：

| 规模 | checkpoint | service_rate_mean | rejected_rate_mean | unfulfilled_rate_mean | total_cost_raw_mean | business_acceptance.clean |
|---|---|---:|---:|---:|---:|---:|
| N100 | `outputs_cmp/n100_ours_rw1_fixed/pomo_n100_optimized/model_best_service.pt` | 0.84592 | 0.15408 | 0.00000 | 9604.0489 | true |
| N200 | `outputs_n200_phaseC_rw1_e10/pomo_n200_optimized/model_best_service.pt` | 0.86354 | 0.10661 | 0.02985 | 17880.3951 | false |

说明：
- N100 文件中的 `run.business_clean=false` 与 `business_acceptance.clean=true` 存在口径差异；本看板后续统一以 `business_acceptance.clean` 作为业务门禁字段。
- 本冻结里程碑完成后，后续所有服务率优化实验均以该表为对照，不允许回写覆盖。

---

## 1. 固定协议（先冻结，后执行）

### 1.1 固定随机性与样本
- 训练 seed：`1234`
- 正式评估 seed：`99999`
- 规模：`N25 / N50 / N100 / N200`
- 正式评估样本数：先 `500`，最终主表补 `2000`。

### 1.2 固定硬约束语义
- precedence / capacity / pickup TW / ride-time / trip-time / ops-end / reject 语义不变。
- 统一共享环境参数：
  - `--max-concurrent-open-orders 6`
  - `--min-orders-per-dispatch 4`
  - `--enable-delivery-viability`

### 1.3 固定当前主线真实参数口径（与你当前训练一致）
- `--batch-size 3`
- `--pomo-size 1`
- `--num-workers 4`
- `--amp --amp-dtype bf16`
- `--reject-init-bias -2.5`
- `--decode-pickup-urgency-bias 0.0`
- `--alpha-reject 575`
- 其余 reward 权重沿当前默认：`energy=1.0, delay=2.5, vehicle=3.0, unfulfilled=750, overtime=200`
- normalization profile 路径统一：
  - `--normalization-dir /root/autodl-tmp/project-vrp-v6-4-main/outputs/normalization_profiles`

---

## 2. 对比方法集合（论文主表）

1. **Ours-Champion**（`reject-warmup-epochs=1`）
2. **Ours-Base**（`reject-warmup-epochs=0`）
3. **Gurobi-distance**
4. **Gurobi-cost + distance重评估真实成本**
5. **GA**
6. **SA**

> 当前仓库已支持 GA/SA 的 N200 CLI：
> - `ga_mcvrppdtw.py` 已支持 `choices=[25,50,100,200]`
> - `sa_mcvrppdtw.py` 已支持 `choices=[25,50,100,200]`
> - `evaluate_gurobi_real_cost.py` 已支持 `choices=[25,50,100,200]`

---

## 3. 主矩阵

| 规模 | Ours-Champion | Ours-Base | Gurobi-distance | Gurobi-cost/reeval | GA | SA |
|---|---:|---:|---:|---:|---:|---:|
| N25  | ✅ | ✅ | ⏳ | ⏳ | ⏳ | ⏳ |
| N50  | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| N100 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| N200 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## 4. 消融矩阵（单变量）

1. `reject-warmup-epochs: 0 vs 1`（核心）
2. `reject-warmup-epochs: 1 vs 3`
3. `decode-pickup-urgency-bias: 0.0 vs 0.10`
4. `pomo-size: 1 vs 2`

> 执行纪律：失败即回退，不并行开旁支。

---

## 5. 执行顺序（降成本）

1. N25 全矩阵先跑通（校验流程和字段）
2. N50 全矩阵
3. N100 全矩阵
4. N200 全矩阵
5. 统计检验与论文图表（均值±方差 + CI）

---

## 6. 一键批量命令（云端）

> 工作目录：`/root/autodl-tmp/project-vrp-v6-4-main`

### 6.1 目录准备

```bash
mkdir -p outputs_cmp outputs/gurobi_reevaluated
```

### 6.2 Ours-Champion（rw=1）批量训练

```bash
for N in 25 50 100 200; do
  time python3 run_training_optimized.py \
    --graph_sizes ${N} \
    --normalization-dir /root/autodl-tmp/project-vrp-v6-4-main/outputs/normalization_profiles \
    --epochs 10 --epoch-size 1024 --val-size 500 \
    --batch-size 3 --pomo-size 1 --seed 1234 --num-workers 4 \
    --max-concurrent-open-orders 6 --min-orders-per-dispatch 4 \
    --enable-delivery-viability \
    --amp --amp-dtype bf16 \
    --reject-warmup-epochs 1 \
    --reject-init-bias -2.5 \
    --decode-pickup-urgency-bias 0.0 \
    --alpha-reject 575 \
    --output-root outputs_cmp/n${N}_ours_rw1 \
    2>&1 | tee outputs_cmp/n${N}_ours_rw1.log

done
```

### 6.3 Ours-Base（rw=0）批量训练

```bash
for N in 25 50 100 200; do
  time python3 run_training_optimized.py \
    --graph_sizes ${N} \
    --normalization-dir /root/autodl-tmp/project-vrp-v6-4-main/outputs/normalization_profiles \
    --epochs 10 --epoch-size 1024 --val-size 500 \
    --batch-size 3 --pomo-size 1 --seed 1234 --num-workers 4 \
    --max-concurrent-open-orders 6 --min-orders-per-dispatch 4 \
    --enable-delivery-viability \
    --amp --amp-dtype bf16 \
    --reject-warmup-epochs 0 \
    --reject-init-bias -2.5 \
    --decode-pickup-urgency-bias 0.0 \
    --alpha-reject 575 \
    --output-root outputs_cmp/n${N}_ours_rw0 \
    2>&1 | tee outputs_cmp/n${N}_ours_rw0.log

done
```

### 6.4 Ours 正式评估（500样本）

```bash
for N in 25 50 100 200; do
  python3 evaluate_model.py \
    --graph-size ${N} \
    --checkpoint outputs_cmp/n${N}_ours_rw1/pomo_n${N}_optimized/model_best_service.pt \
    --num-samples 500 --batch-size 32 --seed 99999 --decode greedy \
    --max-concurrent-open-orders 6 --enable-delivery-viability --min-orders-per-dispatch 4 \
    --json-output outputs_cmp/n${N}_ours_rw1/eval_best_service_500.json

done
```

### 6.5 Gurobi（distance / cost）

```bash
# 注：gurobi.py 当前仅显式提供 25/50/100 time-limit 参数；N200 使用默认 1800s。
python3 gurobi.py --graph-sizes 25 50 100 200 --num-samples 500 --seed 99999 \
  --objective-mode distance --time-limit-25 1200 --time-limit-50 1800 --time-limit-100 3600

python3 gurobi.py --graph-sizes 25 50 100 200 --num-samples 500 --seed 99999 \
  --objective-mode cost --time-limit-25 1200 --time-limit-50 1800 --time-limit-100 3600
```

### 6.6 Gurobi distance 重评估为真实成本

```bash
for N in 25 50 100 200; do
  python3 evaluate_gurobi_real_cost.py \
    --graph_size ${N} \
    --input gurobi_results_${N}_distance.json \
    --output outputs/gurobi_reevaluated/gurobi_reevaluated_${N}.json

done
```

### 6.7 GA / SA 批量

```bash
for N in 25 50 100 200; do
  python3 ga_mcvrppdtw.py \
    --graph_size ${N} --num_samples 500 --seed 99999 \
    --population_size 48 --generations 120 \
    --output_file ga_results_${N}.json

done

for N in 25 50 100 200; do
  python3 sa_mcvrppdtw.py \
    --graph_size ${N} --num_samples 500 --seed 99999 \
    --iterations 1500 --initial_temperature 50.0 --cooling_rate 0.995 \
    --output_file sa_results_${N}.json

done
```

### 6.8 DRL vs Gurobi 汇总（兼容 compare 脚本默认 checkpoint 路径）

```bash
for N in 25 50 100 200; do
  mkdir -p outputs/pomo_n${N}_optimized
  cp outputs_cmp/n${N}_ours_rw1/pomo_n${N}_optimized/model_best_service.pt \
     outputs/pomo_n${N}_optimized/model_best.pt

done

python3 compare_drl_vs_gurobi.py --graph-sizes 50 100 200 --num-samples 500 --seed 99999
```

---

## 7. 结果回填模板（每次跑完 3 分钟内）

### 7.1 主对比结果表

| 日期 | 规模 | 方法 | 配置摘要 | service_rate_mean | total_cost_raw_mean | rejected_rate_mean | unfulfilled_rate_mean | time/instance(s) | 结论 |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| 2026-07-11 | N25 | Ours-legacy | checkpoint=`outputs/pomo_n25_optimized/model_best_service.pt`; eval500 seed=99999 | 0.594400 | 6138.241 | 0.403920 | 0.001680 | - | 历史参考（非 fixed protocol 主线） |
| 2026-07-12 | N25 | Ours-rw1 (fixed protocol) | checkpoint=`outputs_cmp/n25_ours_rw1_fixed/pomo_n25_optimized/model_best_service.pt`; eval500 seed=99999（用户日志回传，待同步JSON） | 0.831000 | 2582.804 | 0.169000 | 0.000000 | - | 新 N25 champion（相对 rw0 更优） |
| 2026-07-12 | N25 | Ours-rw0 (fixed protocol) | checkpoint=`outputs_cmp/n25_ours_rw0_fixed/pomo_n25_optimized/model_best_service.pt`; eval500 seed=99999（用户日志回传，待同步JSON） | 0.814000 | 2894.412 | 0.185000 | 0.002000 | - | 对照基线（rw1 PASS） |
| 2026-07-11 | N200 | Ours-rw1 (PhaseC-e10) | checkpoint=`outputs_n200_phaseC_rw1_e10/.../model_best_service.pt`; eval500 seed=99999 | 0.863540 | 17880.395 | 0.106610 | 0.029850 | - | PASS（当前可接受阈值>=0.85） |
| 2026-07-11 | N200 | Ours-rw1 (PhaseB baseline) | checkpoint=`outputs_n200_phaseB_rw1_warmup1/.../model_best_service.pt`; eval500 seed=99999 | 0.705630 | 37041.609 | 0.225720 | 0.068650 | - | 对照基线 |
| 2026-07-11 | N50 | Ours-legacy | checkpoint=`outputs/n50_warmstart_from_n25/.../model_best_service.pt`; eval500 seed=99999 | 0.691000 | 9377.027 | 0.309000 | 0.000400 | - | 结果偏低，建议按当前口径重训 |
| 2026-07-11 | N50 | Ours-rw1 (fixed protocol, e10) | checkpoint=`outputs_cmp/n50_ours_rw1/pomo_n50_optimized/model_best_service.pt`; eval500 seed=99999 | 0.837360 | 4954.603 | 0.162640 | 0.000000 | - | 固定口径旧 champion（已被 resume20 超越） |
| 2026-07-11 | N50 | Ours-rw1 (resume20) | checkpoint=`outputs_cmp/n50_ours_rw1_resume20/pomo_n50_optimized/model_best_service.pt`; eval500 seed=99999 | 0.845000 | 4734.072 | 0.155000 | 0.000000 | - | 新 N50 champion（用户云端日志回传） |
| 2026-07-11 | N50 | Ours-rw1 (pomo=2 ablation) | checkpoint=`outputs_cmp/n50_ours_rw1_pomo2/pomo_n50_optimized/model_best_service.pt`; eval500 seed=99999 | 0.778000 | 6690.745 | 0.222000 | 0.000000 | - | FAIL（相对 champion 回退） |
| 2026-07-11 | N100 | Ours-rw1 (fixed protocol) | checkpoint=`outputs_cmp/n100_ours_rw1_fixed/pomo_n100_optimized/model_best_service.pt`; train best val service=0.836, eval500 seed=99999 | 0.846000 | 9604.049 | 0.154000 | 0.000000 | - | 新口径结果（可入主表） |
| 2026-07-11 | N100 | Ours-legacy(e12) | checkpoint=`outputs/n100_warmstart_from_n50_calibrated_e12/.../model_best_service.pt`; eval500 seed=99999 | 0.638640 | 21513.919 | 0.361360 | 0.000000 | - | 已记录（legacy，仅作历史对照） |

### 7.2 训练效率表（DRL）

| 日期 | 规模 | 方案 | epoch_wall_clock | samples_per_s | best_service(epoch) | best_objective(epoch) | 结论 |
|---|---|---|---:|---:|---|---|---|
| YYYY-MM-DD | N200 | Ours-rw1 |  |  |  |  |  |

### 7.3 消融结果表

| 日期 | 规模 | 变量 | 对照A | 对照B | A结果 | B结果 | PASS/FAIL | 备注 |
|---|---|---|---|---|---|---|---|---|
| 2026-07-11 | N50 | pomo-size | 1 | 2 | service=0.840, cost=7086.755 | service=0.778, cost=6690.745 | FAIL | 服务率明显下降，回退到 pomo=1 |
| YYYY-MM-DD | N100 | reject-warmup-epochs | 0 | 1 |  |  |  |  |

---

## 8. 判定规则（论文口径）

1. 主比较优先看 `service_rate_mean` 与 `total_cost_raw_mean`。
2. 同时报告 `rejected_rate_mean`、`unfulfilled_rate_mean`、推理/求解耗时。
3. 质量接近时优先耗时更低；耗时接近时优先业务洁净项更好。
4. 任一违反硬约束语义的方案直接判无效。

---

## 8.1 N100 对比试验补齐清单（优先执行）

### Step A：确认 Ours-N100 候选 checkpoint

优先顺序：
1. `outputs/n100_warmstart_from_n50_calibrated_e12/pomo_n100_optimized/model_best_service.pt`
2. `outputs/n100_warmstart_from_n50_calibrated/pomo_n100_optimized/model_best_service.pt`
3. 如有新口径训练结果，以新口径为准。

> 严谨性说明：上述 1/2 属于 legacy 路径（checkpoint 内部参数与 1.3 固定口径不完全一致，如 `batch_size/pomo_size/epoch_size/val_size/reject_warmup_epochs`）；用于“历史对照”可以保留，但若进入主表比较，必须补跑固定口径的 N100（与 N50/N200 对齐）。

### Step B：统一评估 Ours（500 样本，seed=99999）

```bash
python3 evaluate_model.py \
  --graph-size 100 \
  --checkpoint outputs/n100_warmstart_from_n50_calibrated_e12/pomo_n100_optimized/model_best_service.pt \
  --num-samples 500 --batch-size 32 --seed 99999 --decode greedy \
  --max-concurrent-open-orders 6 --enable-delivery-viability --min-orders-per-dispatch 4 \
  --json-output outputs_cmp/n100_ours_legacy_e12_eval500.json
```

### Step C：跑 N100 基线（同样 seed=99999）

```bash
python3 gurobi.py --graph-sizes 100 --num-samples 500 --seed 99999 --objective-mode distance --time-limit-100 3600
python3 gurobi.py --graph-sizes 100 --num-samples 500 --seed 99999 --objective-mode cost --time-limit-100 3600

python3 evaluate_gurobi_real_cost.py \
  --graph_size 100 \
  --input gurobi_results_100_distance.json \
  --output outputs/gurobi_reevaluated/gurobi_reevaluated_100.json \
  --seed 99999

python3 ga_mcvrppdtw.py --graph_size 100 --num_samples 500 --seed 99999 --population_size 48 --generations 120 --output_file ga_results_100.json
python3 sa_mcvrppdtw.py --graph_size 100 --num_samples 500 --seed 99999 --iterations 1500 --initial_temperature 50.0 --cooling_rate 0.995 --output_file sa_results_100.json
```

### Step D：参数一致性复核（提交前必做）

- 同规模（N100）对比必须统一：`num_samples=500`, `seed=99999`。
- DRL 评估 state 开关统一：`max_open=6`, `min_orders_per_dispatch=4`, `enable_delivery_viability=True`。
- Gurobi 重评估 seed 已支持外部传入（默认 99999）；必须与 DRL 一致。
- GA/SA 结果文件现已包含：`service_rate_mean`, `rejected_rate_mean`, `unfulfilled_rate_mean`, `business_clean`，以及协议对齐字段（`protocol_version`, `comparable_to_drl`, `hard_violation_*`）。
- 对比主表仅纳入 `comparable_to_drl=true` 的结果；legacy 文件仅作历史参考。

---

## 9. 下一阶段服务率优化队列（基于冻结基线）

### 9.1 固定评估口径（不可漂移）
- `evaluate_model.py --decode greedy --seed 99999 --num-samples 500`（候选筛选）
- 晋级候选必须补 `--num-samples 2000` 复核
- state 参数固定：
  - `--max-concurrent-open-orders 6`
  - `--min-orders-per-dispatch 4`
  - `--enable-delivery-viability`
  - `--disable-viability-fallback`（若显式传参）

### 9.2 N200 单变量队列（优先执行）
1. `decode_pickup_urgency_bias: 0.00 -> 0.03`
2. `decode_pickup_urgency_bias: 0.03 -> 0.06`
3. `alpha_unfulfilled: 750 -> 850`
4. `alpha_reject: 575 -> 620`（仅在前3项不达标时启用）

执行纪律：
- 一次只改 1 个变量；
- 每个候选都输出独立 JSON（文件名包含变量和值）；
- FAIL 即回退，不并行开枝。

### 9.3 晋级门槛（相对冻结基线）
- N200 `service_rate_mean` 至少提升 +1pp；
- `unfulfilled_rate_mean` 不升高（目标下降）；
- `business_acceptance.clean` 不退化（优先从 false 变 true）；
- `audit.partition_consistent_samples == total_samples` 且 `audit.core_aggregate_match_samples == total_samples`。

### 9.4 当前待办
1. ✅ 完成主实验冻结里程碑回填（N100/N200 基线写入本看板）。
2. 跑 N200 单变量 A1：`decode_pickup_urgency_bias=0.03`。
3. 若 A1 FAIL，顺序执行 A2/A3/A4（按 9.2 队列）。
4. 对晋级候选补 `num_samples=2000` 正式复核。
5. 同步更新主表 + 消融表 + 效率表（只记录通过门槛的候选）。

---

## 10. 备注

- 本文件是“对比试验唯一记录入口”。
- 保持“失败即回退、单变量、口径一致”。
