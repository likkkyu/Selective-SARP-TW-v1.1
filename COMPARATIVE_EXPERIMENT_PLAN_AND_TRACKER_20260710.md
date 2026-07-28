# 对比试验总控看板（N25 / N50 / N100 / N200）

- 更新时间：2026-07-28
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
| N25  | ✅ | ✅ | ⏳ | ⏳ | ⚠️ | ✅ |
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
| 2026-07-12 | N25 | Ours-rw1 (fixed protocol) | checkpoint=`outputs_cmp/n25_ours_rw1_fixed/pomo_n25_optimized/model_best_service.pt`; eval500 seed=99999（新代码复评：`outputs_cmp/code_sync_reval_20260723_104059/n25_eval500_newcode.json`） | 0.831000 | 2582.804 | 0.169000 | 0.000000 | - | 新 N25 champion（相对 rw0 更优）；新代码复评一致（service=0.831, unfulfilled=0） |
| 2026-07-25 | N25 | Ours-rw1 (S1 warmup=0, +8epoch) | checkpoint=`outputs_cloud/n25_sr_s1_warmup0_20260724_201223/pomo_n25_optimized/model_best_service.pt`; eval500=`outputs_cmp/n25_sr_s1_warmup0_eval500_20260724_201223.json`; seed=99999 | 0.850000 | 2287.508 | 0.150000 | 0.000000 | - | 相对当前 N25 固定基线显著提升（service +1.9pp、cost 下降），且 clean 维持；通过晋级门槛 |
| 2026-07-25 | N25 | Ours-rw1 (S1 warmup=0, +8epoch, eval2000) | checkpoint=`outputs_cloud/n25_sr_s1_warmup0_20260724_201223/pomo_n25_optimized/model_best_service.pt`; eval2000=`outputs_cmp/n25_sr_s1_warmup0_eval2000_20260724_201223.json`；seed=99999 | 0.853000 | 2249.912 | 0.147000 | 0.000000 | - | eval2000 复核继续优于当前 N25 固定基线（service +2.2pp、cost 进一步下降、clean 维持）；可作为 N25 新正式候选 |
| 2026-07-28 | N25 | Ours-rw1 (S1 warmup=0, eval2000补充4-seed) | checkpoint=`outputs_cloud/n25_sr_s1_warmup0_20260724_201223/pomo_n25_optimized/model_best_service.pt`; dir=`outputs_cmp/n25_eval2000_multiseed_20260728_175848`; files=`eval2000_seed10007/10037/10067/10099.json`（云端已完成） | 0.852500 | 2257.860 | 0.147500 | 0.000000 | - | 新增4个 seed 后结论稳定，且均通过审计一致性（每个 seed `partition/core=2000/2000`）。 |
| 2026-07-28 | N25 | Ours-rw1 (S1 warmup=0, eval2000合并5-seed) | 合并口径=`seed99999` + `10007/10037/10067/10099`；evidence=`outputs_cmp/N25_N50_EVAL2000_MULTI_20260728_SUMMARY.md` | 0.852604 | 2256.270 | 0.147396 | 0.000000 | - | 5-seed 合并后主结论不变：N25 继续 clean（unfulfilled=0）且稳定优于 N25 fixed baseline。 |
| 2026-07-12 | N25 | Ours-rw0 (fixed protocol) | checkpoint=`outputs_cmp/n25_ours_rw0_fixed/pomo_n25_optimized/model_best_service.pt`; eval500 seed=99999（用户日志回传，待同步JSON） | 0.814000 | 2894.412 | 0.185000 | 0.002000 | - | 对照基线（rw1 PASS） |
| 2026-07-24 | N25 | SA (drl_aligned_v1, strict, n=500) | `outputs_cmp/sa_results_25_aligned_strict_500.json`; semantic=`drl_aligned`; hard=`strict`; fill_missing_orders=false | 0.894720 | 1645.082 | 0.105280 | 0.000000 | - | 结果有效且 `business_clean=true`，相对 N25 DRL 当前候选（S1 eval500）在该口径下 service 更高、raw 更低；可作为 N25 经典基线主参照 |
| 2026-07-24 | N25 | GA (drl_aligned_v1, strict, n=500) | `outputs_cmp/ga_results_25_aligned_strict_500.json`; hard=`strict`; fill_missing_orders=false；**semantic_modes=['legacy']** | 0.719440 | 5388.904 | 0.000000 | 0.280560 | - | 虽标记 comparable，但该批次语义为 legacy 且出现大规模 `untouched_unrejected`（clean=false）；与 SA/DRL 可比性与有效性存疑，建议补跑对齐语义后再入主表 |
| 2026-07-26 | N50 | SA (drl_aligned_v1, strict, n=500, canonical-rerun) | `outputs_cmp/sa_results_50_aligned_strict_500.json`（由 `..._seed99999_20260725_142448.json` 覆盖更新）；semantic=`drl_aligned`; hard=`strict`; state=`max_open=6,min_dispatch=4,delivery_viability=true` | 0.868360 | 4037.900 | 0.130560 | 0.001080 | - | 旧 `sa_results_50_aligned_strict_500.json` 口径不一致，已由对齐重跑结果替换；当前结果硬约束可行率高（hard_feasible_rate=1.0）且接近 clean，可作为 N50-SA 主表候选（建议在文中注明极小 unfulfilled）。 |
| 2026-07-11 | N200 | Ours-rw1 (PhaseC-e10) | checkpoint=`outputs_n200_phaseC_rw1_e10/.../model_best_service.pt`; eval500 seed=99999（新代码复评：`outputs_cmp/code_sync_reval_20260723_104059/n200_eval500_newcode.json`） | 0.863540 | 17880.395 | 0.106610 | 0.029850 | - | 旧口径 PASS（服务率阈值）；新代码复评约 `service=0.865, unfulfilled=0.028, raw=17660.884`，方向略优但 `clean=false` 未改变，仍需继续优化 |
| 2026-07-26 | N200 | Ours-rw1 (PhaseC-e10, eval500 5-seed稳定性) | checkpoint=`outputs_n200_phaseC_rw1_e10/pomo_n200_optimized/model_best_service.pt`; eval500 seeds=`10007/10037/10067/10099/99999`; TS=`20260725_214810`; files=`outputs_cmp/n200_stability_eval500_20260725_214810/eval500_seed*.json`; report=`outputs_cmp/service_closure_report_20260725_214810.json` | 0.860116 | - | - | 0.025974 | - | 服务率在 5 个 seed 上稳定（min=0.85352, max=0.86499）；按“服务率优先”口径可接受，审计一致性通过（partition/core=500/500）。 |
| 2026-07-26 | N200 | Ours-rw1 (PhaseC-e10, eval2000收口) | checkpoint=`outputs_n200_phaseC_rw1_e10/pomo_n200_optimized/model_best_service.pt`; eval2000=`outputs_cmp/n200_phasec_eval2000_20260725_214810.json`; seed=99999 | 0.864760 | 17658.945 | 0.107730 | 0.027510 | - | eval2000 与 eval500 多 seed 结论一致；服务率维持在高位（0.86476），按当前封版口径可入最终结果。 |
| 2026-07-27 | N200 | Ours-rw1 (PhaseC-e10, refinal eval500 5-seed) | checkpoint=`outputs_n200_phaseC_rw1_e10/pomo_n200_optimized/model_best_service.pt`; refinal dir=`outputs_cmp/n200_phaseC_refinal_20260726_162441`; files=`eval500_seed10007/10037/10067/10099/99999.json` | 0.860200 | 18117.307 | 0.113600 | 0.026000 | - | refinal 复验与既有 closure 结论一致：服务率稳定在≈0.86，审计一致性全部通过（每个 seed 均 `partition/core=500/500`），可作为封版补强证据。 |
| 2026-07-27 | N200 | Ours-rw1 (PhaseC-e10, refinal eval2000) | checkpoint=`outputs_n200_phaseC_rw1_e10/pomo_n200_optimized/model_best_service.pt`; file=`outputs_cmp/n200_phaseC_refinal_20260726_162441/eval2000_seed99999.json`; seed=99999 | 0.865000 | 17652.095 | 0.108000 | 0.028000 | - | refinal eval2000 与 5-seed eval500 方向一致；审计一致性通过（`partition/core=2000/2000`），可直接并入封版证据链。 |
| 2026-07-28 | N200 | Ours-rw1 (PhaseC-e10, eval2000补充4-seed) | checkpoint=`outputs_n200_phaseC_rw1_e10/pomo_n200_optimized/model_best_service.pt`; dir=`outputs_cmp/n200_eval2000_multiseed_20260727_120637`; files=`eval2000_seed10007/10037/10067/10099.json`（用户侧已完成拉取） | 0.857000 | 18511.704 | 0.115750 | 0.027000 | - | 补充4个 eval2000 seed 后，结论保持稳定：服务率维持在≈0.857，且所有 seed 审计一致性均通过（每个 seed `partition/core=2000/2000`）。 |
| 2026-07-28 | N200 | Ours-rw1 (PhaseC-e10, eval2000合并5-seed) | 合并口径=`seed99999(refinal)` + `10007/10037/10067/10099(multiseed)`；evidence=`outputs_cmp/N200_EVAL2000_MULTI_20260727_120637_SUMMARY.md` | 0.858600 | 18339.782 | 0.114200 | 0.027200 | - | 5-seed 合并后主结论不变：N200 在服务率维度保持高位且波动可控；但 `unfulfilled≈0.027`，业务 clean 仍未达成，维持“可封版但非 clean”判断。 |
| 2026-07-24 | N200 | Ours-rw1 (PhaseD A4: alpha_reject=620, +2epoch) | closeout=`outputs_cmp/n200_phaseD_a4_closeout_20260724_112107.json`; evidence=`outputs_cmp/n200_phaseD_a4_evidence_20260724_112107.tar.gz`; eval500=`outputs_cmp/n200_phaseD_a4_r620_e2_eval500_"$TS".json`（文件名含字面量`"$TS"`，由旧脚本引号问题导致）；eval2000=`outputs_cmp/n200_phaseD_a4_r620_e2_eval2000_20260723_221750.json`; seed=99999 | 0.817650 | 22151.716 | 0.181350 | 0.000980 | - | A4 失败：虽 `unfulfilled` 显著下降，但 service 相对基线明显下降（0.865→0.818），且 `clean` 仍为 false；eval2000 同结论（service≈0.820, unfulfilled≈0.00133），停止该分支 |
| 2026-07-24 | N200 | Ours-rw1 (PhaseD A5: horizon=1.5, +2epoch) | checkpoint=`outputs_cloud/n200_phaseD_a5_h150_e2_20260724_114253/pomo_n200_optimized/model_best_service.pt`; eval500=`outputs_cmp/n200_phaseD_a5_h150_e2_eval500_20260724_114253.json`; gate=`outputs_cmp/n200_phaseD_a5_h150_e2_gate_20260724_114253.json`; seed=99999 | 0.663050 | 39701.437 | 0.335970 | 0.000980 | - | A5 失败：`unfulfilled` 下降但 service 严重回退（0.865→0.663），`clean` 未改善（false→false）；gate: `service_not_drop=false`, `clean_improved=false`, `GO_EVAL2000=false`，不晋级 |
| 2026-07-24 | N200 | Ours-rw1 (PhaseD A6: alpha_unfulfilled=780, +1epoch) | checkpoint=`outputs_cloud/n200_phaseD_a6_u780_e1_20260724_173537/pomo_n200_optimized/model_best_service.pt`; eval500=`outputs_cmp/n200_phaseD_a6_u780_e1_eval500_20260724_173537.json`; gate=`outputs_cmp/n200_phaseD_a6_u780_e1_gate_20260724_173537.json`; seed=99999 | 0.634090 | 43120.518 | 0.359960 | 0.005950 | - | A6 失败：service 进一步回落（0.865→0.634），虽 `unfulfilled` 下降但仍 `clean=false`；gate: `service_not_drop=false`, `clean_improved=false`, `GO_EVAL2000=false`，不晋级（并触发“1epoch+warmup=1”的 smoke 警告） |
| 2026-07-11 | N200 | Ours-rw1 (PhaseB baseline) | checkpoint=`outputs_n200_phaseB_rw1_warmup1/.../model_best_service.pt`; eval500 seed=99999 | 0.705630 | 37041.609 | 0.225720 | 0.068650 | - | 对照基线 |
| 2026-07-11 | N50 | Ours-legacy | checkpoint=`outputs/n50_warmstart_from_n25/.../model_best_service.pt`; eval500 seed=99999 | 0.691000 | 9377.027 | 0.309000 | 0.000400 | - | 结果偏低，建议按当前口径重训 |
| 2026-07-11 | N50 | Ours-rw1 (fixed protocol, e10) | checkpoint=`outputs_cmp/n50_ours_rw1/pomo_n50_optimized/model_best_service.pt`; eval500 seed=99999 | 0.837360 | 4954.603 | 0.162640 | 0.000000 | - | 固定口径旧 champion（已被 resume20 超越） |
| 2026-07-11 | N50 | Ours-rw1 (resume20) | checkpoint=`outputs_cmp/n50_ours_rw1_resume20/pomo_n50_optimized/model_best_service.pt`; eval500 seed=99999 | 0.845000 | 4734.072 | 0.155000 | 0.000000 | - | 新 N50 champion（用户云端日志回传） |
| 2026-07-11 | N50 | Ours-rw1 (pomo=2 ablation) | checkpoint=`outputs_cmp/n50_ours_rw1_pomo2/pomo_n50_optimized/model_best_service.pt`; eval500 seed=99999 | 0.778000 | 6690.745 | 0.222000 | 0.000000 | - | FAIL（相对 champion 回退） |
| 2026-07-21 | N50 | Ours-rw1 (short-train candidate v3 provisional) | checkpoint=`outputs_cloud/n50_short_train_candidate_v3/pomo_n50_optimized/model_best_service.pt`; eval500 seed=99999; compare baseline=`outputs_cmp/n50_ours_rw1_pomo2_eval500.json` | 0.813880 | 5732.625 | 0.173360 | 0.012280 | - | 性能门禁通过（benchmark promote），但 formal eval 出现 `business_clean_regression`（baseline clean=true → candidate clean=false）且 unfulfilled 上升，判定 REJECT，不晋级 |
| 2026-07-22 | N50 | Ours-rw0 (v3 +8epoch, b1, eval500) | checkpoint=`outputs_cloud/n50_short_train_candidate_v3_rw0_plus8_b1/pomo_n50_optimized/model_best_service.pt`; eval500 seed=99999; compare baseline=`outputs_cmp/n50_ours_rw1_pomo2_eval500.json` | 0.859000 | 4314.004 | 0.141000 | 0.000000 | - | 相对该 baseline 在 eval500 上显著更优，strict `promote_to_formal_eval`（`service +0.08068`, `unfulfilled +0.000`）；可作为高服务率候选进入更严格复核 |
| 2026-07-22 | N50 | Ours-rw0 (v3 clean-repair +8epoch, lr=1e-5, eval500) | checkpoint=`outputs_cloud/n50_short_train_candidate_v3_rw0_cleanrepair_plus8_lr1e5/pomo_n50_optimized/model_best_service.pt`; eval500 seed=99999; compare baseline=`outputs_cmp/n50_ours_rw1_pomo2_eval500.json` | 0.897000 | - | - | 0.000000 | - | `compare_eval500_research` 与 `strict@500` 均 `promote_to_formal_eval`，且 `service_rate_mean_delta=+0.11844`、`unfulfilled_rate_mean_delta=+0.00000` |
| 2026-07-22 | N50 | Ours-rw0 (v3 clean-repair +8epoch, lr=1e-5, eval2000 vs prev-best) | candidate=`outputs_cmp/n50_rw0_cleanrepair_plus8_lr1e5_eval2000.json`; baseline=`outputs_cmp/n50_formal_eval2000_20260722/n50_prev_best_eval2000.json`; seed=99999 | 0.889000 | 3452.238 | 0.111000 | 0.000000 | - | 对旧最优在 formal eval2000 上实现稳定超越：strict/research 均 `promote_to_formal_eval`，`service_rate_mean_delta=+0.05095`、`unfulfilled_rate_mean_delta=+0.00000`、`business_clean=true`，可晋级为新的 N50 strict 候选冠军 |
| 2026-07-23 | N50 | Ours-rw0 (v3 clean-repair +8epoch, lr=1e-5, stability overnight) | checkpoint=`outputs_cloud/n50_short_train_candidate_v3_rw0_cleanrepair_plus8_lr1e5/pomo_n50_optimized/model_best_service.pt`; eval500(5 seeds)=`outputs_cmp/n50_stability_eval500_20260723_004056/*`; eval2000(3 seeds)=`outputs_cmp/n50_stability_eval2000_20260723_004500/*` | eval500 mean≈0.891064 | - | - | 0.000000 | - | 稳定性复核通过：eval500/eval2000 全 seeds 均 `clean=true` 且 `unfulfilled=0`；`overall_stability_pass=true`，N50 进入可封版阶段（不再继续加 epoch） |
| 2026-07-28 | N50 | Ours-rw0 (v3 clean-repair +8epoch, lr=1e-5, eval2000补充2-seed) | checkpoint=`outputs_cloud/n50_short_train_candidate_v3_rw0_cleanrepair_plus8_lr1e5/pomo_n50_optimized/model_best_service.pt`; dir=`outputs_cmp/n50_eval2000_multiseed_add_20260728_193951`; files=`eval2000_seed10067/10099.json`（云端已完成） | 0.889500 | 3446.302 | 0.110500 | 0.000000 | - | 新增2个 seed 后维持 clean，且审计一致性通过（每个 seed `partition/core=2000/2000`）。 |
| 2026-07-28 | N50 | Ours-rw0 (v3 clean-repair +8epoch, lr=1e-5, eval2000合并5-seed) | 合并口径=`10007/10037/99999` + `10067/10099`；evidence=`outputs_cmp/N25_N50_EVAL2000_MULTI_20260728_SUMMARY.md` | 0.888660 | 3463.079 | 0.111340 | 0.000000 | - | 5-seed 合并后主结论不变：N50 结果稳定、`clean=true`，可继续作为封版证据。 |
| 2026-07-22 | N50 | Ours-rw0 (v3 +8epoch, b1, eval2000 vs prev-best) | candidate=`outputs_cmp/n50_formal_eval2000_20260722/n50_candidate_rw0_plus8_b1_eval2000.json`; baseline=`outputs_cmp/n50_formal_eval2000_20260722/n50_prev_best_eval2000.json`; seed=99999 | 0.858990 | 4324.802 | 0.140200 | 0.000810 | - | 对 `resume20` 旧最优在服务率/成本上领先（`service +0.02099`，cost 显著下降），但出现极小 unfulfilled（+0.00081）导致 `business_clean_regression`，strict `reject`、research `hold`；尚不能按 strict 晋级为新正式冠军 |
| 2026-07-11 | N100 | Ours-rw1 (fixed protocol) | checkpoint=`outputs_cmp/n100_ours_rw1_fixed/pomo_n100_optimized/model_best_service.pt`; train best val service=0.836, eval500 seed=99999 | 0.846000 | 9604.049 | 0.154000 | 0.000000 | - | 新口径结果（可入主表） |
| 2026-07-20 | N100 | Ours-rw0 round11 stateopt e8 | checkpoint=`outputs_n100_round11_stateopt_rw0_e8/pomo_n100_optimized/model_best_service.pt`; best ckpt=epoch5; eval500 seed=99999 | 0.831 | 10139.974 | 0.169 | 0.000 | - | 当前最强 stateopt+rw0 候选；formal eval 明显优于 round9，但仍低于 frozen baseline，不晋级 |
| 2026-07-25 | N100 | Ours-rw1 (S1 warmup=0, +6epoch) | checkpoint=`outputs_cloud/n100_sr_s1_warmup0_20260724_203812/pomo_n100_optimized/model_best_service.pt`; eval500=`outputs_cmp/n100_sr_s1_warmup0_eval500_20260724_203812.json`; seed=99999 | 0.833000 | 10100.523 | 0.167000 | 0.000000 | - | 相对 frozen baseline（0.846 / 9604.049 / 0）仍未达晋级门槛，判定不晋级；但优于 round9，接近 round11 stateopt 候选 |
| 2026-07-25 | N100 | Ours-rw0 stateopt overnight（snapshot锚定） | snapshot=`outputs_cloud/_snapshots/n100-mainline-20260725_023150-a00180e.tgz`; baseline=`outputs_cmp/n100_mainline_overnight_n100-mainline-20260725_023150-a00180e_20260725_023428_baseline_eval500.json`; e6_s1234=`..._e6_s1234_eval500.json`; e6_s4321=`..._e6_s4321_eval500.json`; tail2_s1234=`..._tail2_s1234_eval500.json`; seed(eval)=99999 | 0.828340 | 10425.029 | 0.171500 | 0.000160 | - | fail-fast：tail2 仍低于 baseline（baseline=`0.845920/9603.894/0`；e6_s1234=`0.825300/10560.350/0.000180`；e6_s4321=`0.804140/12293.269/0.033940`）。结论：不建议进入主线晋级；云端 final eval2000 已手动停止，不再等待该轮 2000 样本收口 |
| 2026-07-19 | N100 | Ours-rw0 round9 e8 | checkpoint=`outputs_n100_round9_scratch_mainline_rw0_e8/pomo_n100_optimized/model_best_service.pt`; eval500 seed=99999 | 0.803000 | 12037.032 | 0.197000 | 0.000000 | - | 此前最强短训候选的正式评估；已被 round11 stateopt+rw0 超越，但仍保留为历史对照 |
| 2026-07-26 | N100 | Ours-rw1 fixed（eval500 5-seed稳定性） | checkpoint=`outputs_cmp/n100_ours_rw1_fixed/pomo_n100_optimized/model_best_service.pt`; eval500 seeds=`10007/10037/10067/10099/99999`; TS=`20260725_214810`; files=`outputs_cmp/n100_stability_eval500_20260725_214810/eval500_seed*.json`; report=`outputs_cmp/service_closure_report_20260725_214810.json` | 0.848960 | - | - | 0.000624 | - | 服务率均值高于门槛（0.84592），但单 seed 存在轻微波动（min=0.84226, max=0.85470）；审计一致性通过（partition/core=500/500）。 |
| 2026-07-26 | N100 | Ours-rw1 fixed（eval2000收口） | checkpoint=`outputs_cmp/n100_ours_rw1_fixed/pomo_n100_optimized/model_best_service.pt`; eval2000=`outputs_cmp/n100_frozen_eval2000_20260725_214810.json`; seed=99999 | 0.843830 | 9735.041 | 0.155690 | 0.000480 | - | eval2000 服务率处于门槛邻域（0.84383）；按“服务率优先且可接受边界波动”口径可封版，若按 strict 门槛则建议标注为边界结果。 |
| 2026-07-27 | N100 | Ours-rw1 fixed（eval2000补充4-seed + baseline合并5-seed） | checkpoint=`outputs_cmp/n100_ours_rw1_fixed/pomo_n100_optimized/model_best_service.pt`; new seeds=`10007/10037/10067/10099`; files=`outputs_cmp/n100_eval2000_multiseed_20260727_120637/eval2000_seed*.json`（已同步本地） + baseline=`outputs_cmp/n100_frozen_eval2000_20260725_214810.json`(seed=99999) | ≈0.84397 | ≈9734.104 | ≈0.15574 | ≈0.00050 | - | 补齐后 eval2000(5-seed) 均值与既有收口一致（服务率仍在门槛邻域），不改变“可封版但属边界结果”的主结论；新增4个 seed 审计一致性均为 `partition/core=2000/2000`。 |
| 2026-07-28 | N100 | Ours-rw1 fixed（checkpoint biascheck, eval500） | checkpoints=`model_best_service/model_best/model_final/model_best_objective`; dir=`outputs_cmp/n100_ckpt_biascheck_eval500_20260727_120637`（云端已完成，待同步本地）; seed=99999 | 0.846000 | 9603.894 | 0.154000 | 0.000000 | - | 偏差复核结论：`model_best_service` 与 `model_final` / `model_best_objective` 完全一致；`model_best` 明显退化（service=0.789, raw=12800.965）。主结论稳健，无“挑选最优checkpoint导致结论反转”风险。 |
| 2026-07-28 | N200 | Ours-rw1 (PhaseC-e10, checkpoint biascheck, eval500) | checkpoints=`model_best_service/model_best/model_best_objective`（`model_final` 缺失已跳过）；dir=`outputs_cmp/n200_ckpt_biascheck_eval500_20260727_120637`（云端已完成，待同步本地）; seed=99999 | 0.865000 | 17660.884 | 0.107000 | 0.028000 | - | 偏差复核结论：`model_best_service` 与 `model_best_objective` 一致并保持当前主结论；`model_best` 呈“更低raw/零unfulfilled但更高reject与更低service”（service=0.860, rejected=0.140, raw=17134.827）。若维持“服务率优先”口径，当前checkpoint选择合理。 |
| 2026-07-11 | N100 | Ours-legacy(e12) | checkpoint=`outputs/n100_warmstart_from_n50_calibrated_e12/.../model_best_service.pt`; eval500 seed=99999 | 0.638640 | 21513.919 | 0.361360 | 0.000000 | - | 已记录（legacy，仅作历史对照） |

### 7.2 训练效率表（DRL）

| 日期 | 规模 | 方案 | epoch_wall_clock | samples_per_s | best_service(epoch) | best_objective(epoch) | 结论 |
|---|---|---|---:|---:|---|---|---|
| 2026-07-22 | N50 | v3 repair（rw0, e2, from `v3` ckpt） | 约 `~5min/epoch`（batch=16, epoch-size=1024） | - | train best `0.771` (e2), eval100 `0.7830` | train best `6759.42` (e2), eval100 raw `6556.796` | 相对 baseline eval100 呈 `service +0.0168`、`unfulfilled +0.0024`；strict 触发 `business_clean_regression` + `unfulfilled_rate_regression` => reject；research => hold（可继续研究，不可晋级）。同轮 ckpt 重筛（`model_best_service/model_best/model_best_objective/model_final`）指标与 verdict 一致，未发现可晋级漏网点。 |
| 2026-07-22 | N50 | v3 repair（rw0, +8epoch, b1） | 约 `~5.2min/epoch`（batch=16, epoch-size=1024） | - | train best `0.849` (e7), eval100 `0.8580` | train best `4377.27` (e7), eval100 raw `4342.416` | 关键突破：相对 baseline eval100 为 `service +0.0920`、`unfulfilled +0.0000`、`business_clean=true`；strict/research 均 `promote_to_formal_eval`，可进入 formal eval500 |
| 2026-07-22 | N50 | v3 clean-repair（rw0, +8epoch, lr=1e-5, resume-tail） | 约 `~5min/epoch`（batch=16, epoch-size=1024） | - | eval500 `0.8970`；eval2000 `0.8890` | eval2000 raw `3452.238` | 续跑尾段已完整收口：相对旧最优 `eval2000` 对比中 strict/research 均 `promote_to_formal_eval`，`service_rate_mean_delta=+0.05095`、`unfulfilled_rate_mean_delta=+0.00000`；满足 strict 晋级条件 |
| 2026-07-23 | N50 | v3 clean-repair（stability overnight） | eval500 5-seed + eval2000 3-seed（same checkpoint, strict protocol） | - | eval500 seeds=`0.88776/0.88720/0.89100/0.89248/0.89688`; eval2000 seeds=`0.88798/0.88742/0.88890` | - | 稳定性报告 `overall_stability_pass=true`；`stable_eval500=true`、`stable_eval2000=true`、`cross_consistent_500_vs_2000=true`；全部 seeds `clean=true` 且 `unfulfilled=0`，建议封版并停止继续加 epoch |
| 2026-07-22 | N50 | v3 repair（lr=1e-5, e2） | 约 `~5min/epoch`（batch=16, epoch-size=1024） | - | train best `0.812` (e1), eval100 `0.7972` | train best `5569.42` (e1), eval100 raw `6256.534` | 相对 baseline eval100（`service=0.766, unfulfilled=0`）呈现 `service +0.0312` 但 `unfulfilled +0.0074`、`clean=false`，strict 对比触发 `business_clean_regression`，reject；research 仅可 hold，不可晋级 |
| 2026-07-22 | N50 | v3 repair（alpha_unfulfilled=1200, e2） | 约 `~5min/epoch`（batch=16, epoch-size=1024） | - | train best `0.751` (e2), eval100 `0.7604` | train best `8462.39` (e2), eval100 raw `7570.799` | 双退化：相对 baseline eval100 为 `service -0.0056`、`unfulfilled +0.0438`，strict reject；该变量方向判定失败并停止 |
| 2026-07-21 | N50 | Harness gate + short-train candidate v3 | gate benchmark: candidate `11.381s` vs baseline `12.010s`（-5.24%） | gate benchmark: candidate `2.917` vs baseline `2.764`（+5.53%） | train best `0.815` (e4), formal eval500 `0.81388` | train best `5444.49` (e4), formal eval500 raw `5732.625` | gate 阶段 `benchmark_verdict=promote_to_short_train`；formal 对比 `outputs_cmp/n50_ours_rw1_pomo2_eval500.json` 时出现 `business_clean_regression`（baseline clean=true → candidate clean=false）且 `unfulfilled_rate +0.01228`，`eval_verdict=reject`，不晋级 |
| 2026-07-21 | N100 | Scratch stateopt rw0 e5 | - | - | 0.7675 (e5) | 13703.62 (e5) | `logs/train_n100_round12_stateopt_rw0_e5.log` 与用户回传的 `training_log.json` 显示：同一 stateopt+rw0 配方改为 `epochs=5` 后，Epoch 5 最佳训练侧仅 `service=0.7674600220`、`rejected=0.2193600082`、`unfulfilled=0.0131800008`、`business_clean=false`、`objective=13703.615234375`，明显未能复现 round11 的 epoch5 水平。随后用户在云端完成 formal eval500（结果 JSON 尚未同步回本地），`business_acceptance={'gate_unfulfilled_zero': false, 'gate_pickup_only_zero': true, 'gate_started_not_completed_zero': true, 'gate_untouched_unrejected_zero': false, 'clean': false, 'ranking_key': [0, 0.76216, -15057.06534423828]}`，且 `partition_consistent_samples=500/500`、`core_aggregate_match_samples=500/500`。结论：`e5` 早停方案失败，不可替代当前 `e8` 筛选，也不具备晋级资格 |
| 2026-07-21 | N100 | Scratch stateopt rw0 e5 seed4321 | - | - | - | - | `logs/overnight_n100_stateopt_rw0_queue.nohup.log` 用户云端日志显示：同配方换 seed 后的 formal eval500 更差，`Audited Service Rate=0.604`、`Audited Rejected Before Service Rate=0.328`、`Audited Residual Unfulfilled Rate=0.068`，`business_acceptance={'gate_unfulfilled_zero': false, 'gate_pickup_only_zero': true, 'gate_started_not_completed_zero': true, 'gate_untouched_unrejected_zero': false, 'clean': false, 'ranking_key': [0, 0.60352, -24970.444010742187]}`，且 `partition_consistent_samples=500/500`、`core_aggregate_match_samples=500/500`。结论：`e5` 早停跨 seed 明显不稳定，应停止将其作为默认短筛选路线 |
| 2026-07-20 | N100 | Scratch stateopt rw0 e8 | - | - | 0.8350 (e5) | 9500.34 (e5) | `logs/train_n100_round11_stateopt_rw0_e8.log`；在当前 stateopt 提速代码上复跑主线 `reject_warmup_epochs=0` from-scratch 命令，Epoch 5 达到最佳训练侧指标：`service=0.8350200653`、`rejected=0.1649800110`、`unfulfilled=0.0000000000`、`business_clean=true`、`objective=9500.33984375`。但后续 Epoch 6-8 明显回落（尤其 Epoch 7 出现 `unfulfilled=0.1502`），说明该配方后段不稳，最佳点锁定在 Epoch 5。随后对 `model_best_service.pt` 的 formal eval500（`outputs_cmp/n100_round11_stateopt_rw0_e8_eval500.json`）给出 `service_rate=0.831`、`rejected_rate=0.169`、`residual_unfulfilled_rate≈0.000`、`raw_cost=10139.974`，且 `partition_consistent_samples=500/500`、`core_aggregate_match_samples=500/500`。结论：formal eval 明显优于 round9 rw0 e8，是当前最强的 stateopt+rw0 候选，但仍低于 frozen baseline（0.846 / 9604.049 / 0.000），且审计仍有极少量 `pickup_only/started_not_completed` 残留，因此不晋级、不替代当前基线 |
| 2026-07-19 | N100 | Scratch mainline rw0 e8 | - | - | 0.8093 (e8) | 11241.45 (e8) | `logs/train_n100_round9_scratch_mainline_rw0_e8.log`；将主线 from-scratch 的 `reject_warmup_epochs` 从 1 改为 0 后，Epoch 7 已达 `service=0.80534, rejected=0.19466, unfulfilled=0.000, business_clean=true`，Epoch 8 进一步提升至 `service=0.80930, rejected=0.19070, unfulfilled=0.000, objective=11241.45, business_clean=true`。用户云端自检还确认 `model_best.pt` / `model_best_objective.pt` / `model_best_service.pt` / `model_final.pt` 四者完全一致，均锁定在 Epoch 8。随后同一代码快照下的正式 eval500（`outputs_cmp/n100_round9_rw0_e8_eval500.json`）给出 `service_rate=0.803`、`rejected_rate=0.197`、`unfulfilled_rate=0.000`、`raw_cost=12037.032`，且 `partition_consistent_samples=500/500`、`core_aggregate_match_samples=500/500`。结论：训练侧曾是最强 N100 短训候选，但 formal eval 明显落后于 frozen baseline（0.84592 / 9604.0489 / 0.00000），现已被 round11 stateopt+rw0 超越，不晋级、不替代当前基线 |
| 2026-07-19 | N100 | Scratch mainline e8 | - | - | 0.793 (e7) | 13063.72 (e7) | `logs/train_n100_round8_scratch_mainline_e8.log`；恢复主线 curriculum 后，from-scratch 方向明显优于 round6/round7：Epoch 7 达到 `service=0.793, rejected=0.163, unfulfilled=0.044, objective=13063.72`，Epoch 8 为 `service=0.779, rejected=0.219, unfulfilled=0.002`。但最佳点仍未超过 round3 候选（0.796 / 0.000），且 unfulfilled 偏高，判定为“接近但未晋级”，不直接扩成长跑 |
| 2026-07-19 | N100 | Scratch no-curriculum e4 | - | - | 0.482 (e1) | 35810.43 (e1) | `logs/train_n100_round7_scratch_no_curriculum_e4.log`；from-scratch 且 `disable_rideshare_curriculum` 的短筛选方向在 Epoch 1/2 即显著失败：`e1 service=0.482, unfulfilled=0.338, objective=35810.43`，`e2 service=0.410, unfulfilled=0.275, objective=38911.98`。远差于 round3 候选（0.796 / 0.000）与 frozen baseline（0.846 / 0.000），判定立即停止，不再继续该路线 |
| 2026-07-19 | N100 | Frozen round6 conservative FT | - | - | 0.776 (e3, service-priority) / 0.774 (e4, business) | 13031.71 (e4) | `logs/train_n100_round6_frozen_conservative_ft.log`；基于已同步的保守微调控制（`lr=1e-5`, `disable_rideshare_curriculum`, `reject_warmup_epochs=0`, `resume_weights_only`）后，训练过程不再像 round5 那样崩坏，但最佳 service-priority 仅 `0.776` 且伴随 `unfulfilled=0.016`，最佳 business/objective 为 `0.774 / unfulfilled=0.000 / objective=13031.71`。仍显著落后于 frozen baseline（0.846 / 0.000）与 round3 候选（0.796 / 0.000），判定 FAIL，不继续长跑 |
| 2026-07-18 | N100 | Conservative fine-tune controls | - | - | - | - | `run_training_optimized.py` 已新增 `--lr` 与 `--disable-rideshare-curriculum`，用于 frozen checkpoint 保守微调；本地已验证新参数进入 `--help`、`build_phase_args()` 可得到 `lr=1e-5` 且 curriculum 关闭，并通过 `python3 -m py_compile`；云端 `/root/autodl-tmp/project-vrp-v6-4-main` 复核也已通过，确认可直接基于该代码快照发起 round6 短程 warm-start |
| 2026-07-18 | N100 | Frozen ckpt on round3 code (eval500) | - | - | 0.846 (eval500) | 9603.894 raw cost | `outputs_cmp/n100_frozen_ckpt_on_round3_code_eval500.json`；固定口径 frozen checkpoint 在当前 round3 代码上复评与主表几乎完全一致，证明当前代码推理语义未漂移，无需整体回滚 |
| 2026-07-18 | N100 | Frozen warm-start on round3 code | - | - | 0.74988 (e4) | 14737.6123 (e4) | `logs/train_n100_round5_warmstart_frozen_round3code.log`；云端 checkpoint 自检显示 `model_best.pt` / `model_best_objective.pt` / `model_best_service.pt` 三者完全相同，均锁定在 Epoch 4，指标一致（`service=0.7498800659179687`, `rejected=0.2501200103759766`, `unfulfilled=0.0`, `avg_objective=14737.6123046875`）；`resume_weights_only` + 延长 curriculum 的精修配方失败，且 Epoch 5 已进一步退化到 `service=0.359 / unfulfilled=0.049`，建议立即停止并废弃该配方 |
| 2026-07-18 | N100 | Ours-rw1 round4 es1024 | 2657 | - | 0.790 (e10) | 12508.50 (e10) | `logs/train_n100_round4_es1024.log`；配置 `epochs=10, epoch-size=1024, val-size=500, batch-size=3, seed=1234`。最终 best 低于 round3 resume 候选（0.796@e3）与 frozen baseline（0.846 eval500），判定 FAIL，不升级 |
| 2026-07-17 | N100 | Ours-rw1 round3 code benchmark | 1867.801 | 0.556 | - | - | 同口径云端 benchmark：`avg_mean_decode_get_mask_ms=6084.441`、`avg_mean_mask_pickup_commitment_ms=4583.442`，相对前一版 N100 benchmark 吞吐约 +14.9%，证明 round3 热路径优化有效 |
| 2026-07-17 | N100 | Ours-rw1 round3 resume scan | 9806 | - | 0.796 (e3) | 11952.59 (e3) | 从 `outputs_n100_overnight_round3/.../model_best.pt` 续跑，Epoch 3 达到当前最佳；Epoch 4-7 未持续超越，建议停止长跑并保留 Epoch 3 作为服务率优化候选，不替代 frozen baseline |
| YYYY-MM-DD | N200 | Ours-rw1 |  |  |  |  |  |

### 7.3 消融结果表

| 2026-07-22 | N50 | gate profile（评估门禁） | strict | research (`max_unfulfilled_delta=0.015`, `max_unfulfilled_abs=0.015`) | strict：对 `business_clean_regression`/`unfulfilled_rate_regression` 直接 reject | research：仅在“无 service 回退且 unfulfilled 回退在阈值内”时将 reject 降级为 hold（允许继续研究，不允许晋级） | PASS（流程） | 双轨门禁长期化：research 用于短训筛选，strict 用于晋级/封版；本次候选中 `v3_rw0_e2` 在 strict=reject / research=hold，`v3_lr1e5_e2` 与 `v3_aunf1200_e2` 在 strict 下 reject |

| 日期 | 规模 | 变量 | 对照A | 对照B | A结果 | B结果 | PASS/FAIL | 备注 |
|---|---|---|---|---|---|---|---|---|
| 2026-07-11 | N50 | pomo-size | 1 | 2 | service=0.840, cost=7086.755 | service=0.778, cost=6690.745 | FAIL | 服务率明显下降，回退到 pomo=1 |
| 2026-07-16 | N200 | decode-pickup-urgency-bias | 0.00 | 0.03 | service=0.86354, unfulfilled=0.02985, raw=17880.395 | service=0.84807, unfulfilled=0.02951, raw=19633.149 | FAIL | 服务率下降约 1.55pp，成本显著上升 |
| 2026-07-16 | N200 | decode-pickup-urgency-bias | 0.00 | 0.06 | service=0.86354, unfulfilled=0.02985, raw=17880.395 | service=0.86270, unfulfilled=0.03531, raw=18176.966 | FAIL | 服务率未提升且 unfulfilled 升高 |
| 2026-07-23 | N200 | alpha_unfulfilled | 750 | 850（+2epoch, from phaseC ckpt） | service≈0.865, unfulfilled≈0.028, raw≈17660.884 | service=0.716, unfulfilled=0.004, raw=33717.441 | FAIL | 虽未履约下降，但服务率与成本断崖式退化；训练早期出现严重不稳定（Epoch1: service=0.295, unfulfilled=0.614），不晋级、不进 eval2000 |
| 2026-07-24 | N200 | alpha_reject | 575 | 620（A4，+2epoch, from phaseC ckpt） | service≈0.865, unfulfilled≈0.028, clean=false | eval500: service=0.81765, unfulfilled=0.00098；eval2000: service=0.82005, unfulfilled=0.00133；clean=false | FAIL | 虽未履约大幅下降，但 service 明显低于门槛（floor=0.85499），且 clean 未改善；closeout=`outputs_cmp/n200_phaseD_a4_closeout_20260724_112107.json` 判定 `FAIL_STOP_A4` |
| 2026-07-24 | N200 | decode-pickup-urgency-horizon-hours | 1.0 | 1.5（A5，+2epoch, from phaseC ckpt） | service≈0.865, unfulfilled≈0.028, clean=false | eval500: service=0.66305, unfulfilled=0.00098, rejected=0.33597, raw=39701.437；clean=false | FAIL | service 大幅退化（低于 gate floor=0.85499），虽 unfulfilled 下降但 `clean` 未改善；gate=`outputs_cmp/n200_phaseD_a5_h150_e2_gate_20260724_114253.json` => `GO_EVAL2000=false` |
| 2026-07-24 | N200 | alpha_unfulfilled | 750 | 780（A6，+1epoch, from phaseC ckpt） | service≈0.865, unfulfilled≈0.028, clean=false | eval500: service=0.63409, unfulfilled=0.00595, rejected=0.35996, raw=43120.518；clean=false | FAIL | service 断崖式下降（低于 gate floor=0.85499），`clean` 未改善；gate=`outputs_cmp/n200_phaseD_a6_u780_e1_gate_20260724_173537.json` => `GO_EVAL2000=false`；该 run 含 `n_epochs<=reject_warmup_epochs` 警告，仅可作 fail-fast 证据 |
| 2026-07-22 | N50 | reject-warmup-epochs（from `v3` ckpt） | 1 | 0 | eval100（e2）：service=0.7830, unfulfilled=0.0024（dirty） | eval100（+8epoch,b1）：service=0.8580, unfulfilled=0.0000（clean） | PASS（阶段性） | 该变量在短训后段出现“先脏后净”动态：e2 时 strict reject，+8epoch 后 eval100/eval500 均通过 strict；但在更严格的 eval2000 对旧最优对比中出现极小 unfulfilled（+0.00081）触发 strict reject（research hold），因此暂不晋级正式冠军 |
| 2026-07-22 | N50 | lr（from `v3` ckpt） | 5e-5 (旧) | 1e-5 | eval100：service≈0.831, unfulfilled≈0.0088（dirty） | eval100：service=0.7972, unfulfilled=0.0074（dirty） | FAIL | 仅降 lr 未能恢复 clean；虽然 unfulfilled 略降，但仍 >0，strict reject |
| 2026-07-22 | N50 | alpha_unfulfilled（from `v3` ckpt） | 750 | 1200 | eval100：service≈0.831, unfulfilled≈0.0088 | eval100：service=0.7604, unfulfilled=0.0438 | FAIL | 惩罚增大导致服务率回落且 unfulfilled 恶化，方向错误，停止 |
| 2026-07-25 | N100 | reject-warmup-epochs | 1（frozen rw1 baseline） | 0（S1: +6epoch） | eval500: service=0.846, rejected=0.154, unfulfilled=0.000, raw=9604.049 | eval500: service=0.833, rejected=0.167, unfulfilled=0.000, raw=10100.523 | FAIL | `unfulfilled` 保持 0 但 service/cost 双退化，未过 strict 晋级门槛（service floor≈0.836）；训练中出现阶段性不稳（Epoch4 临时 unfulfilled 飙升） |

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
2. ✅ 完成新代码快照一致性复评（`outputs_cmp/code_sync_reval_20260723_104059/*`）：N25/N100 与历史主结果一致，无需因代码同步补训；N200 仍为 clean=false。
3. ✅ N200 A3（`alpha_unfulfilled=850`）已完成并判定 FAIL（服务率/成本大幅退化），不进入 eval2000。
4. ✅ N200 A4（`alpha_reject=620`, +2epoch）已完成并判定 FAIL：`outputs_cmp/n200_phaseD_a4_closeout_20260724_112107.json`（eval500/eval2000 均未达到 service 门槛，clean 仍未改善）。
5. ✅ N200 A5（`decode_pickup_urgency_horizon_hours=1.5`, +2epoch）已完成并判定 FAIL：`outputs_cmp/n200_phaseD_a5_h150_e2_eval500_20260724_114253.json` + `outputs_cmp/n200_phaseD_a5_h150_e2_gate_20260724_114253.json`（`service=0.663`，`GO_EVAL2000=false`）。
6. ✅ N200 A6（`alpha_unfulfilled=780`, +1epoch）已完成并判定 FAIL：`outputs_cmp/n200_phaseD_a6_u780_e1_eval500_20260724_173537.json` + `outputs_cmp/n200_phaseD_a6_u780_e1_gate_20260724_173537.json`（`service=0.63409`，`GO_EVAL2000=false`，且该 run 含 `n_epochs<=reject_warmup_epochs` smoke 警告）。
7. ✅ 已完成 N25/N100 首轮 warmup=0 探针：N25-S1（+8epoch）eval500 结果 `service=0.850 / unfulfilled=0 / raw=2287.508`；N100-S1（+6epoch）eval500 结果 `service=0.833 / unfulfilled=0 / raw=10100.523`，低于 frozen baseline，不晋级。
8. ✅ 已完成 N25-S1 的 eval2000 收口（用户日志回传）：`service=0.853 / rejected=0.147 / unfulfilled=0.000 / raw=2249.912`，相对 N25 fixed baseline 持续领先，建议作为 N25 新正式候选。
9. ✅ 已完成 N100 stateopt 主线 overnight（snapshot=`n100-mainline-20260725_023150-a00180e`）并形成 fail-fast 结论：baseline/e6/tail2 的 eval500 均值显示候选仍低于 frozen baseline（最佳 tail2=`service=0.82834, raw=10425.029, unfulfilled=0.00016`），不建议晋级；本轮 final eval2000 已手动停止。
10. ✅ 已完成 N100/N200 closure 收口（TS=`20260725_214810`）：N100 eval500 5-seed 均值约 `0.8488`，N100 eval2000=`0.844`；N200 eval500 5-seed 均值约 `0.8602`，N200 eval2000=`0.865`；按“服务率优先且接受边界波动”口径可封版。
11. ✅ 本地已完成 N25 的 GA/SA strict500（`outputs_cmp/ga_results_25_aligned_strict_500.json`、`outputs_cmp/sa_results_25_aligned_strict_500.json`）：SA 为 `service=0.89472 / unfulfilled=0 / raw=1645.082 / clean=true`，可纳入 N25 经典基线。
12. ✅ 已用最新对齐结果替换 N50-SA canonical 文件：`outputs_cmp/sa_results_50_aligned_strict_500.json <- outputs_cmp/sa_results_50_aligned_strict_500_seed99999_20260725_142448.json`；替换后指标 `service=0.86836 / rejected=0.13056 / unfulfilled=0.00108 / raw=4037.900`，`semantic_mode=drl_aligned`、`hard_feasible_rate=1.0`。
13. ✅ 已修复 GA 语义与状态口径参数：`ga_mcvrppdtw.py` 新增 `--semantic_mode`（默认 `drl_aligned`）及 `--max_concurrent_open_orders/--min_orders_per_dispatch/--enable_delivery_viability/--allow-reject` 等参数并接入评估路径；后续 GA 结果可与 DRL/SA 严格对齐比较。
14. ⚠️ 仍需补跑 GA strict500（N25/N50/N100，建议补 N200）以替换历史 `semantic_modes=['legacy']` 文件后再入主表。
15. ✅ 已生成 closure 结果索引与拉取清单：`outputs_cmp/CLOSURE_SERVICE_20260725_214810_INDEX.md`、`outputs_cmp/CLOSURE_SERVICE_20260725_214810_PULL_LIST.txt`，用于确保所有相关文件归档在当前项目目录内。
16. ✅ 已完成 N200 PhaseC 云端→本地同步审计：新增 `outputs_cmp/N200_PHASEC_SYNC_AUDIT_20260726.md`；确认 checkpoint 与 eval/closure 关键文件已在本地主项目目录，且云端源端缺失 `outputs_n200_phaseC_rw1_e10/pomo_n200_optimized/training_log.json` 与 `outputs_n200_phaseC_rw1_e10.log`（不可补拉）。
17. ✅ 已完成 N200 PhaseC refinal 复验（云端目录：`outputs_cmp/n200_phaseC_refinal_20260726_162441`）：eval500(5-seed) 均值约 `service=0.8602 / rejected=0.1136 / unfulfilled=0.0260 / raw=18117.307`；eval2000(seed99999)=`service=0.865 / rejected=0.108 / unfulfilled=0.028 / raw=17652.095`；审计一致性通过（eval500 每 seed `partition/core=500/500`，eval2000 `partition/core=2000/2000`）。
18. ✅ 已完成 N100 eval2000 补充4-seed（云端目录：`outputs_cmp/n100_eval2000_multiseed_20260727_120637`，seeds=`10007/10037/10067/10099`）：四个新增 seed 指标分别约 `service=0.8407/0.8500/0.8421/0.8427`、`raw=9919.783/9389.766/9822.793/9803.135`，且均满足 `partition/core=2000/2000`。与既有 seed99999 合并后，N100 eval2000(5-seed) 均值约 `service=0.8440 / rejected=0.1557 / unfulfilled=0.0005 / raw=9734.104`，结论维持“门槛邻域、可封版但边界”。
19. ✅ 已完成 N200 eval2000 补充4-seed（云端目录：`outputs_cmp/n200_eval2000_multiseed_20260727_120637`，seeds=`10007/10037/10067/10099`，已同步到当前主目录）：新增4个 seed 指标约 `service=0.859/0.856/0.859/0.854`、`raw=18298.976/18584.251/18352.665/18810.922`，均满足 `partition/core=2000/2000`。4-seed 均值约 `service=0.8570 / rejected=0.1158 / unfulfilled=0.0270 / raw=18511.704`；与 refinal seed99999 合并后 5-seed 均值约 `service=0.8586 / rejected=0.1142 / unfulfilled=0.0272 / raw=18339.782`，主结论不变（可封版但非 clean）。
20. ✅ 已完成 N100/N200 checkpoint 偏差复核（eval500, seed=99999，云端目录：`outputs_cmp/n100_ckpt_biascheck_eval500_20260727_120637`、`outputs_cmp/n200_ckpt_biascheck_eval500_20260727_120637`）：
   - N100：`model_best_service`=`model_final`=`model_best_objective`（均 `service=0.846 / raw=9603.894 / unfulfilled=0`），`model_best` 明显回退（`service=0.789 / raw=12800.965`），主结论稳健。
   - N200：`model_best_service`=`model_best_objective`（均 `service=0.865 / raw=17660.884 / unfulfilled=0.028`）；`model_best` 为替代权衡点（`service=0.860 / raw=17134.827 / unfulfilled=0 / rejected=0.140`，且 `model_final` 缺失已跳过）。按“服务率优先”口径，当前主checkpoint选择合理。
21. ✅ 已完成 N25 eval2000 补充4-seed（云端目录：`outputs_cmp/n25_eval2000_multiseed_20260728_175848`，seeds=`10007/10037/10067/10099`）：新增4个 seed 指标约 `service=0.853/0.852/0.853/0.852`、`raw=2247.390/2261.271/2255.499/2267.278`，均满足 `partition/core=2000/2000`。4-seed 均值约 `service=0.8525 / rejected=0.1475 / unfulfilled=0.0000 / raw=2257.860`；与 seed99999 合并后 5-seed 均值约 `service=0.8526 / rejected=0.1474 / unfulfilled=0.0000 / raw=2256.270`，结论维持“稳定且 clean，继续优于 N25 fixed baseline”。
22. ✅ 已完成 N50 eval2000 补充2-seed（云端目录：`outputs_cmp/n50_eval2000_multiseed_add_20260728_193951`，seeds=`10067/10099`）：新增2个 seed 指标约 `service=0.889/0.890`、`raw=3462.350/3430.253`，均满足 `partition/core=2000/2000`。与既有 3-seed（`outputs_cmp/n50_stability_eval2000_20260723_004500`）合并后 5-seed 均值约 `service=0.88866 / rejected=0.11134 / unfulfilled=0.00000 / raw=3463.079`，主结论不变（稳定、clean、可封版）。

---

## 10. 备注

- 本文件是“对比试验唯一记录入口”。
- 保持“失败即回退、单变量、口径一致”。
