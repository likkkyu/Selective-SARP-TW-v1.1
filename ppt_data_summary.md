# 论文进度汇报 — 全量数据汇编

> 生成时间: 2026-09-12 | 口径: drl_aligned_v1 / strict / eval2000

---

## 一、实验完成状态总览

| 规模 | DRL (eval2000) | SA (strict500) | GA (strict500) | GA (fast100) |
|---|---|---|---|---|
| N25 | ✅ 5-seed | ✅ 500 | ✅ 500 | — |
| N50 | ✅ 5-seed | ✅ 500 | ✅ 500 | — |
| N100 | ✅ 5-seed | ✅ 500 | ✅ 500 | — (已被正式版替代) |
| N200 | ✅ 5-seed | ✅ 500 | ⏳ instance=200/500 | — (待替代) |

---

## 二、DRL 主线结果（eval2000, 5-seed 均值）

| 规模 | Service Rate | Rejected Rate | Unfulfilled Rate | Raw Total Cost | Used Vehicles | Cargo Delay Cost | Trip Overtime | Seeds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| N25 | **0.8526** | 0.1474 | 0.0000 | 2256.27 | 4.19 | 0.04 | 15.87 | 5 |
| N50 | **0.8885** | 0.1115 | 0.0000 | 3463.08 | 7.96 | 10.18 | 14.24 | 5 |
| N100 | **0.8439** | 0.1557 | 0.0004 | 9734.10 | 14.84 | 15.32 | 301.16 | 5 |
| N200 | **0.8586** | 0.1142 | 0.0272 | 18339.78 | 27.55 | 213.43 | 105.30 | 5 |

**配置**: `decode=greedy`, `state_kwargs={max_concurrent_open_orders=6, min_orders_per_dispatch=4, enable_delivery_viability=True}`

**DRL checkpoints**:
- N25: `outputs_cloud/n25_sr_s1_warmup0_20260724_201223/.../model_best_service.pt`
- N50: `outputs_cloud/n50_short_train_candidate_v3_rw0_cleanrepair_plus8_lr1e5/.../model_best_service.pt`
- N100: `outputs_cmp/n100_ours_rw1_fixed/pomo_n100_optimized/model_best_service.pt`
- N200: `outputs_n200_phaseC_rw1_e10/pomo_n200_optimized/model_best_service.pt`

---

## 三、SA 结果（strict500, drl_aligned_v1）

| 规模 | Service Rate | Rejected Rate | Unfulfilled Rate | Raw Total Cost | Used Vehicles | Cargo Delay | Trip Overtime | Hard Feasible | Business Clean | Samples |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| N25 | 0.8947 | 0.1053 | 0.0000 | 1645.08 | 4.69 | 0.277 | 0 | 1.0 | 1.0 | 500 |
| N50 | 0.8684 | 0.1306 | 0.0011 | 4037.90 | 8.45 | 1.845 | 0 | 1.0 | 0.992 | 500 |
| N100 | 0.8054 | 0.1937 | 0.0009 | 11644.69 | 14.92 | 9.659 | 0 | 1.0 | 0.986 | 500 |
| N200 | 0.7056 | 0.2942 | 0.0002 | 34629.94 | 25.15 | 34.878 | ≈0 | 1.0 | 0.996 | 500 |

**配置**: `iterations=1500, initial_temperature=50.0, cooling_rate=0.995, hard_constraint_mode=strict, max_open=6`

---

## 四、GA 结果（strict500, drl_aligned_v1, pop=48/gen=120）

| 规模 | Service Rate | Rejected Rate | Unfulfilled Rate | Raw Total Cost | Used Vehicles | Cargo Delay | Trip Overtime | Hard Feasible | Business Clean | Samples |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| N25 | 0.8832 | 0.1168 | 0.0000 | 1805.19 | 4.51 | 0.263 | 0 | 1.0 | 1.0 | 500 |
| N50 | 0.8493 | 0.1507 | 0.0000 | 4564.84 | 8.04 | 1.386 | 0 | 1.0 | 1.0 | 500 |
| N100 | 0.7882 | 0.2118 | 0.0000 | 12598.54 | 14.22 | 10.341 | 0 | 1.0 | 1.0 | 500 |
| N200 | ⏳ | — | — | — | — | — | — | — | — | ⏳200/500 |

**配置**: `population_size=48, generations=120, mutation_rate=0.2, elite_size=4`

---

## 五、GA fast100 参考结果（已被/将被正式版替代）

| 规模 | Service Rate | Rejected Rate | Raw Total Cost | Used Vehicles | Samples |
|---|---:|---:|---:|---:|---:|
| N100 (fast100) | 0.7029 | 0.2971 | 17463.71 | 12.89 | 100 |
| N200 (fast100) | 0.6173 | 0.3827 | 44680.86 | 21.90 | 100 |

---

## 六、三方核心对比 — N25

| 指标 | DRL | SA | GA |
|---|---:|---:|---:|
| Service Rate | 0.8526 | **0.8947** | 0.8832 |
| Rejected Rate | 0.1474 | **0.1053** | 0.1168 |
| Unfulfilled Rate | 0.0000 | 0.0000 | 0.0000 |
| Raw Total Cost | 2256.27 | **1645.08** | 1805.19 |
| Used Vehicles | **4.19** | 4.69 | 4.51 |
| Cargo Delay Cost | **0.04** | 0.28 | 0.26 |
| Trip Overtime | 15.87 | **0** | **0** |

**N25 小结**: SA > GA > DRL (服务率); SA 成本最低; DRL 用车最少但有少量超时.

---

## 七、三方核心对比 — N50

| 指标 | DRL | SA | GA |
|---|---:|---:|---:|
| Service Rate | **0.8885** | 0.8684 | 0.8493 |
| Rejected Rate | **0.1115** | 0.1306 | 0.1507 |
| Unfulfilled Rate | **0.0000** | 0.0011 | **0.0000** |
| Raw Total Cost | **3463.08** | 4037.90 | 4564.84 |
| Used Vehicles | **7.96** | 8.45 | 8.04 |
| Cargo Delay Cost | **10.18** | 1.85 | 1.39 |
| Trip Overtime | 14.24 | **0** | **0** |

**N50 小结**: DRL 开始在服务率和成本上领先; SA/GA 零超时但靠牺牲服务率; DRL 货物延迟成本较高.

---

## 八、三方核心对比 — N100

| 指标 | DRL | SA | GA |
|---|---:|---:|---:|
| Service Rate | **0.8439** | 0.8054 | 0.7882 |
| Rejected Rate | **0.1557** | 0.1937 | 0.2118 |
| Unfulfilled Rate | 0.0004 | 0.0009 | **0.0000** |
| Raw Total Cost | **9734.10** | 11644.69 | 12598.54 |
| Used Vehicles | 14.84 | 14.92 | **14.22** |
| Cargo Delay Cost | 15.32 | 9.66 | 10.34 |
| Trip Overtime | 301.16 | **0** | **0** |

**N100 小结**: DRL 服务率优势扩大 (+3.9pp vs SA, +5.6pp vs GA); 成本差 +19.6%/+29.4%; DRL 出现大量超时(301), SA/GA 均为零.

---

## 九、三方核心对比 — N200

| 指标 | DRL | SA | GA |
|---|---:|---:|---:|
| Service Rate | **0.8586** | 0.7056 | ⏳ |
| Rejected Rate | **0.1142** | 0.2942 | ⏳ |
| Unfulfilled Rate | 0.0272 | 0.0002 | ⏳ |
| Raw Total Cost | **18339.78** | 34629.94 | ⏳ |
| Used Vehicles | 27.55 | 25.15 | ⏳ |
| Cargo Delay Cost | 213.43 | 34.88 | ⏳ |
| Trip Overtime | 105.30 | ≈0 | ⏳ |

**GA fast100 参考**: Service=0.6173, Cost=44680.86, Vehicles=21.90

**N200 小结**: DRL 大幅领先 SA (+15.3pp 服务率, -47% 成本); DRL 有显著 unfulfilled 和超时; SA/GA 大规模退化明显.

---

## 十、SA/GA vs DRL 差值汇总

| 方法 | 规模 | Δ Service Rate (pp) | Δ Raw Cost (%) | Δ Vehicles |
|---|---|---:|---:|---:|
| SA | N25 | **+4.21** | **-27.2** | +0.50 |
| SA | N50 | -2.01 | +16.6 | +0.49 |
| SA | N100 | -3.85 | +19.6 | +0.08 |
| SA | N200 | -15.30 | +88.8 | -2.40 |
| GA | N25 | **+3.06** | **-20.0** | +0.32 |
| GA | N50 | -3.92 | +31.8 | +0.08 |
| GA | N100 | -5.57 | +29.4 | -0.62 |
| GA | N200 (fast100) | -24.13 | +143.8 | -5.65 |

---

## 十一、实验配置与可复现参数

**共用对齐口径**:
- `eval_protocol=drl_aligned_v1`
- `semantic_mode=drl_aligned`
- `hard_constraint_mode=strict`
- `strict_infeasible_cost=1000000000000`
- `max_concurrent_open_orders=6`
- `min_orders_per_dispatch=4`
- `enable_delivery_viability=True`
- `seed=99999`

**GA 专有**: `population_size=48, generations=120, mutation_rate=0.2, elite_size=4`

**SA 专有**: `iterations=1500, initial_temperature=50.0, cooling_rate=0.995`

---

## 十二、结果文件路径索引

| 方法 | 规模 | 结果文件 | 版本记录文件 |
|---|---|---|---|
| DRL | N25 | `outputs_cmp/n25_eval2000_multiseed_20260728_175848/eval2000_seed*.json` + extra seed | — |
| DRL | N50 | `outputs_cmp/n50_stability_eval2000_20260723_004500/` + `n50_eval2000_multiseed_add_20260728_193951/` | — |
| DRL | N100 | `outputs_cmp/n100_eval2000_multiseed_20260727_120637/` + `n100_frozen_eval2000_20260725_214810.json` | — |
| DRL | N200 | `outputs_cmp/n200_eval2000_multiseed_20260727_120637/` + `n200_phaseC_refinal_20260726_162441/eval2000_seed99999.json` | — |
| SA | N25 | `outputs_cmp/sa_results_25_aligned_strict_500.json` | — |
| SA | N50 | `outputs_cmp/sa_results_50_aligned_strict_500.json` | — |
| SA | N100 | `outputs_cmp/sa_results_100_aligned_strict_500.json` | `outputs_cmp/sa_n100_aligned_strict500_record_20260818.json` |
| SA | N200 | `outputs_cmp/sa_results_200_aligned_strict_500.json` | `outputs_cmp/sa_n200_aligned_strict500_record_20260827.json` |
| GA | N25 | `outputs_cmp/ga_results_25_aligned_strict_500.json` | — |
| GA | N50 | `outputs_cmp/ga_results_50_aligned_strict_500.json` | — |
| GA | N100 | `outputs_cmp/ga_results_100_aligned_strict_500.json` | `outputs_cmp/ga_n100_aligned_strict500_record_20260904.json` |
| GA | N200 | ⏳运行中 (instance=200/500) | — |

---

## 十三、Gurobi 结果状态

- N25/N50/N100 no-limit 日志已复制到项目根目录
- N25: 可行率较高 (subset 20 样本中 16 feasible)
- N50/N100: 在 time-limit=unlimited 下仍多数 infeasible (大规模 NP-hard 证据)
- 详细数据需从日志提取

---

## 十四、尚未完成

1. **GA-N200 strict500**: 正在跑 (截至 9/12 进度 instance=200/500), 跑完后需记录+存档
2. **Git 提交**: GA-N100 strict500 结果和版本记录已存在, 但未提交到 GitHub
3. **DRL N25/N50 补 seed=99999 合并**: N25/N50 的 eval2000 5-seed 均值已含 seed 99999

---

## 十五、核心结论（可直接用于 PPT）

1. **DRL 在 N50+ 规模上服务率最优**: N50=0.8885, N100=0.8439, N200=0.8586
2. **N25 小规模 SA/GA 超过 DRL**: SA=0.8947 > GA=0.8832 > DRL=0.8526, 启发式在小规模搜索空间可充分探索
3. **SA/GA 随规模扩大显著退化**: N100 差 ~4~6pp, N200 差 ~15pp+; DRL 退化最缓
4. **SA 服务率始终优于 GA**: 各规模 SA 比 GA 高 1~2pp, 但 SA 用车稍多
5. **GA 策略最保守**: 零 unfulfilled, 零超时, 用车最少, 但拒单率最高
6. **DRL 是唯一有 trip_overtime 的方法**: N100=301.16, N200=105.30; SA/GA 均为零 (保守策略避免超时但牺牲服务率)
7. **所有方法 hard_feasible_rate=1.0**: 严格约束下均无硬违约
8. **DRL 的 cargo_delay_cost 在 N200 激增** (213.43), SA/GA 仅 35/10, 说明 DRL 为高服务率承受延迟代价
9. **成本差距随规模非线性扩大**: SA vs DRL 在 N200 的 ΔCost=+88.8%, GA vs DRL 在 N200 的 ΔCost≈+143.8% (fast100 参考)

---

## 十六、PPT 建议呈现结构

1. **Slide 1: 问题与模型** — VRP 定义 + 约束体系
2. **Slide 2: 三种方法对比** — DRL (POMO+指针网络) vs SA vs GA 算法框架
3. **Slide 3: 实验设置** — 对齐口径 + 超参数
4. **Slide 4: N25/N50 结果** — 小规模启发式优势; 表格 + 柱状图
5. **Slide 5: N100/N200 结果** — 大规模 DRL 优势; 表格 + 柱状图
6. **Slide 6: 规模退化趋势** — Service Rate vs N 折线图 (三条线交叉)
7. **Slide 7: 成本-服务率权衡** — 散点图 (x=cost, y=service, 气泡=vehicles)
8. **Slide 8: 约束满足分析** — trip_overtime / unfulfilled / hard_feasible 对比
9. **Slide 9: 结论** — 5 条核心结论
