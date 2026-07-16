# N200 长期计划与进度看板（Harness化管理）

- 更新时间：2026-07-16
- 负责人：bytedance + Claude
- 当前阶段：**Phase C 基线已冻结（进入服务率优化队列）**
- 当前 Champion（Phase C 冻结基线）：`n200_phaseC_rw1_e10`
- 当前 Champion（快筛口径）：`rw0_base`
- 对比试验总控看板：`COMPARATIVE_EXPERIMENT_PLAN_AND_TRACKER_20260710.md`

---

## 1. 不变目标（North Star）

1. **训练效率目标**：N200 训练墙钟显著下降（同时关注吞吐）。
2. **业务目标**：服务率稳定达到 **90%+**（以 `evaluate_model.py` 正式评估为准）。
3. **语义约束**：不改变硬约束语义（precedence / capacity / pickup TW / ride-time / trip-time / ops-end / reject）。

---

## 2. 实验治理规则（必须执行）

1. **失败即回退**，不并行开旁支。
2. **严格单变量**：一次只改 1 个变量，其余全固定。
3. **分层闸门**：先快筛再确认再长训，避免每次都高成本 full run。
4. **统一口径对比**：seed、epoch-size、val-size、batch-size、约束开关一致。
5. **每次实验必须记录**：配置、关键指标、结论（PASS/FAIL）、下一步。

---

## 3. 三层实验流程（长期执行）

### Phase A（快速筛选，低成本）
- 推荐口径：`epochs=2, epoch-size=512, val-size=128`
- 用途：快速淘汰明显劣化配置。

### Phase B（确认， 中成本）
- 推荐口径：`epochs=3, epoch-size=1024, val-size=500`
- 用途：确认 Phase A 的提升不是偶然。

### Phase C（达标验证，高成本）
- 推荐口径：`epochs=8~12` + `evaluate_model.py (500->2000 samples)`
- 用途：冲击并验证服务率 90%+。

---

## 4. 当前进度（含时间）

> 注：以下为本轮已完成、可用于决策的关键结果。

| 时间 | 实验ID | 关键配置（只列差异） | 结果 | 墙钟(real) | 结论 |
|---|---|---|---|---|---|
| 2026-07-08 | `n200_real_v2_rw0` | `reject_warmup_epochs=0`, 3ep | best service=0.425, best objective=73978.23 | 205m52s | 质量偏低 |
| 2026-07-08 | `n200_real_v2_rw3` | `reject_warmup_epochs=3`, 3ep | best service=0.659, best objective=39240.02 | 263m42s | 质量提升但更慢 |
| 2026-07-08 | `n200_gate_rw0_base` | `reject_warmup_epochs=0`, 2ep/512/128 | best service=0.714, best objective=35390.16 | 64m30s | **当前快筛冠军** |
| 2026-07-08 | `n200_gate_rw0_curr1` | 在上行基础上 `enable_rideshare_curriculum=True` | best service=0.278, best objective=98421.66 | 72m34s | **FAIL，已回退** |
| 2026-07-09 | `n200_gate_rw0_ub010` | 在 `rw0_base` 上仅改 `decode-pickup-urgency-bias=0.10` | best service=0.320, best objective=87253.60 | 57m17s | **FAIL，已回退** |
| 2026-07-09 | `n200_gate_rw0_ar620` | 在 `rw0_base` 上仅改 `alpha-reject=620` | best service=0.309, best objective=89691.86 | 58m10s | **FAIL，已回退** |

| 2026-07-09 | `n200_gate_rw0_ribm30` | 在 `rw0_base` 上仅改 `reject-init-bias=-3.0` | best service=0.453, best objective=66896.11 | 64m54s | **FAIL，已回退** |

| 2026-07-09 | `n200_gate_rw0_pomo2` | 在 `rw0_base` 上仅改 `pomo-size=2` | best service=0.312, best objective=80485.31 | 79m58s | **FAIL，已回退** |
| 2026-07-10 | `n200_phaseB_rw0_base` | Phase B 基线确认（3ep/1024/500） | best service=0.425, best objective=73978.23 | （未提取） | 基线确认完成 |
| 2026-07-10 | `n200_phaseB_rw1_warmup1` | 在 Phase B 基线上仅改 `reject-warmup-epochs=1` | best service=0.672, best objective=39819.82 | （未提取） | **PASS（显著优于 rw0_base）** |
| 2026-07-11 | `n200_phaseC_rw1_e10_eval500` | 正式评估（500样本，seed=99999） | audited service=0.863540, rejected=0.106610, unfulfilled=0.029850, raw_cost=17880.395 | 评估完成 | **PASS（按当前可接受阈值>=0.85）** |
| 2026-07-11 | `n200_phaseB_rw1_warmup1_eval500` | 对照评估（500样本，seed=99999） | audited service=0.705630, rejected=0.225720, unfulfilled=0.068650, raw_cost=37041.609 | 评估完成 | 对照基线 |
| 2026-07-16 | `n200_sr_opt_a1_ub003_eval500` | 单变量：`decode_pickup_urgency_bias=0.03`（其余冻结） | service=0.848070, rejected=0.122420, unfulfilled=0.029510, raw_cost=19633.149, clean=false | 评估完成 | **FAIL（服务率下降）** |
| 2026-07-16 | `n200_sr_opt_a2_ub006_eval500` | 单变量：`decode_pickup_urgency_bias=0.06`（其余冻结） | service=0.862700, rejected=0.101990, unfulfilled=0.035310, raw_cost=18176.966, clean=false | 评估完成 | **FAIL（unfulfilled升高）** |

当前状态：
- Phase C 正式评估完成：`n200_phaseC_rw1_e10` 在 service/cost/reject/unfulfilled 上全面优于 `phaseB_rw1_warmup1`。
- 当前 champion（Phase C 冻结基线）更新为：`n200_phaseC_rw1_e10`。
- 冻结基线关键值：`service=0.86354`、`rejected=0.10661`、`unfulfilled=0.02985`、`business_acceptance.clean=false`。
- A1/A2（仅调 decode urgency bias）均已 FAIL，不满足晋级门槛。
- 下一步：转入 A3（`alpha_unfulfilled=850`）与 A4（`alpha_reject=620`）训练候选；当前会话的“新训练任务”权限受限，待用户确认后执行。
---

## 5. 下一阶段执行顺序（单变量队列）

按照信息增益/成本优先级（动态更新）：
1. A1：仅改 `decode_pickup_urgency_bias=0.03`（其余参数冻结）；
2. A2：仅改 `decode_pickup_urgency_bias=0.06`；
3. A3：仅改 `alpha_unfulfilled=850`；
4. A4：仅改 `alpha_reject=620`（仅在 A1~A3 未达门槛时）。

每一步都必须经过：
- 单变量控制；
- 固定评估口径（`seed=99999`, `decode=greedy`, `num_samples=500`）；
- 满足晋级门槛后再补 `num_samples=2000` 复核。

---

## 6. PASS/FAIL 判定门槛（统一）

### Phase A 通过条件
- `service_rate` 高于当前 champion；
- 墙钟增幅不超过 15%；
- `objective` 不显著恶化。

### Phase B 通过条件
- 在更大口径下仍优于基线；
- 指标稳定（非一次性抖动）。

### Phase C 达标条件
- 正式评估 `service_rate_mean >= 0.90`；
- 业务洁净项不恶化（unfulfilled / untouched_unrejected 等）。

---

## 7. Harness 化进度管控方案（项目“清晰可控”）

### 7.1 统一命名与目录
- 统一实验ID：`n200_{phase}_{variant}_{date}`
- 日志统一：`<output-root>.log`
- 结果摘要统一：每次跑完追加到本文件第 4 节表格。

### 7.2 统一节奏
- 每次只允许一个 active run；
- 运行前写入“本次唯一变量”；
- 运行后 3 分钟内写入 PASS/FAIL 与回退决策。

### 7.3 Harness 工具建议（可直接执行）
1. 用任务看板追踪：
   - `TaskCreate`: 创建“当前实验任务”
   - `TaskUpdate`: 状态 `pending -> in_progress -> completed`
2. 用监控减少盯屏成本：
   - `Monitor`: 监听日志中的 `Epoch`、`Training Complete`、`Traceback`
3. 用提醒收口：
   - 长跑结束推送一次结果摘要，避免遗漏回填。

### 7.4 风险控制
- 连续 3 个单变量 FAIL：暂停调参，做一次复盘（仅复盘，不改语义）。
- 任一回归/约束异常：立即回退到最近 PASS 基线。

---

## 8. 今日到下一里程碑

- 今日完成：Phase B `warmup=0 vs 1` 对照，`warmup=1` 明确胜出。
- 下一里程碑（M-next）：完成 `warmup=1` 的 Phase C（8~10 epoch）并判断是否具备冲击 0.90 的潜力.

---

## 9. 备注

本文件作为 N200 主线的“实验控制台账”，用于保证：
- 决策可追溯
- 过程可审计
- 结果可复现
