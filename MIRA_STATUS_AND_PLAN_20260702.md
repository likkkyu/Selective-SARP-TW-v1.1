# 给 Mira 的阶段沟通稿（2026-07-02）

## 1) 当前现状（结论先行）

当前 v2 版本已经完成业务语义增强的核心改造，并且在 N=100 的对比中相对 v1 达到：
- 服务率上升（service_rate ↑）
- 拒单率下降（reject_rate ↓）
- 总成本下降（total_raw_cost ↓）

同时，乘客约束表现稳定（乘客延误和超乘时长整体可控），说明“优先保障 passenger 体验”的方向有效。

---

## 2) 已完成改造（代码层）

已在主干训练/评估链路中接入以下语义（train / eval / 回归测试已打通）：

1. **Cargo pickup 硬时间窗（hard TW）**
   - 货物 pickup 超窗直接 mask，不再走软惩罚兜底。

2. **Cargo delivery 分段软惩罚（piecewise delay）**
   - delivery 仍为软约束，但迟到成本按分段加重（如 0~30 / 30~60 / >60 分钟）。

3. **最少派车服务单数（min-orders-per-dispatch）机制接入**
   - 状态里引入“本次发车已完成订单数”计数；
   - 发车后未达阈值时对回 depot 做约束；
   - dead-end 情形保持 reject 可用，避免无动作可选。

4. **解码 fallback 修正**
   - all-masked 情况优先 reject，减少通过 depot fallback 绕过硬语义的概率。

5. **回归测试补齐**
   - 已新增 cargo pickup hard / piecewise delay / min-dispatch / dead-end reject 等测试用例并通过。

---

## 3) 目前最关键卡点（必须如实披露）

尽管 v2 的总体指标优于 v1，但“每车>=4单”**尚未达到全局严格硬约束**。

我们在 500 样本审计中得到：
- `total_dispatch_trips = 6932`
- `viol_dispatch_trips(<4 deliveries) = 399`
- `viol_rate_all_dispatch = 0.0576`
- `samples_with_any_violation = 399/500`

这说明：
- 局部样本中看起来满足 min-4（很多车 4~8 单），
- 但全局仍有非零违反，不能在论文里写成“严格 hard guarantee 已完全满足”。

此外还有两个工程/科研瓶颈：
1. **车辆使用数偏多**（部分路线接近“刚达 4 单就回库”）；
2. **Cargo 延误仍有长尾**（>30、>60 分钟段仍需压缩）。

---

## 4) 下一步计划（可执行版本）

### Phase A（优先级最高）：把 min-4 变成“可证明严格”

目标：在审计口径下把 `viol_rate_all_dispatch` 压到 0。

执行要点：
1. 清理所有可能绕过 min-4 的 fallback 路径（包含 state 与 decoder 两侧）。
2. 将“未达 min-4 禁止回 depot”与“不可服务则 reject”做成一致的终态策略。
3. 保留并强化审计脚本：
   - trip-level 违反数
   - sample-level 违反样本数
   - 按场景（订单规模、时间窗紧度）分层统计

**验收标准**：500 样本审计 `viol_rate_all_dispatch = 0`。

---

### Phase B：控制 vehicle/reject/delay 三者 trade-off

目标：在保持 service/reject 优势的同时，降低车辆数和高分位延误。

执行方式：
1. 做小网格搜索（推荐先 3x3 规模）：
   - `alpha_vehicle`（提高车辆成本权重）
   - `alpha_reject`（防止“省车靠拒单”）
   - cargo delay 分段系数（重点抑制 >60 分钟长尾）
2. 固定 checkpoint 选择规则（以 objective 主导，service 作为约束门槛）。
3. 用同一评估协议输出 Pareto 前沿（车数-拒单-延误）。

---

### Phase C：达到论文可发表的实证强度

至少补齐：
1. **多随机种子**（建议 3~5 seeds）并报告均值±标准差；
2. **约束合规率**（尤其 min-4）独立章节，不与主目标混写；
3. **延误分布报告**（P50/P90/P95、>30、>60 占比），不能只报均值；
4. **公平基线对比**（v1/v2 同数据同评估协议）；
5. **消融实验**：
   - 去掉 cargo pickup hard
   - 去掉 piecewise
   - 去掉 min-4
   证明每个模块贡献。

---

## 5) 当前对外表述建议（避免过度 claim）

在严格修复完成前，建议论文/汇报使用以下措辞：

- 可说：
  - “v2 在服务率、拒单率与总成本上优于 v1；”
  - “引入了更贴近业务的 hard/soft 混合建模，并显著改善了结果质量。”

- 暂不宜说：
  - “已严格保证每车>=4单（hard guarantee）。”

更准确说法：
- “当前实现已显著降低违反风险，但在全局审计中仍存在少量 min-4 违反，正在进行 strict enforcement 收敛。”

---

## 6) 里程碑与交付物（建议）

### M1（1~2 天）
- 交付 strict min-4 修复版代码
- 交付 500 样本合规审计（目标 0 违反）

### M2（2~3 天）
- 完成 vehicle/reject/delay 参数网格
- 交付 Pareto 表与推荐参数

### M3（2~3 天）
- 完成多 seed 与消融
- 输出论文结果表 + 图（主表、消融表、延误分布图）

---

## 7) 给 Mira 的一句话版本

“v2 已经在核心业务指标上全面优于 v1，但要达到论文级别的‘可严格宣称’，还需把 min-4 从‘大多数样本有效’推进到‘全局 0 违反’，并用多 seed + 消融 + 分布统计补齐证据链。”
