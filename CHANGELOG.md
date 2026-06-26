# Selective SARP-TW / v6 代码整改与 v1.1 改造日志

> 本文件不再按旧 v5/v6 评审轮次堆叠记录，而是聚焦当前代码库已经完成的整改与网络改造结果。
> 如旧文档与当前实现冲突，以代码为准。

---

## 2026-06-25：checkpoint 业务验收口径更新

### 新增：formal eval JSON 摘要输出

`evaluate_model.py` 新增可选 JSON 输出：

- `--json-output <path>`

该摘要复用现有 replay-audit 结果，显式写出：

- aggregate 指标
- audited 指标
- `partition_consistent_samples`
- `core_aggregate_match_samples`
- `business_acceptance`

其中 `business_acceptance.clean` 的判定口径是：

- audited `unfulfilled == 0`
- audited `pickup_only == 0`
- audited `started_not_completed == 0`
- audited `untouched_unrejected == 0`

这样可以对多个已保存 checkpoint 做机器可读的正式验收比较，而不必只靠控制台日志人工抄表。

### 更新：默认 best checkpoint 改为 business-first

`run_training_optimized.py` 中默认 `model_best.pt` 的选择规则已从 service-first 改为 business-first：

1. 先满足业务硬门槛：
   - `avg_unfulfilled_orders == 0`
   - `avg_pickup_only_orders == 0`
   - `avg_started_not_completed_orders == 0`
   - `avg_untouched_unrejected_orders == 0`
2. 在 clean checkpoint 中，再比较：
   - 更高 `service_rate`
   - 更低 `avg_objective`

同时保留：

- `model_best_service.pt`：旧 service-priority 对照口径
- `model_best_objective.pt`
- `model_final.pt`

### 说明：关于“clean checkpoint”的稳定性

单个 clean checkpoint 只说明训练方向**能够到达**业务可接受点，
不说明该方向已经稳定收敛到“持续零 residual buckets”。

因此后续正式验收仍应采用：

- 先对已保存 checkpoint 做 formal eval
- 再基于更长训练 / 多 seed 判断是否稳定可复现

## 2026-06-07：v1.1 整改第一阶段（第 11 章执行）

### 1.1 低风险废弃代码清理

已删除：

- `nets/critic_network.py`
- `nets/pointer_network.py`
- `utils/boolmask.py`
- `utils/data_utils.py`
- `utils/log_utils.py`

已清理：

- `utils/functions.py` 中旧的 `load_problem` / `load_model` 多问题模板加载链
- `utils/__init__.py` 中对旧 helper 的通配导出

保守保留：

- `utils/beam_search.py`
- `utils/lexsort.py`
- `utils/tensor_functions.py`
- `utils.functions.sample_many`

原因：beam-search 链虽然不是训练主路径核心，但尚未完全确认没有辅助路径依赖，暂不激进删除。

### 1.2 删除死惩罚链

已删除 `service_shortfall_penalty` 整链：

- `problem_mcvrptw_v2.py`
- `run_training_optimized.py`
- `evaluate_model.py`

原因：该值曾被计算，但从未进入训练目标，只是残留在 `details` / validate / 打印中，属于死代码。

### 1.3 修复失效 POMO

`run_training_optimized.py` 已修复原先“只跑单 rollout”的伪 POMO 实现：

- `_pomo_forward()` 现在在 `pomo_size > 1` 时复制 batch 并并行 rollout
- `_pomo_loss()` 终于能真实接收 `costs (B, pomo)`
- 同一 instance 的多个 rollout 使用共享均值 baseline

当前实现是：
- **可工作版 POMO-style 多 rollout**
- 不是“强制不同首步”的经典 N-start POMO

### 1.4 热点向量化

`problem_mcvrptw_v2._compute_time_and_delay()` 已把原先最重的：

- `for b in range(batch_size)`
- `.item()`
- GPU / CPU 同步式乘车时长统计

改为张量化实现：

- 记录 passenger pickup finish time
- delivery 侧通过 `gather` 整批取 pickup 完成时刻
- 整批计算 ride time / excess ride time / violation

该项是当前性能整改中的核心收益点之一。

### 1.5 Config 口径更新到 v1.1

当前代码值已调整为：

| 字段 | 当前值 |
|---|---|
| `PASSENGER_CAPACITY` | 15 |
| `CARGO_CAPACITY` | 20 |
| `PASSENGER_TW_WIDTH` | 1.0 |
| `CARGO_TW_WIDTH` | 1.0 |
| `SERVICE_TIME` | 3/60 |
| `PASSENGER_RATIO` | 0.6 |
| `PASSENGER_RATIO_LARGE` | 0.6 |
| passenger 大组 | {3,4} |
| cargo demand | 1–3 |

本轮有意保留：

- `VEHICLE_COST = 20`
- `DEFAULT_NUM_VEHICLE_RATIO = 1/6`

原因：这两项更像实验设计决策，不应在未确认前被文档强行覆盖。

### 1.6 被动 reject 旧口径移除

已删除旧的：

- `REJECT_TIME`
- `REJECT_DELAY_HOURS`

当前 reject 口径已进一步拆分为：

- `reject_penalty = rejected_orders * ALPHA_REJECT`（主动 reject）
- `unfulfilled_penalty = unfulfilled_orders * ALPHA_UNFULFILLED`（未显式 reject 但最终未完成）

这样既保留“主动拒单”的干净语义，也避免模型通过静默漏单逃逸。

---

## 2026-06-07：v1.1 改造第二阶段（第 4 章三项网络改造）

### 2.1 类型嵌入 (X) 已实现

文件：`nets/attention_model.py`

主要改动：

- 新增 `type_embedding = nn.Embedding(5, embedding_dim)`
- 新增 `numeric_proj = nn.Linear(6, embedding_dim)`
- depot 使用 `init_embed_depot + type_embedding(0)`
- 节点类型映射为 5 类：
  - 0 depot
  - 1 passenger pickup
  - 2 passenger delivery
  - 3 cargo pickup
  - 4 cargo delivery

含义：`node_type` 不再只是连续标量，而是离散语义 embedding 与连续特征投影相加。

### 2.2 PD 对感知注意力 bias 已实现

文件：`nets/graph_encoder.py`

主要改动：

- `MultiHeadAttention` 新增 `pd_bias`
- `forward(..., pd_pair_mask=None)` 支持 pickup-delivery 对 bias
- `MultiHeadAttentionLayer` 从 `nn.Sequential` 改成显式模块容器
- `GraphAttentionEncoder.forward()` 支持透传 `pd_pair_mask`

数据侧联动：

- `MCVRPPDTWDataset` 现在生成 `pd_pair_mask`
- `AttentionModel._build_pd_pair_mask()` 优先读取 batch 中的 `pd_pair_mask`

### 2.3 可学习 reject 动作 已实现

涉及文件：

- `nets/attention_model.py`
- `state_mcvrptw_v2.py`
- `problem_mcvrptw_v2.py`

主要改动：

- 新增 `reject_proj`
- decoder logits 从 `N+1` 扩展到 `N+2`
- state 中新增：
  - `rejected_`
  - `reject_count`
  - `reject_index`
- `get_mask()` 为 reject 动作提供独立可行性判断
- `update()` 新增 reject 分支：
  - 在可达、未服务、未被拒的 pickup 中选择 `tw_late` 最紧迫者
  - 永久屏蔽对应 pickup / delivery
  - 记录 reject_count
- `_compute_vehicle_and_penalty()` 已能消费 `reject_count`

说明：
当前实现已经把 reject 动作打通到网络 / state / 成本路径，但是否能在训练中学出理想策略，还需要训练级 smoke 和正式实验验证。

---

## 2026-06-07：文档同步

已重写或同步：

- `MODELING.md`
- `ARCHITECTURE.md`

同步内容包括：

- v1.1 配置口径
- `VEHICLE_COST=20`
- `DEFAULT_NUM_VEHICLE_RATIO=1/6`
- passenger ride time = 70 / 30
- 类型嵌入 / pd_bias / reject 动作三项改造
- 当前 POMO 为多 rollout 共享 baseline 版本
- 当前等待语义采用标准 `start_service = max(arrival, earliest)`，不单独建模 depot 端延迟出发
- 当前成本口径已拆分主动 reject 与 unfulfilled penalty

---

## 当前代码状态总结

当前训练核心链路：

- `problem_mcvrptw_v2.py`
- `state_mcvrptw_v2.py`
- `nets/attention_model.py`
- `nets/graph_encoder.py`
- `run_training_optimized.py`

已经完成：

1. 第 11 章整改
2. 第 4 章三项网络改造编码
3. compileall / dry-run / forward smoke 基本验证

尚建议继续做：

1. 最小训练 smoke（尤其验证 reject 动作、pd_bias、type embedding 下的 backward）
2. normalization recalibration
3. 正式 25 → 50 → 100 课程训练
4. 若需要，再做 `attention_model.py` 原始 Kool 死分支的最终清理
