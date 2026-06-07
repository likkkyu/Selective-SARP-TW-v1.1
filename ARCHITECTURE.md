# Selective SARP-TW 代码架构文档（v1.1）

> 当前仓库仍沿用 `project-vrp-v6-4-main` 的历史目录与若干类名，
> 但训练核心已经按 Selective SARP-TW v1.1 的业务口径和三项网络改造完成更新。

---

## 1. 整体技术栈

| 层级 | 技术选型 | 说明 |
|---|---|---|
| 模型 | PyTorch 2.x + Attention Model (Kool 风格) | encoder-decoder + 多头注意力 |
| 训练算法 | REINFORCE + 共享均值基线 (POMO-style) | 多 rollout，但非强制 N-start |
| 数据 | `MCVRPPDTWDataset` 在线生成 | 50% 均匀 + 50% 聚类 |
| 严格基线 | Gurobi MILP | 仍暂留，当前阶段不作为主改造对象 |
| 启发式基线 | SA / GA | 暂留 |
| 统一评估 | `MCVRPPDTW.get_costs(return_details=True)` | DRL / baseline 共用一套成本口径 |

---

## 2. 模块依赖图

```text
problem_mcvrptw_v2.py
├── Config                         # v1.1 参数真值来源
├── MCVRPPDTWDataset               # 在线数据生成 + pd_pair_mask
├── MCVRPPDTW.get_costs            # 训练目标 / 报告目标统一入口
│   ├── _compute_distance_energy
│   ├── _compute_time_and_delay    # 乘车时长统计已向量化
│   └── _compute_vehicle_and_penalty
└── calibrate_normalization

state_mcvrptw_v2.py
├── StateMCVRPPDTW.initialize
├── StateMCVRPPDTW.get_mask        # 5 层硬约束 + reject 动作 mask
└── StateMCVRPPDTW.update          # 包含 reject 分支和永久屏蔽

nets/graph_encoder.py
├── MultiHeadAttention             # 新增 pd_bias
├── MultiHeadAttentionLayer        # 支持 pd_pair_mask 透传
└── GraphAttentionEncoder

nets/attention_model.py
├── 类型嵌入 type_embedding + numeric_proj
├── reject_proj                    # 动作空间 N+1 -> N+2
├── _build_pd_pair_mask
└── AttentionModel.forward/_inner

run_training_optimized.py
├── POMOTrainerOptimized
├── _pomo_forward                  # 修复为真正多 rollout 张量
├── validate
└── checkpoint / log
```

---

## 3. 当前训练主路径

### 3.1 数据流

1. `MCVRPPDTWDataset.__getitem__()` 生成：
   - `depot`
   - `loc`
   - `node_type`
   - `demand_passenger`
   - `demand_cargo`
   - `time_windows`
   - `pickup_delivery_pairs`
   - `pd_pair_mask`

2. `AttentionModel._init_embed()`
   - 连续特征经 `numeric_proj`
   - 离散节点类型经 `type_embedding`
   - depot 走 `init_embed_depot + type_embedding(0)`

3. `GraphAttentionEncoder.forward(..., pd_pair_mask=...)`
   - 多层 encoder 计算节点 embedding
   - pickup-delivery 对通过 `pd_bias` 获得额外 compatibility bias

4. `AttentionModel._inner()`
   - `StateMCVRPPDTW.initialize()` 建状态
   - 每一步调用 `get_mask()` 得可行动作
   - 动作空间为：`depot + 真实节点 + reject`

5. `StateMCVRPPDTW.update()`
   - 普通节点：更新时间、容量、pickup/delivery 状态
   - reject 动作：选择 tw_late 最紧迫的可达未服务 pickup，永久屏蔽对应订单

6. `MCVRPPDTW.get_costs()`
   - 输出训练目标 `objective_total`
   - 输出报告目标 `total_cost_raw`

---

## 4. Config 当前关键字段

```python
class Config:
    OPERATION_START = 10.0
    OPERATION_END   = 16.0
    VEHICLE_SPEED   = 25.0
    AREA_SIZE       = 10.0
    SERVICE_TIME    = 3 / 60
    MAX_TRIP_TIME   = 3.0

    PASSENGER_CAPACITY = 15
    CARGO_CAPACITY     = 20
    PASSENGER_WEIGHT_KG  = 65.0
    CARGO_UNIT_WEIGHT_KG = 1.0

    PASSENGER_TW_WIDTH = 0.5
    CARGO_TW_WIDTH     = 1.0
    PASSENGER_RATIO    = 0.6
    PASSENGER_RATIO_LARGE = 0.6

    PASSENGER_MAX_RIDE_TIME_MINUTES = 70.0
    PASSENGER_MAX_EXCESS_RIDE_TIME_MINUTES = 30.0

    DEFAULT_NUM_VEHICLE_RATIO = 1 / 6
    VEHICLE_COST = 20.0
    ALPHA_REJECT = 500.0
    ALPHA_UNFULFILLED = 600.0
    ALPHA_TRIP_OVERTIME = 200.0
```

> 注：`VEHICLE_COST=20` 与 `DEFAULT_NUM_VEHICLE_RATIO=1/6` 是当前代码口径；若实验设计后续调整，需同步所有文档。

---

## 5. 三项网络改造落点

### 5.1 类型嵌入 (X)

文件：`nets/attention_model.py`

当前实现：

- `type_embedding = nn.Embedding(5, embedding_dim)`
- `numeric_proj = nn.Linear(6, embedding_dim)`
- 5 类 type id：
  - 0 depot
  - 1 passenger pickup
  - 2 passenger delivery
  - 3 cargo pickup
  - 4 cargo delivery

作用：让模型能显式区分 5 类节点的离散语义，而不是把 `node_type` 当作连续标量。

### 5.2 PD 对感知注意力 bias (A)

文件：`nets/graph_encoder.py`

当前实现：

- `MultiHeadAttention` 新增 `pd_bias`
- `GraphAttentionEncoder.forward(..., pd_pair_mask=...)`
- `MultiHeadAttentionLayer` 不再用 `nn.Sequential`，改成显式 `forward`，支持透传附加参数

作用：让 encoder 在编码阶段显式偏向同订单 pickup-delivery 对之间的信息交互。

### 5.3 可学习 reject 动作 (甲)

文件：
- `nets/attention_model.py`
- `state_mcvrptw_v2.py`

当前实现：

- `reject_proj` 生成独立 reject logit
- decoder logits 从 `N+1` 扩展到 `N+2`
- `StateMCVRPPDTW` 新增：
  - `rejected_`
  - `reject_count`
  - `reject_index`
- `get_mask()` 会决定 reject 动作是否可用
- `update()` 中 reject 分支会：
  - 选择当前 mask 下可达、未服务、tw_late 最紧迫的 pickup
  - 永久屏蔽该订单 pickup / delivery
  - 记录 reject_count
- 时间推进仍采用标准等待机制：`start_service = max(arrival, earliest)`，不额外建模 depot 端延迟出发

---

## 6. 训练器状态

### 6.1 当前 POMO 逻辑

`run_training_optimized.py` 已修复之前的“单 rollout 假 POMO”问题：

- `pomo_size > 1` 时复制 batch 并并行前向
- `_pomo_forward()` 返回 `costs (B, pomo)` 与 `log_probs (B, pomo)`
- `_pomo_loss()` 对同一个 instance 的多个 rollout 取共享 baseline

当前属于：
- **可工作版 POMO-style 多 rollout**
- 但**并非强制不同首步**的完全经典 POMO

### 6.2 validate 口径

`validate()` 当前聚合：
- energy / passenger delay / cargo delay
- trip overtime
- reject penalty / unfulfilled penalty
- completed / rejected / unfulfilled / used vehicles
- passenger pickup hard violations
- passenger ride-time violations

不再保留旧的 `service_shortfall_penalty`。

---

## 7. 已删除 / 已清理模块

本轮整改已删除：

- `nets/critic_network.py`
- `nets/pointer_network.py`
- `utils/boolmask.py`
- `utils/data_utils.py`
- `utils/log_utils.py`

并清理了：
- `utils/functions.py` 中旧的多问题 loader / pointer loader 路径

保守保留：
- `utils/beam_search.py`
- `utils/lexsort.py`
- `utils/tensor_functions.py`
- `sample_many` 链

因为它们虽非训练主路径核心，但仍可能被 beam-search / 辅助路径依赖。

---

## 8. 当前文档与代码的一致性说明

下面这些点以代码为准，并已在本文档同步：

| 项 | 当前值 |
|---|---|
| Passenger Capacity | 15 |
| Cargo Capacity | 20 |
| Passenger TW Width | 0.5 h |
| Cargo TW Width | 1.0 h |
| Service Time | 3 min |
| Passenger Ratio | 0.6 |
| Vehicle Cost | 20 |
| Vehicle Ratio | 1/6 |
| Passenger Max Ride Time | 70 min |
| Passenger Excess Ride Time | 30 min |
| Reject Penalty | 500 |
| Unfulfilled Penalty | 600 |

---

## 9. 仍然值得继续关注的点

1. 当前 reject 动作已经打通，但还需要正式训练验证其学习效果。
2. 当前 POMO 是“多 rollout 共享 baseline”版本，不是强制 N-start 版本。
3. `attention_model.py` 中仍保留部分原始 Kool 多问题死分支，尚未做最后一轮大清理。
4. baseline / 可视化 / 论文脚本虽然大部分能继续运行，但口径与文档仍建议后续再专项同步一次。

---

## 10. 建议执行顺序

如果继续推进实验，建议：

1. 先跑 `compileall + dry_run + 小 batch forward`
2. 再跑一个最小训练 smoke（尤其验证 reject 动作 / pd_bias / type embedding 不破坏 backward）
3. 然后执行 `calibrate_normalization`
4. 最后正式进入 25 → 50 → 100 的课程训练
