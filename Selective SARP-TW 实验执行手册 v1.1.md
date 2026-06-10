

# Selective SARP-TW 实验执行手册 v1.1

> 📌 **本手册定位**：面向自用的**工程参考手册**。基线代码 = `project-vrp-v6-4`（下称 v6）。v6 仅以 POMO 构思，网络侧缺少**类型嵌入(a)、PD 对感知注意力(c)、可学习拒单动作(e)**三项；且实测发现“POMO 多起点其实未生效”。本手册逐项给出改造方案、代码位置与改动注意，并在**第 11 章**给出可直接交给本地 Claude Code 执行的整改清单。仅做 DRL 本身（规模 25/50/100，纯合成数据），对比实验（ALNS/Gurobi/GA/SA）后续另行处理。  
> **口径警示**：v6 中部分参数与 MODELING.md / 注释 / docstring 不一致，且存在废弃代码与失效逻辑。凡需核对/修正处本手册以 ⚠️ 标注，一律以**实际代码**为准。

# 第 0 章 · TL;DR 与一个月执行节奏

**一句话**：先按第 11 章整改清单清理 v6（删废弃代码 + 修失效 POMO + 向量化瓶颈 + 改口径），再补齐三项网络改造（类型嵌入 / PD 对 bias / 可学习拒单），跑归一化校准，最后按 25→50→100 课程、3 种子训练并报告。

## 0.1 必做事项核对清单

- **(P0) 修复失效的 POMO 多起点**：当前 `_pomo_forward` 只跑单 rollout，`pomo_size` 形同虚设（见 6.4 / 11.2）。
- **(X) 类型嵌入**：5 类节点用 `nn.Embedding` 加到初始嵌入。文件 `nets/attention_model.py · _init_embed`。
- **(A) PD 对感知注意力**：同订单 pickup↔delivery 在 MHA compatibility 上加可学习标量 bias。文件 `nets/graph_encoder.py · MultiHeadAttention.forward`。
- **(甲) 可学习拒单动作**：解码器新增独立 reject_logit 通道，N+1→N+2；确定性放弃“可达且 tw 上界最紧迫的未服务 pickup 及其 delivery”。文件 `attention_model.py` + `state_mcvrptw_v2.py`。
- **归一化校准**：`NORMALIZATION_PROFILES` 三档全为占位 1.0，训练前必须跑 `calibrate_normalization`（256 样本）。
- **口径与废弃代码**：见第 6、11 章逐项处理。

## 0.2 一个月节奏（2026-06-07 起）

| 周 | 目标 | 交付物 |
|---|---|---|
| 第 1 周 | 本地 Claude Code 执行第 11 章整改（清废弃 + 修 POMO + 向量化 + 改口径）+ 三项网络改造编码 + 小 batch 自测 | 清理后可训练代码、25 单 smoke 曲线 |
| 第 2 周 | 归一化校准 + 25/50 课程训练（3 种子） | 25/50 checkpoint、归一化 profile JSON、收敛曲线 |
| 第 3 周 | 100 单课程训练（3 种子）+ 消融（类型嵌入/PD bias/reject 各关一项） | 100 checkpoint、消融对照表 |
| 第 4 周 | 全规模评测 + 结果整理（成本分项、拒单率、约束违反率）、补跑回归 | 评测报告原始数据（纯 RMB 口径） |

# 第 1 章 · 问题定义与 v1.1 业务参数

## 1.1 Selective SARP-TW 问题描述

非高峰时段（10:00–16:00）需求响应公交（DRT）的客货共享：同一车队在带时间窗的取送网络上，同时服务乘客与货物订单。车辆为双隔间（乘客舱 + 货物舱），需满足取送先后、时间窗、单趟时长、最大乘车时长等约束。“Selective”体现为模型可**主动拒单**，在服务收益与运营成本间权衡。建模为 MCVRP-PDTW（多隔间 + 取送 + 时间窗）。

## 1.2 v1.1 七项业务约束（冻结）

| # | 约束 | v1.1 取值 |
|---|---|---|
| 1 | 时间窗宽度 | 乘客 TW = 30 min；货物 TW = 1 h |
| 2 | 双隔间容量 | 15 名乘客 + 20 货物单位 |
| 3 | 单趟时长上限 | ≤ 3 h |
| 4 | 每节点服务时长 | 3 min |
| 5 | 提前到达 | 提前 ≤ 20 min（允许等待）；当前实现采用标准等待机制，不单独建模 depot 端延迟出发 |
| 6 | depot 发车 | 任意时刻可发车；delivery 的 tw_e（上界）不强制 |
| 7 | 需求/订单分布 | 乘客:货物 = 60:40；乘客需求 80% 取 {1,2} / 20% 取 {3,4}；货物 1–3 单位；乘客绕行惩罚高于货物 |

> 💡 **v1.1 业务约束与 v6 现状差异（必须改 Config）**：容量 v6=20 人/50 kg（改 15 人/20 单位）；乘客 TW v6=1 h（改 30 min）；货物 TW v6=2 h（改 1 h）；服务时长 v6=5 min（改 3 min）；乘客需求大组 v6={3,4,5}（改 {3,4}）；货物 v6=1–5（改 1–3）；订单比例 v6 `PASSENGER_RATIO=0.4` 即客 40%（改客 60%）。详见第 9 章速查表“v1.1 目标值”列。

## 1.3 业务约束 → 代码落点映射

| 业务约束 | 数学体现 | 代码位置 / 开关 |
|---|---|---|
| 乘客 pickup 硬时间窗 | 到达 ≤ tw_late | `state.get_mask` passenger_pickup_mask；`HARD_PASSENGER_PICKUP_TIMEWINDOW` |
| 最大乘车时长 | 总 ≤ 70 min 且超额 ≤ 30 min | `get_mask` ride_time_mask；`PASSENGER_MAX_RIDE_TIME_MINUTES=70` / `..EXCESS..=30` |
| 单趟 ≤ 3 h | trip_time ≤ MAX_TRIP_TIME | `get_mask` trip_mask；`HARD_MAX_TRIP_TIME` / `MAX_TRIP_TIME=3.0` |
| 运营 16:00 截止 | 到达 ≤ OPERATION_END | `get_mask` ops_end_mask；`HARD_OPERATION_END` |
| 双隔间容量 | 剩余容量 ≥ 需求 | `get_mask` cap_mask_p / cap_mask_c |
| 取送先后 | delivery 不先于 pickup | `get_mask` PD precedence（向量化） |
| 车队上限 K_max | 已用车 ≤ K_max | `get_mask` K_max block；`get_default_num_vehicles` |
| 拒单（选择性） | 主动放弃 → 拒单惩罚 | v6：被动惩罚（`_compute_vehicle_and_penalty`）；v1.1：改可学习动作（4.4） |

# 第 2 章 · 数据生成

## 2.1 数据集类与生成入口

核心文件 `problem_mcvrptw_v2.py`。数据集类 `MCVRPPDTWDataset`（788 行起），生成入口 `_generate_data`（860–982），时间窗 `_generate_time_windows`（984–1050），位置 `_generate_locations`（827–858）。所有参数集中在 `Config`（43–165），是**唯一参数源**。

> 💡 **docstring 失真**：`MCVRPPDTWDataset` 类 docstring（796–797 行）写“乘客时间窗 0.6 小时、货物 1.2 小时、需求 1–5”，与 Config 实际值（1.0 h / 2.0 h）**不符**，属遗留注释。整改时一并更新（见 11.4）。

## 2.2 节点编码与特征向量（node_dim=7）

节点特征 `node_dim=7`（`attention_model.py` 91–92），由 `_init_embed`（216–231）拼接：`[loc_x, loc_y, node_type, demand_passenger, demand_cargo, tw_early, tw_late]`。5 类节点编号：0=depot，1=pax_pickup，2=pax_delivery，3=frt_pickup，4=frt_delivery。

> 💡 **关键发现**：v6 中 `node_type` 仅作为**数值标量**经 `nn.Linear(node_dim, embedding_dim)` 映射 —— 无 `nn.Embedding`，模型无法学到 5 类节点的离散语义。这是改造 (X) 要解决的（见 4.2）。

## 2.3 需求 / 时间窗 / 空间分布采样（v6 现状）

- **乘客需求**（939–950 行）：80% 小组 `randint(1,3)`={1,2}；20% 大组 `randint(3,6)`={3,4,5}。
- **货物需求**：`randint(1,6)`={1..5}。
- **订单比例**：`PASSENGER_RATIO=0.4`（25 单主线客 40%）；n≥50 用 `PASSENGER_RATIO_LARGE=0.25`（874 行）。
- **时间窗**：三时段 [10,12) / [12,14) / [14,16) 混合采样；当前默认权重为乘客 `(0.56, 0.29, 0.15)`、货物 `(0.58, 0.28, 0.14)`；乘客 TW=1 h、货物=2 h。
- **空间**：50% 随机 + 50% 聚类（`NUM_CLUSTERS=3`）；PD 距离分档（短/中/长）。

## 2.4 v1.1 需要改动的数据生成项

| 项 | v6 | v1.1 | 改动点（行号） |
|---|---|---|---|
| 乘客大组 | {3,4,5} | {3,4} | `randint(3,6)`→`randint(3,5)`（950） |
| 货物需求 | 1–5 | 1–3 | `randint(1,6)`→`randint(1,4)`（939，且乘客被覆盖） |
| 乘客占比 | 0.4/0.25 | 0.6 | `PASSENGER_RATIO=0.6`，复核 LARGE 档（99–100） |
| 乘客 TW 宽 | 1.0 h | 1.0 h | `PASSENGER_TW_WIDTH=1.0`（58） |
| 货物 TW 宽 | 2.0 h | 1.0 h | `CARGO_TW_WIDTH=1.0`（58） |

# 第 3 章 · RL 环境与 5 层 mask

## 3.1 状态表示 StateMCVRPPDTW

文件 `state_mcvrptw_v2.py`（471 行），`NamedTuple` 维护：coords / node_type / demand / time_windows（固定输入）、current_time / trip_start_time / used_capacity_passenger / used_capacity_cargo / used_vehicles / visited_ / picked_up_ / passenger_pickup_time / deadlock_count / terminal_ / i（动态）。容量在 state 内归一化为 1.0（56–57 行），其余常量从 Config 读取（58–63）。

## 3.2 五层 mask 逐层拆解（get_mask）

| 层 | 含义 | 实现 |
|---|---|---|
| 1 visited | 已访问不可重复 | visited 布尔索引 |
| 2 PD precedence | delivery 不先于 pickup | 向量化屏蔽未取订单的 delivery |
| 3 dual capacity | 双隔间剩余 ≥ 待取需求 | cap_mask_p + cap_mask_c |
| 4a pax pickup TW | 乘客 pickup 到达 ≤ tw_late | passenger_pickup_mask |
| 4a ride time | 总 ≤70min 且超额 ≤30min | ride_time_mask（双层） |
| 4b trip | 单趟 ≤3 h | trip_mask |
| 4c ops end | 到达 ≤16:00 | ops_end_mask |
| 5 misc | 禁带载回 depot；F3 depot 空闲屏蔽；K_max 屏蔽；死锁兜底 | no-carry / K_max / deadlock fallback |

> 💡 encoder 一侧**不做**节点 mask（`graph_encoder.py` 204–211 显式 `NotImplementedError`），所有 masking 集中在解码时 `get_mask()`。改造 PD bias 时需新增**独立** pd_pair_mask 通道，不要复用被禁用的 node mask（见 4.3）。

## 3.3 状态转移 update()

关键逻辑：① depot 重置（回 depot 时 time→OPERATION_START、容量重置、trip_start 重置）；② 服务时刻 `start_service = max(new_time, earliest)`；③ deadlock_count / terminal 终止判定。当前实现采用标准等待机制，不额外引入 depot 端延迟出发决策。

# 第 4 章 · 网络架构：v6 现状 + 三项必做改造

## 4.1 v6 现状：Kool 注意力编解码器

v6 = Kool 2019 标准 `GraphAttentionEncoder` + `AttentionModel` 解码器。**注意**：`attention_model.py` 保留了大量原版 Kool 的**多问题分支**（TSP/CVRP/SDVRP/OP/PCTSP），本项目（mcvrppdtw）**从不走**这些分支 —— 属可清理的死分支（见 11.3）。三项组件现状：

| 组件 | v6 状态 | 动作 |
|---|---|---|
| (d) 解码器扩展上下文 | ✅ 已有（`embedding_dim+4`：cur_time / 剩余乘客容量 / 剩余货物容量 / 剩余车辆预算） | 保留 |
| (a/X) 类型嵌入 | ❌ 缺失（node_type 仅数值特征） | **必做** |
| (c/A) PD 对感知注意力 | ❌ 缺失（原版 Kool MHA，无 edge bias） | **必做** |
| (e/甲) 可学习拒单动作 | ❌ 缺失（仅事后被动惩罚） | **必做** |

(d) 见 `attention_model.py` 88–90 行 `step_context_dim = embedding_dim + 4` 及 `_get_parallel_step_context`（439–473）。

## 4.2 改造 (X) 类型嵌入

**方案**：为 5 类节点建 `nn.Embedding(5, embedding_dim)`，与连续特征线性投影**相加**。

```python
# __init__ 内（替换原 self.init_embed = nn.Linear(node_dim, embedding_dim)）:
self.type_embedding = nn.Embedding(5, embedding_dim)          # 0..4
self.numeric_proj = nn.Linear(node_dim - 1, embedding_dim)    # 去掉 node_type 标量

# _init_embed (is_mvrptw 分支):
node_type = input['node_type'].long()                         # (batch, n_loc)
numeric = torch.cat((input['loc'], input['demand_passenger'][...,None],
                     input['demand_cargo'][...,None], input['time_windows']), -1)
h_loc = self.numeric_proj(numeric) + self.type_embedding(node_type)
# depot 用 init_embed_depot(2->dim) 并叠加 type id=0 的 embedding
```

**注意**：① depot 也分配 type id=0；② depot 走 `init_embed_depot`（2→dim）需同样叠加 `type_embedding(0)`；③ node_dim 仍为 7，但 type 维改走 embedding。

## 4.3 改造 (A) PD 对感知注意力 bias

**方案**：在 encoder `MultiHeadAttention.forward` 计算 compatibility 后，对“同订单 pickup↔delivery”位置对加**可学习标量 bias**。

```python
# __init__:
self.pd_bias = nn.Parameter(torch.zeros(1))

# forward: compatibility (n_heads, batch, n_query, graph_size)
compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))
if pd_pair_mask is not None:                # (batch, n_query, graph_size) 布尔
    compatibility = compatibility + self.pd_bias * pd_pair_mask.unsqueeze(0)
```

> 💡 **改动联动**：① encoder 当前不接收任何 mask（`NotImplementedError`），需把 `pd_pair_mask` 作为**独立参数**沿 `GraphAttentionEncoder.forward → MultiHeadAttentionLayer → MultiHeadAttention` 透传；② 因 `MultiHeadAttentionLayer` 继承 `nn.Sequential`，透传额外参数需改为**显式 forward 或自定义容器**（否则 Sequential 只传单一张量）；③ `pd_pair_mask` 由数据生成阶段的取送配对一次性构造（`pickup_delivery_pairs` 已在数据里），随 batch 传入。

## 4.4 改造 (甲) 可学习拒单动作

**方案（甲，轻量级）**：解码器在 N+1 个节点 logit 外新增 **1 个独立 reject_logit**，动作空间 N+1→N+2。选中 reject 时按确定性规则：放弃**当前 mask 下可达、tw 上界最紧迫的未服务 pickup 及其 delivery**。

**为何“甲”也能学会“每辆车放哪些订单”**：车-订单分配由 POMO 路由 + 解码器扩展上下文（剩余车辆预算信号在 (d) 中）隐式学习；reject 只负责主动放弃。二者正交 —— 甲同时覆盖“分配”与“放弃”。

```python
self.reject_proj = nn.Linear(step_context_dim, 1)            # 或接在 glimpse 后
reject_logit = self.reject_proj(context)                    # (batch, 1)
all_logits = torch.cat([node_logits, reject_logit], dim=-1) # N+1 -> N+2
# 选中 N+1 索引(reject)即在 state 中按确定性规则定位被弃订单
```

**state 侧**：`update()` 增 reject 分支 —— 当 action==reject 索引时，选当前可达未服务 pickup 中 tw_late 最小者及其 delivery，标记 rejected（后续 mask 永久屏蔽），累计拒单计数供成本。

> 💡 **口径联动**：v6 被动拒单触发（`REJECT_TIME=16:00`、`REJECT_DELAY_HOURS=1h`）与新 30min/1h TW 语义不一致。当前实现已改为“主动 reject + 未履约兜底”双口径：显式 reject 走 `ALPHA_REJECT`，未显式 reject 但最终未完成的订单走 `ALPHA_UNFULFILLED`。

## 4.5 改造汇总与改动文件清单

| 文件 | 改动 |
|---|---|
| `nets/attention_model.py` | (X) `_init_embed` 加 type_embedding+numeric_proj；(甲) reject_proj + N+2 logit + 解码循环传 reject |
| `nets/graph_encoder.py` | (A) MHA 加 pd_bias + pd_pair_mask 透传；改 Sequential 容器 |
| `state_mcvrptw_v2.py` | (甲) update() 加 reject 分支 + 永久屏蔽；get_mask() 同步 rejected；保持标准等待机制 |
| `problem_mcvrptw_v2.py` | (数据) 生成 pd_pair_mask；(口径) Config 改容量/TW/需求/比例；移除被动拒单；向量化瓶颈；删废弃代码 |
| `run_training_optimized.py` | (P0) 修复 shared-baseline 多 rollout（POMO-style，见 11.2） |

# 第 5 章 · 训练流程

## 5.1 POMO 训练器与分阶段课程

文件 `run_training_optimized.py`。`POMOTrainerOptimized` 用 AdamW + CosineAnnealingLR。课程 `PHASE_CONFIGS`（35–63 行）：

| 规模 | epochs | batch | pomo | layers/dim | lr | epoch_size |
|---|---|---|---|---|---|---|
| 25 | 50 | 8 | 2 | 6/256 | 5e-5 | 8000 |
| 50 | 60 | 8 | 2 | 6/256 | 5e-5 | 8000 |
| 100 | 80 | 4 | 1 | 6/256 | 3e-5 | 10000 |

> 💡 **P0 失效**：`_pomo_forward`（162–164）只调一次 `self.model(batch)` 然后 `cost.unsqueeze(1)`，`_pomo_loss` 因 `costs.size(1)<=1` 走**批均值 baseline** —— `pomo_size` 被解析/打印但**从不影响计算**，当前实为单 rollout REINFORCE，非真正 POMO。**train_gpu.sh 注释里的 batch=64/32/16、pomo=8/8/6 与代码完全不符**。必须修（见 11.2）。

## 5.2 归一化校准（必做前置步骤）

`NORMALIZATION_PROFILES`（134–165）三档全占位 **1.0**、`num_samples=0`。训练前必须运行 `calibrate_normalization`（`--calibrate-before-train`，256 样本）生成各分项真实尺度，写入 `outputs/normalization_profiles`。否则训练目标加权失真。

## 5.3 随机种子方案

> 📌 **推荐：每规模 3 个种子**。预算与时间不是约束（RTX PRO 6000），3 种子可报均值±标准差，满足论文严谨性；审稿需要可补到 5。每种子独立训练+独立评测，固定数据生成种子（训练 `epoch*1000`、验证 `12345`）以保证可比。

# 第 6 章 · 目标函数与口径核对

## 6.1 训练目标 vs 报告目标

- **训练目标** `total_cost`（get_costs 580–586）：归一化+α 加权（`ALPHA_ENERGY=1` / `DELAY=2.5` / `VEHICLE=3` / `REJECT=575` / `UNFULFILLED=750` / `TRIP_OVERTIME=200`）。
- **报告目标** `total_cost_raw`（588–595）：纯 RMB 求和（能耗+延误+派车+拒单+超时），论文/评测用此口径。

## 6.2 成本分项与代码位置

| 分项 | 公式/参数 | 代码位置 |
|---|---|---|
| 能耗费 | η(W)=0.18·(1+W/10000) kWh/km × 1.0 元/kWh | `_compute_distance_energy`（250–307） |
| 延误费 | 乘客 0.6、货物 0.06 元/min（仅 delivery） | `_compute_time_and_delay`（309–457） |
| 派车费 | `VEHICLE_COST` 元/车 | `_compute_vehicle_and_penalty`（459–520） |
| 拒单费 | `ALPHA_REJECT=575` | 同上；v1.1 改主动拒单 |
| 超时费 | `ALPHA_TRIP_OVERTIME=200` 元/h | `get_costs`（508–509） |

## 6.3 ⚠️ 口径问题清单

| # | 问题 | 现状 | 处理 |
|---|---|---|---|
| 1 | 废弃 service_shortfall_penalty | `get_costs` 576–578 算 `service_shortfall.pow(2)*200` 但**未**加入 total_cost（580–586），仅出现在 details/打印 | 删死代码（连带 validate 里的打印，见 11.1） |
| 2 | Python for 循环瓶颈 | `_compute_time_and_delay` 400–425（`for b in range(batch)` 带 `.item()` 强制同步）、`_compute_vehicle_and_penalty` 488、数据生成 911/956/1009 | 向量化乘车时长统计（最重），见 11.2 |
| 3 | 被动拒单与新 TW 不一致 | `REJECT_DELAY_HOURS=1h`、`REJECT_TIME=16:00` | 改主动拒单后移除/改写（4.4） |
| 4 | K_max 比例不一致 | code `DEFAULT_NUM_VEHICLE_RATIO=1/6` vs MODELING.md 注释 `1/8`（170 行注释也写 N/8） | 统一口径，且与新 15/20 容量匹配后重定 |
| 5 | 归一化全占位 1.0 | 三档 num_samples=0 | 训练前跑 calibrate（5.2） |
| 6 | VEHICLE_COST 不一致 | code=20 vs MODELING.md=5；注释称“更贴近固定派车成本” | 确认业务意图，倾向保留 20 并回写 MODELING.md |
| 7 | 注释/docstring 失真 | 数据集 docstring（796–797）写 0.6h/1.2h；`_compute_time_and_delay` docstring（315）写“45 min”实为 70 min | 更新注释与代码一致（11.4） |

## 6.4 ⚠️ POMO 失效详解

真正的 POMO 需对每个 instance 用 N 个不同起点并行 rollout，以“同 instance 多解的均值”作共享 baseline（无 critic）。v6 现状只跑单解，baseline 退化为**跨 instance 的批均值**，方差更大、收敛更慢，且未利用 POMO 的对称增广。整改方案见 11.2。

# 第 7 章 · 训练监控与收敛诊断

- **核心曲线**：训练 `objective`（下降）、报告 `total_cost_raw`（评测对照）、loss / baseline。
- **业务指标**：拒单率、约束违反率（硬 mask 下应恒 0：`passenger_pickup_hard_violations` 等）、平均用车数、乘客绕行 / 延误、货物延误。
- **诊断红旗**：① 拒单率异常高→`ALPHA_REJECT` 偏低或 mask 死锁多；② raw 不降但 objective 降→归一化/α 失衡；③ deadlock_count 升高→mask 过紧或容量/TW 过严。
- **课程切换**：25→50→100 切换时用 `--resume-path --resume-weights-only` warm-start。

# 第 8 章 · checkpoint / 复现 / RTX PRO 6000 操作

- **复现三要素**：固定 ① 训练种子 ② 数据生成种子 ③ 归一化 profile JSON；三者齐全方可复现。checkpoint 已保存 `normalization_profile` 与 `args`（`_save_model` 349–358）。
- **checkpoint**：按（规模, 种子）命名归档；保留 best + final；`--resume-path` 续训。
- **AutoDL / RTX PRO 6000**：租用后先 smoke（25 单小 batch 跑通前向反向），再正式课程；长任务后台运行+日志落盘；每档训练完先评测再切下一档。
- **显存**：96GB 下 100 单档修复 POMO 后可上调 batch 与 pomo 起点；向量化（11.2）是吞吐前提。

# 第 9 章 · Config 全字段速查表

| 字段 | v6 默认 | v1.1 目标 | 含义 |
|---|---|---|---|
| AREA_SIZE | 10.0 | 10.0 | 10×10 km |
| PASSENGER_CAPACITY | 20 | **15** | 乘客容量（人） |
| CARGO_CAPACITY | 50 | **20** | 货物容量（单位） |
| OPERATION_START/END | 10/16 | 同 | 运营时段 |
| MAX_TRIP_TIME | 3.0 | 3.0 | 单趟上限(h) |
| PASSENGER_TW_WIDTH | 1.0 | **1.0** | 乘客 TW(h) |
| CARGO_TW_WIDTH | 2.0 | **1.0** | 货物 TW(h) |
| SERVICE_TIME | 5/60 | **3/60** | 每节点服务(h) |
| VEHICLE_SPEED | 25.0 | 25.0 | km/h |
| ENERGY_BASE / WEIGHT_REF | 0.18 / 10000 | 同 | 能耗模型 |
| ELECTRICITY_PRICE | 1.0 | 1.0 | 元/kWh |
| PASSENGER/CARGO_DELAY_COST | 0.6 / 0.06 | 同 | 延误 元/min |
| VEHICLE_COST | 20.0 ⚠️ | 待确认 | 派车 元/车（MODELING=5） |
| ALPHA_ENERGY/DELAY/VEHICLE | 1/2.5/3 | 同 | 训练加权 |
| ALPHA_REJECT / ALPHA_UNFULFILLED / TRIP_OVERTIME | 575 / 750 / 200 | 同/调 | 拒单 / 未履约 / 超时惩罚 |
| REJECT_TIME / REJECT_DELAY_HOURS | 16:00 / 1.0 ⚠️ | 移除/改写 | 被动拒单触发 |
| PASSENGER_RATIO / _LARGE | 0.4 / 0.25 | **0.6** | 乘客订单占比 |
| DEMAND_MIN/MAX | 1/5 | 见 2.4 | 需求范围 |
| PASSENGER_SMALL_GROUP_RATIO | 0.8 | 0.8 | 小组占比 |
| PASSENGER_WEIGHT_KG / CARGO_UNIT_WEIGHT_KG | 65 / 1.0 | 同 | 能耗用重量 |
| PASSENGER_MAX_RIDE_TIME / EXCESS | 70 / 30 | 同 | 最大乘车(min) |
| DEFAULT_NUM_VEHICLE_RATIO | 1/6 ⚠️ | 待统一 | K_max 比例（MODELING=1/8） |
| HARD_* 开关 | 全 True | 全 True | 硬约束开关 |
| NORMALIZATION_PROFILES | 占位 1.0 ⚠️ | 校准生成 | 归一化尺度 |

# 第 10 章 · 风险与已知问题清单

- **R1 POMO 失效(P0)**：不修则“POMO”名不副实，收敛慢、方差大 —— 优先级最高。
- **R2 性能**：乘车时长统计的 `for b`+`.item()` 在 n=100 时严重拖慢（GPU/CPU 反复同步）—— 必须向量化。
- **R3 改造耦合**：`pd_pair_mask` 透传需改 Sequential 容器，易引入 forward 签名错误 —— 小 batch 先验证前向/反向。
- **R4 reject 语义**：主动 reject 与未履约兜底要拆分统计，避免把显式拒单和静默漏单混成一个指标。
- **R5 归一化**：未跑 calibrate 直接训练→目标失真、“假收敛”。
- **R6 口径漂移**：Config 与 MODELING.md / 注释多处不一致 —— 改造后以代码为准并回写文档。
- **R7 容量调小**：15/20 比 v6（20/50）更紧，K_max 与 deadlock 兜底需重评，避免大量拒单。
- **R8 业务约束验证**：当前实现采用标准等待机制；若未来要建模 depot 端延迟出发，应作为新的状态/动作设计单独论证。

# 第 11 章 · v6 代码整改清单（交本地 Claude Code 执行）

> 本章是可执行的整改任务单。**建议顺序**：11.1 删废弃（低风险，先做）→ 11.2 修失效逻辑+向量化 → 11.3 删死分支 → 11.4 改口径/注释 → 第 4 章三项网络改造。每步改完用 `dry_run_test.py` 或小 batch smoke 验证再进行下一步。

## 11.1 删除废弃代码（P1，低风险）

| 对象 | 位置 | 处理 |
|---|---|---|
| `service_shortfall_penalty` 及 target_rejected_orders | problem 576–578；details 619；validate 220/257/326 | 整链删除（未进 loss 的死惩罚） |
| `nets/critic_network.py`（CriticNetwork） | 整文件 | POMO 无 critic，全项目无引用 → 删文件 |
| `nets/pointer_network.py` | 整文件 + `utils/functions.py` 的 `load_model` 中 `'pointer'` 分支 | 仅被通用 load_model 引用，本项目不用 → 删 |
| `utils/boolmask.py`、`utils/data_utils.py`、`utils/log_utils.py` | 整文件 | 无任何引用 → 删 |
| `utils.functions.load_model / load_problem` | functions.py 18–128 | 通用多问题加载器，本项目用各脚本自带 load_model_for_size → 删（注意 `utils/__init__.py` 的 `from .functions import *`） |
| `test_dataset()` | problem 1059–1106 | 临时测试函数，可移入 dry_run_test 或删 |

> 💡 删 `utils/functions.py` 内容前，确认 `attention_model.py` 仍需要的 `sample_many` 是否在该文件 —— `sample_many` 当前在 functions.py 且被 `attention_model.sample_many` 调用，但 `sample_many` / `beam_search` / `propose_expansions` 这条链在**训练/评测主路径未使用**（仅 beam search 用）。若确认不跑 beam search，可连 `compute_in_batches` / `CachedLookup` / `sample_many` 及 `utils/beam_search.py`、`utils/lexsort.py` 一并清理。**保守做法**：先只删 critic / pointer / boolmask / data_utils / log_utils，beam 链留到确认后再删。

## 11.2 修复失效逻辑 + 向量化（P0）

- **POMO 多起点**：重写 `_pomo_forward` —— 对每个 batch 沿“起点”维复制 `pomo_size` 份（或用 N 个不同首步强制起点），并行 rollout 得 `costs (B, pomo)`；`_pomo_loss` 走 `size(1)>1` 分支（已实现共享 baseline）。需让模型 / state 支持指定 / 增广首步，或最简版：同 batch 复制 pomo 份 + 采样多解取共享均值 baseline。
- **向量化乘车时长**：把 `_compute_time_and_delay` 400–425 的 `for b in range(batch_size)` 改为张量化 —— 用 scatter 记录每个 passenger pickup 的完成时刻到 `(B, n_orders)`，delivery 步用 gather 取出，整批计算 ride_time / excess，避免逐元素 `.item()`（它强制 GPU→CPU 同步，是最大瓶颈）。
- **次要循环**：`_compute_vehicle_and_penalty` 488 的 `for order_idx` 可用 one-hot / scatter 向量化；数据生成的 Python 循环（911/956/1009）是 CPU 端一次性预生成，影响小，可选优化。

## 11.3 删除死分支（P2，清理 attention_model）

- `attention_model.py` 的 `is_vrp / is_orienteering / is_pctsp / allow_partial / TSP` 全部分支（`__init__` 86–112、`_init_embed` 232–251、`_get_parallel_step_context` 474–543、`_get_attention_node_data` 583–598）本项目**从不触发**，可大幅精简为仅 mvrptw 路径。
- **注意**：此项纯减负，收益是可读性，风险中等（改 forward 主干）。建议**最后**做，且改完跑完整 smoke。若时间紧，可仅保留不删（不影响训练）。

## 11.4 口径与注释修正（P1）

- **Config 值**：按第 9 章“v1.1 目标”列改容量 / TW / 服务时长 / 需求 / 比例。
- **被动拒单**：移除/改写 `_compute_vehicle_and_penalty` 498–505 的 reject_mask（改主动拒单后）。
- **K_max / VEHICLE_COST**：确认口径后统一 code 与 MODELING.md。
- **注释**：更新数据集 docstring（796–797）、`_compute_time_and_delay` docstring（315 “45 min”→70）、`train_gpu.sh` 头部 batch / pomo 注释（与 PHASE_CONFIGS 对齐）。

## 11.5 文件去留总表

| 文件 | 分类 | 动作 |
|---|---|---|
| `problem_mcvrptw_v2.py` / `state_mcvrptw_v2.py` / `nets/attention_model.py` / `nets/graph_encoder.py` / `run_training_optimized.py` | 训练核心 | 保留+改造 |
| `evaluate_model.py` / `dry_run_test.py` | 评测/自测 | 保留 |
| `nets/critic_network.py` / `nets/pointer_network.py` / `utils/boolmask.py` / `utils/data_utils.py` / `utils/log_utils.py` | 孤儿模块 | **删** |
| `utils/functions.py`（load_model/load_problem 部分） | 通用加载器（本项目不用） | 删相关函数，保留必要 util |
| `utils/beam_search.py` / `utils/lexsort.py` / `utils/tensor_functions.py` | beam search 链（主路径不用） | 确认不跑 beam 后删 |
| `gurobi.py` / `ga_mcvrppdtw.py` / `sa_mcvrppdtw.py` / `baseline_utils.py` / `compare_drl_vs_gurobi.py` / `evaluate_gurobi_real_cost.py` | 对比实验（后续） | 暂留，本阶段不动 |
| `visualize_routes.py` / `visualize_gantt.py` / `analyze_on_time_rate.py` / `calculate_ontime_ratio.py` / `generate_paper_figures.py` | 可视化/分析 | 暂留，训练后用 |

