# Selective SARP-TW 数学建模文档 (v1.1)

> **问题全称**: Selective Shared Autonomous/Responsive Passenger-Parcel Transport with Time Windows（Selective SARP-TW）
> 
> 当前代码实现承载在 `project-vrp-v6-4-main` 上，核心问题类名仍保留为 `MCVRPPDTW`，但业务口径已按 v1.1 收敛到 selective passenger-cargo sharing 场景。

---

## 1. 问题定义与背景

某城市公交公司在 10:00-16:00 平峰时段，使用一支双隔间客货共享车队，
对当日提前预订的乘客与货物订单进行静态批调度。研究目标：
**在硬约束（容量、时间窗、PD 先序、单趟 ≤3h、运营终点 16:00、车辆数 K_max）下，以 RMB 计的总成本最小化，并允许模型主动拒单。**

Selective 的含义是：模型不再被动等待订单在后处理阶段“被拒”，而是在解码阶段显式拥有 reject 动作，在服务收益与运营成本之间权衡。

---

## 2. v1.1 物理与业务参数（代码口径）

### 2.1 固定物理参数

| 符号 | 数值 | 含义 |
|---|---|---|
| $T_{start}, T_{end}$ | 10:00, 16:00 | 运营时段（硬） |
| $v$ | 25 km/h | 车辆均速 |
| $L$ | 10 km × 10 km | 服务区面积 |
| $\tau_{srv}$ | 3 min | 单点服务时间 |
| $\tau_{trip}^{max}$ | 3 h | 单次连续行程时长上限 |
| $Q^P$ | 15 人 | 单车乘客舱容量 |
| $Q^C$ | 20 单位 | 单车货物舱容量 |
| $w_p$ | 65 kg | 单乘客折算重量（能耗） |
| $w_c$ | 1 kg | 单位货物重量 |
| $K_{max}$ | $\lceil N/6 \rceil$ | 当前代码中的硬车辆上限 |

### 2.2 时间窗与需求分布

| 项 | v1.1 值 | 说明 |
|---|---|---|
| Passenger pickup TW 宽度 | 1 h | 硬时间窗 |
| Cargo pickup TW 宽度 | 1 h | delivery 仍只计软迟到 |
| Passenger : Cargo | 60 : 40 | 数据生成比例 |
| Passenger demand | 80% 属于 {1,2}；20% 属于 {3,4} | group size |
| Cargo demand | 1–3 单位 | 单位需求 |
| 早到处理 | 允许等待，零罚 | 当前采用标准等待机制，不显式建模 depot 端延迟出发 |

### 2.3 单价 / 罚款

| 符号 | 数值 | 含义 |
|---|---|---|
| $\omega_p$ | 0.6 元/min | Passenger delivery 延误单价 |
| $\omega_c$ | 0.06 元/min | Cargo delivery 延误单价 |
| $\omega_v$ | 20 元/车 | 固定派车成本（当前代码保留 20） |
| $\omega_{rej}$ | 575 元/单 | 主动 reject 惩罚（当前 best 默认） |
| $\omega_{unf}$ | 750 元/单 | 未显式 reject 但最终未完成订单惩罚 |
| $\omega_{trip}$ | 200 元/h | 单趟超时软兜底 |
| $\eta(W)$ | $0.18\cdot(1+W/10000)$ kWh/km | 随载重增加的能耗系数 |
| $p_e$ | 1.0 元/kWh | 电价 |

---

## 3. 目标函数

### 3.1 报告目标（论文 / 评测口径，纯 RMB）

$$
\min J^{raw} =
\underbrace{\sum c_{ij}(W_{ij})}_{\text{能耗}}
+ \underbrace{\sum_{i \in D^P} \omega_p \delta_i^P}_{\text{乘客 delivery 延误}}
+ \underbrace{\sum_{i \in D^C} \omega_c \delta_i^C}_{\text{货物 delivery 延误}}
+ \underbrace{\omega_v K_{used}}_{\text{车辆成本}}
+ \underbrace{\omega_{rej} N_{rej}}_{\text{主动拒单}}
+ \underbrace{\omega_{unf} N_{unf}}_{\text{未履约兜底}}
+ \underbrace{\omega_{trip} \sum_k (T_k^{trip} - \tau_{trip}^{max})^+}_{\text{单趟超时}}
$$

其中：

- $\delta_i$ 仅在 **delivery 节点** 累计；pickup 侧不计软延误。
- 当前代码中 `reject_penalty` 由 `rejected_orders * ALPHA_REJECT` 体现主动 reject。
- 对未显式 reject 但最终未完成的订单，额外施加 `unfulfilled_penalty = unfulfilled_orders * ALPHA_UNFULFILLED`，防止模型通过静默漏单逃逸。

### 3.2 训练目标（归一化 + 加权）

$$
J^{train} =
\alpha_E \frac{c^{raw}}{\bar c^{(N)}}
+ \alpha_D \left(\frac{\delta_p^{raw}}{\bar\delta_p^{(N)}} + \frac{\delta_c^{raw}}{\bar\delta_c^{(N)}}\right)
+ \alpha_V \frac{V^{raw}}{\bar V^{(N)}}
+ \omega_{rej} N_{rej}
+ \omega_{unf} N_{unf}
+ \omega_{trip} \sum_k (T_k^{trip} - \tau_{trip}^{max})^+
$$

当前代码对应：

- `ALPHA_ENERGY = 1`
- `ALPHA_DELAY = 2.5`
- `ALPHA_VEHICLE = 3`
- `ALPHA_REJECT = 575`
- `ALPHA_UNFULFILLED = 750`
- `ALPHA_TRIP_OVERTIME = 200`

归一化 profile 仍需按 graph size 通过 `calibrate_normalization` 预先校准。

---

## 4. 约束体系

### 4.1 Pickup-Delivery 先序

delivery 节点只有在对应 pickup 完成后才能访问。DRL 中通过 `StateMCVRPPDTW.get_mask()` 中的 precedence mask 实现。

### 4.2 双舱容量约束

- Passenger 与 Cargo 使用独立容量槽。
- `demand_passenger` / `demand_cargo` 仍以各自容量归一化后进入 state。
- pickup 为正，delivery 为负。

### 4.3 Passenger pickup 硬时间窗

Passenger pickup 到达时刻必须满足：

$$
t_{p_i} \le b_{p_i}
$$

该约束在 DRL 侧被硬 mask；晚到的 pickup 动作不会出现在可行动作集里。

### 4.4 Passenger ride-time 双层硬约束

Passenger 从 pickup 服务完成到 delivery 到达的总 ride time 要满足：

$$
ride_i \le 70\text{ min}
$$

并且相对直达路程的超额 ride time 要满足：

$$
ride_i - ride_i^{direct} \le 30\text{ min}
$$

两层都在 `ride_time_mask` 中以硬过滤实现。

### 4.5 单趟时长与运营终点

- 单趟时长：`MAX_TRIP_TIME = 3h`
- 运营结束：任何候选动作若导致“到点 + 服务 + 回库”超过 16:00，则直接被 mask。

### 4.6 车辆上限

当前代码维持：

$$
K_{used} \le \lceil N / 6 \rceil
$$

这一点与旧 v6 文档中的 $\lceil N/8 \rceil$ 已不同；若将来重新调整，需同步修改代码与文档。

### 4.7 可学习 reject 动作

当前网络已把动作空间从 `N+1` 扩展到 `N+2`：

- `0`：depot
- `1..2N`：真实节点
- `2N+1`：reject 动作

reject 动作被选中时，state 依据当前 mask 中**可达且 tw_late 最紧迫的未服务 pickup**，确定性拒绝该订单，并同时永久屏蔽其 pickup / delivery。

---

## 5. 数据生成与节点表示

### 5.1 数据生成

`MCVRPPDTWDataset` 当前口径：

| 维度 | 当前实现 |
|---|---|
| 节点坐标 | 50% U(0,1)^2 + 50% 高斯聚类 |
| Passenger/Cargo 比例 | 0.6 / 0.4 |
| Passenger demand | 1–2 / 3–4 混合 |
| Cargo demand | 1–3 |
| 时间窗 | 10-12 / 12-14 / 14-16 三时段混合采样；默认权重：乘客 `(0.56, 0.29, 0.15)`、货物 `(0.58, 0.28, 0.14)` |
| Delivery TW | 基于 pickup TW + 直达行驶 + service 时间推导 |

### 5.2 节点输入特征与类型嵌入

当前网络侧已完成类型嵌入改造：

- 连续特征：`[x, y, demand_passenger, demand_cargo, tw_start, tw_end]`
- 离散类型：通过 `type_embedding` 处理

5 类 type id：

| id | 语义 |
|---|---|
| 0 | depot |
| 1 | passenger pickup |
| 2 | passenger delivery |
| 3 | cargo pickup |
| 4 | cargo delivery |

因此 `node_type` 不再只作为单个数值标量输入线性层，而是被拆成离散语义 embedding + 连续特征投影再相加。

### 5.3 PD 对感知注意力 bias

encoder 的 `MultiHeadAttention` 已加入 `pd_bias` 与 `pd_pair_mask`：

- `pd_pair_mask[b, i, j] = True` 表示节点 `i,j` 是同一订单的 pickup / delivery 对
- compatibility 上叠加可学习标量 bias

这使得 encoder 在编码阶段就能对 PD 配对结构产生显式 inductive bias。

---

## 6. DRL 状态与动作约束

`StateMCVRPPDTW` 当前维护：

- 当前时间 / 当前 trip start
- 双舱已用容量
- 已启用车辆数
- 已访问节点
- 已 pickup 标记
- passenger pickup finish time
- rejected 订单标记与 reject_count

`get_mask()` 当前同时覆盖：

1. visited mask
2. PD precedence
3. dual-capacity
4. passenger pickup TW hard mask
5. passenger ride-time hard mask
6. trip-time hard mask
7. operation-end hard mask
8. no-carry-back-to-depot
9. vehicle-limit hard mask
10. reject-action feasibility mask

---

## 7. 网络架构与训练方式

### 7.1 Encoder-Decoder 主体

- `GraphAttentionEncoder`：Kool 风格多层 transformer encoder
- `AttentionModel`：decoder 逐步生成动作序列
- decoder step context 包含：
  - 当前节点 embedding
  - 剩余 passenger 容量
  - 剩余 cargo 容量
  - 归一化当前时间
  - 剩余车辆预算

### 7.2 当前三项关键改造状态

| 改造 | 状态 | 说明 |
|---|---|---|
| 类型嵌入 | 已实现 | `type_embedding + numeric_proj` |
| PD 对感知 bias | 已实现 | `pd_bias + pd_pair_mask` 透传 encoder |
| 可学习 reject 动作 | 已实现 | 动作空间扩展至 `N+2`，state 决定 reject 目标 |

### 7.3 POMO 训练路径

当前训练器不再是“单 rollout 假 POMO”，而是：

- `pomo_size > 1` 时复制 batch 并并行 rollout
- 形成 `costs (B, pomo)`
- `_pomo_loss` 使用同 instance 多解均值作为共享 baseline

这是一个**可工作版 POMO-style 多 rollout**，但并未强制“不同首步起点”。在当前含时间窗、类型与 PD 配对的约束问题上，我们保留 shared-baseline 多采样实现，而不强行改造成经典 N-start POMO。

---

## 8. 代码真值说明

当前项目中，如文档与代码冲突，以以下文件为真值来源：

1. `problem_mcvrptw_v2.py`
2. `state_mcvrptw_v2.py`
3. `nets/attention_model.py`
4. `nets/graph_encoder.py`
5. `run_training_optimized.py`

特别注意：

- `VEHICLE_COST` 当前保留 **20**，不是旧文档中的 5
- `DEFAULT_NUM_VEHICLE_RATIO` 当前保留 **1/6**，不是旧文档中的 1/8
- Passenger excess ride time 当前是 **30 min**，不是旧文档里曾出现的 20 min

---

## 9. 后续实验提醒

在进入正式训练前，仍建议执行：

```bash
python run_training_optimized.py --graph_sizes 25 --calibrate-before-train
```

正式课程训练与多种子实验，仍应以新 profile JSON、固定 seed、统一 checkpoint 命名为复现实验基础。
