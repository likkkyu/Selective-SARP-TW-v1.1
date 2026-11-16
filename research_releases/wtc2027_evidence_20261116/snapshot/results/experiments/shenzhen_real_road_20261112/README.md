# 深圳真实路网外部验证（隔离开发）

本目录用于建立不依赖欧氏距离的深圳道路网络评价协议。既有检查点和既有论文结果保持冻结；根目录训练环境已增加**可选**道路矩阵字段，无矩阵输入时维持欧氏逻辑，并由回归测试保护。

## 研究问题

1. 将同一批静态 N50 订单映射到深圳道路节点后，道路最短路距离和行程时间是否会改变可行性、服务率及费用？
2. 冻结模型在真实路网下是否仍能产生满足容量、取送先序、乘客乘车时间、运营时段、单趟时长和“每趟至少完成 4 单”的方案？
3. 在共同道路矩阵和共同订单上，神经构造、严格插入、LNS 与 SCIP 的质量—时间关系如何？

## 分阶段协议

- `P0`：程序化有向小路网，验证解析、单向道路、最短路矩阵、节点映射及矩阵路线回放。
- `P1`：深圳福田—罗湖约 10 km × 10 km 当前 OSM 道路，使用已有合成 N50 订单，仅替换空间与交通层。该阶段同时报告“直线距离+固定25 km/h”“道路距离+固定25 km/h”和“道路距离+OSM等级速度”，分别识别拓扑绕行与速度校准效应。
- `P2`：相同路网，乘客订单由 Shenzhen Urban Data Release 出租车载客状态跳变恢复；货物订单使用深圳货车热点并由 LaDe 的时间窗和距离分布校准。
- `P3`：冻结三训练种子模型、严格插入、LNS、SCIP 在未使用共同实例上的正式比较。先评估，不因结果不利而改协议。

P1/P2 均称为“真实路网驱动的半合成外部验证”，不称为完整实地运营试验。

## 冻结口径

- 主规模：N50，乘客/货物目标比例为 60%/40%；现有生成器逐单伯努利采样，因此逐实例记录实际数量，不声称每例恰为 30/20。
- 运营时段：10:00–16:00。
- 每个订单仅在取送均完成后计 1 单；所有出场车次均须至少完成 4 单。
- 订单可在执行前拒绝；已取件订单不得途中取消。
- 道路距离用于能耗，道路行程时间用于到达、等待、乘车时间、超额乘车时间、单趟时长和运营结束判断。
- 同一实例的所有算法必须共享完全相同的道路距离矩阵、道路时间矩阵、订单、车辆参数和评价器。
- 坐标统一存为 WGS84 `(longitude, latitude)`；任何深圳轨迹数据在确认坐标系前不得直接与 OSM 匹配。

## 数据与许可边界

- OSM/ohsome：保存查询、时间戳、边界框和 ODbL 署名信息；正式稿优先使用与深圳轨迹年份接近的历史快照。
- Shenzhen Urban Data Release：遵守学术用途和再分发限制；仓库只保存解析代码、字段说明、摘要统计和不可逆派生实例，不上传原始轨迹。
- LaDe：仅用于校准货物订单的统计分布，不把其他城市坐标冒充为深圳观测。

## 当前结果

- 早期 P1：32 个冻结 N50 深圳路网实例用于旧模型迁移诊断；这些实例不再作为未见正式测试集。
- 道路最短路替代直线距离后，强 LNS 的平均车辆数增加 2.47 辆、里程增加 175.49 km；详见 `formal_p1_v1/RESULTS_P1.md`。
- 原坐标模型未经修复的道路可行方案为 0/96；它不能被表述为道路感知模型。
- 矩阵感知状态、训练计费和独立路线审计已在 32 例、3477 个动作上交叉验证，零计划动作被误屏蔽。
- 矩阵感知前向、反向、梯度裁剪和优化器更新已通过 CPU 烟雾测试；正式测试集不参与训练。
- 64 例独立开发池的 10 轮 CPU 诊断中，批均值基线最佳验证服务率为 76.0%，自批评贪心基线最佳为 79.0%，各轮硬违规均为 0；该结果仅证明学习链路有效，验证集仅 8 例，不能写作正式效果。
- 现有 32 例 P1 已用于旧模型迁移诊断，后续道路感知模型开发不得把它重新包装为完全未见测试集；模型与超参数冻结后需另生成一次性未观察测试集。

## 2026-11-14 正式道路原生验证

- 三个道路感知模型（训练种子 1234/4321/8765）已先行锁定，模型锁文件为 `road_native_model_lock_v1.json`；随后才用种子 `2026111401` 生成 128 个一次性 N50 正式测试场景。
- 每个场景直接在福田—罗湖最大强连通 OSM 驾车图上抽取 101 个互异道路节点（车场、50 个取件点和 50 个送达点），不存在坐标吸附或距离分层回退。
- 正式测试采用同一有向道路最短路距离矩阵，并按冻结协议的 25 km/h 换算时间矩阵。128/128 场景通过独立结构审计，所有场景都保留了有向路网距离不对称性。
- 需求仍是受控合成取送订单，不是观测出租车或物流订单；因此论文中应称为“深圳真实路网驱动的半合成外部验证”，不能称为完整实地运营数据验证。
- 纯神经构造平均服务率为 90.44%；严格插入为 97.00%。加入相同 3 秒 LNS 后，神经初解与插入初解分别达到 99.99% 和 99.97%，两者配对差异的 95% bootstrap 区间跨越 0。
- 三个训练种子的神经+LNS服务率为 100.00%/100.00%/99.97%；全部 384 个神经+LNS方案和 128 个插入+LNS方案均通过独立硬约束审计。
- 道路矩阵与硬 4 单约束已加入兼容式 SCIP 模型。前 16 个正式测试实例的单线程 5 秒有限时参照全部审计通过，但 16/16 均触及时间上限、0 例证明最优，且未改进插入暖启动；只能报告为有限时 MIP incumbent，不能称为精确最优解。
- 正式统计、置信区间和边界说明见 `RESULTS_REAL_ROAD_FORMAL_V1.md` 与 `road_native_formal_test_v1_analysis.json`；数据池审计见 `road_native_formal_test_v1_pool_audit.json`。
- 论文图、底层数据表、替代文本和来源清单位于 `figures/real_road_validation_v1.*`。

当前 OSM 文件来自 Overpass API，基础数据时间戳为 `2026-11-12T15:36:29Z`，并携带 OpenStreetMap/ODbL 版权声明。正式稿和仓库均需保留 OpenStreetMap contributors 署名；当前路网不能被描述为与 2019 年 Shenzhen Urban Data Release 同期的历史路网。

## 当前入口

```bash
/opt/anaconda3/bin/python experiments/shenzhen_real_road_20261112/test_preflight.py
```

联网获得 OSM 后：

```bash
/opt/anaconda3/bin/python experiments/shenzhen_real_road_20261112/download_overpass.py \
  --output experiments/shenzhen_real_road_20261112/data/futian_luohu_current.json
```

关键本地检查：

```bash
/opt/anaconda3/bin/python experiments/shenzhen_real_road_20261112/test_matrix_aware_neural_state.py
/opt/anaconda3/bin/python experiments/shenzhen_real_road_20261112/test_real_road_state_crosscheck.py
/opt/anaconda3/bin/python experiments/shenzhen_real_road_20261112/smoke_matrix_aware_decode.py
/opt/anaconda3/bin/python experiments/shenzhen_real_road_20261112/smoke_matrix_aware_train_step.py
```

道路原生开发池和正式测试池由 `prepare_road_native_pool.py` 生成，训练由 `train_matrix_aware_finetune.py` 执行。正式测试必须使用 `--purpose formal_test --model-lock ...`，生成器会拒绝在模型锁不存在时创建正式测试集。

正式结果复核入口：

```bash
/opt/anaconda3/bin/python experiments/shenzhen_real_road_20261112/audit_road_native_pool.py \
  --pool experiments/shenzhen_real_road_20261112/road_native_formal_test_v1_seed2026111401_n128.pt \
  --manifest experiments/shenzhen_real_road_20261112/road_native_formal_test_v1_seed2026111401_n128.json \
  --model-lock experiments/shenzhen_real_road_20261112/road_native_model_lock_v1.json \
  --output /tmp/road_pool_audit_recheck.json

/opt/anaconda3/bin/python experiments/shenzhen_real_road_20261112/analyze_frozen_road_results.py
```

以上正式实验均可在本地 CPU 运行；当前不需要租赁 GPU。只有重新训练更大模型、扩大训练池或更换神经架构时才有租 GPU 的必要。

