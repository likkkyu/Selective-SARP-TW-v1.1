# 项目文件控制台账（File Control Ledger）

> 目的：记录本仓库新增文件/文件夹的用途、来源、保留策略与变更历史，避免实验迭代后文件膨胀和证据链断裂。

## 1. 使用规则

1. 新增任何**长期保留**文件/目录（代码、评测结果、可复现快照）时，必须在本台账登记。  
2. 临时文件（如 smoke 输出）默认不入库；若确需保留，需写明“保留原因”和“清理条件”。  
3. 每次重要实验结束后，先归档再清理：
   - 仓库内仅保留当前主线所需文件；
   - 大体积历史产物迁移到仓库外归档目录（例如 `~/Downloads/project-vrp-archives/`）。
4. 变更记录按时间倒序追加，避免覆盖历史。

---

## 2. 路径分级（保留策略）

### A. 核心代码（长期保留）
- `problem_mcvrptw_v2.py`
- `state_mcvrptw_v2.py`
- `run_training_optimized.py`
- `evaluate_model.py`
- `nets/attention_model.py`
- `nets/graph_encoder.py`

### B. 关键配置/文档（长期保留）
- `outputs/normalization_profiles/*.json`
- 本台账 `FILE_CONTROL.md`
- 基线版本记录 `BASELINES.md`
- N200 主线实验看板 `N200_LONGTERM_PLAN_AND_PROGRESS_20260708.md`
- 对比试验总控看板 `COMPARATIVE_EXPERIMENT_PLAN_AND_TRACKER_20260710.md`

### C. 训练与评测产物（按阶段保留）
- `outputs/*`：仅保留当前主线必要目录；历史目录迁移归档。
- `outputs_eval/*`、`outputs_rootcause/*`：仅保留当前分析使用目录。

### D. 临时/中间文件（默认不保留）
- `outputs_test_*`
- `outputs_bench_tmp`
- `.ipynb_checkpoints`
- 本地系统垃圾文件（如 `.DS_Store`）

---

## 3. 当前有效资产快照（2026-07-06）

| 路径 | 类型 | 用途 | 状态 |
|---|---|---|---|
| `state_mcvrptw_v2.py` | 代码 | 状态转移与可行性掩码主逻辑（当前优化主文件） | 活跃 |
| `outputs/` | 目录 | 当前保留训练结果与发布模型 | 活跃 |
| `outputs/release_n100/` | 目录 | 对外可用的 n100 模型版本 | 保留 |
| `outputs/normalization_profiles/` | 目录 | 不同规模归一化常数 | 保留 |

> 备注：历史大文件与实验归档已迁移到仓库外目录 `~/Downloads/project-vrp-archives/`。

---

## 4. 变更记录（Change Log）

| 日期 | 路径 | 变更类型 | 说明 | 操作人 |
|---|---|---|---|---|
| 2026-07-16 | `outputs_cmp/n200_sr_opt_a1_ub003_eval500.json`、`outputs_cmp/n200_sr_opt_a2_ub006_eval500.json`、`COMPARATIVE_EXPERIMENT_PLAN_AND_TRACKER_20260710.md`、`N200_LONGTERM_PLAN_AND_PROGRESS_20260708.md` | 新增+修改 | 完成 N200 服务率优化 A1/A2（decode urgency bias）候选评估并回填台账：A1/A2 均 FAIL（未达到服务率/业务门槛），保留 frozen baseline 不变 | Claude + bytedance |
| 2026-07-16 | `COMPARATIVE_EXPERIMENT_PLAN_AND_TRACKER_20260710.md`、`N200_LONGTERM_PLAN_AND_PROGRESS_20260708.md`、`CURRENT_BEST_CHECKPOINT_SUMMARY.md`、`BASELINES.md`、`FILE_CONTROL.md` | 修改 | 完成 DRL 主实验“服务率优化前冻结”台账：回填 N100/N200 冻结指标与门槛，新增单变量优化队列和基线回退锚点（pre-sr-opt） | Claude + bytedance |
| 2026-07-12 | `baseline_utils.py`、`ga_mcvrppdtw.py`、`sa_mcvrppdtw.py` | 修改 | 完成 GA/SA 与 DRL 的协议对齐升级：新增 `protocol_version/comparable_to_drl`、硬违约统计(`hard_violation_*`)与 aligned cost 机制；支持 `--eval_protocol drl_aligned_v1` 与 `--no-fill-missing-orders`，用于公平主表对比 | Claude + bytedance |
| 2026-07-12 | `outputs_cmp/*_aligned_smoke*.json`、`outputs_cmp/*_legacy_smoke*.json` | 新增（验证产物） | 新增 GA/SA 对齐协议 smoke 验证结果（N25），用于验证“硬违约可见化 + comparable 标记”生效，不直接进入论文主表 | Claude |
| 2026-07-12 | `outputs_cmp/ga_results_25_smoke10_fast.json`、`outputs_cmp/sa_results_25_smoke10_fast.json`、`outputs_cmp/sa_results_25_smoke20.json`（及对应`.log`） | 新增（临时试跑） | 本地后端完成 N25 小样本 GA/SA 对比试验：GA(10)、SA(10/20)；用于明日全量500前快速校验，结论倾向 SA 成本更优（同为 business_clean） | Claude + bytedance |
| 2026-07-12 | `outputs_cmp/ga_results_25_smoke20.json` | 记录（中断未产出） | N25 GA 20样本任务被中止，未生成 JSON；已以 GA(10) 快速试跑替代，避免阻塞本地验证节奏 | Claude + bytedance |
| 2026-07-12 | `COMPARATIVE_EXPERIMENT_PLAN_AND_TRACKER_20260710.md` | 修改 | 回填 N25 fixed protocol 的 rw1/rw0 主结果（基于用户日志：rw1 service=0.831, raw=2582.804；rw0 service=0.814, raw=2894.412），主矩阵将 N25 的 Ours-Champion/Ours-Base 置为完成，并更新 N25 待办为“基线补齐中” | Claude + bytedance |
| 2026-07-12 | `outputs_cmp/n25_ours_rw1_fixed_eval500.json`、`outputs_cmp/n25_ours_rw0_fixed_eval500.json` | 记录/待同步 | 用户侧已完成 eval500 并产生 JSON（日志显示已保存），当前本地仓库尚未收到文件；待同步后纳入证据归档 | Claude + bytedance |
| 2026-07-11 | `COMPARATIVE_EXPERIMENT_PLAN_AND_TRACKER_20260710.md` | 修改 | 扩展总控范围至 N25（标题/固定样本规模/主矩阵/批量命令），并记录 N25 legacy-eval500 与用户回传 N50 resume20 新 champion（service=0.845, raw=4734.072） | Claude + bytedance |
| 2026-07-11 | `outputs_cmp/n25_ours_legacy_eval500.json` | 新增 | 新增 N25 500样本正式评估 JSON（legacy checkpoint，seed=99999，统一 state kwargs）用于先补齐 N25 证据入口 | Claude |
| 2026-07-11 | `outputs/normalization_profiles/normalization_n25.json` | 修改 | 按主线规范将 N25 normalization 校准样本从 8 提升到 256（seed=1234），提升后续 fixed protocol 训练严谨性 | Claude |
| 2026-07-11 | `COMPARATIVE_EXPERIMENT_PLAN_AND_TRACKER_20260710.md` | 修改 | 纠正 N50 固定口径 eval500 回填值（service=0.83736, raw_cost=4954.603），避免沿用旧占位数值 | Claude |
| 2026-07-11 | `outputs_cmp/n50_ours_rw1_eval500.json` | 新增 | 本地完成 N50 固定口径正式评估（500样本，seed=99999），补齐主线证据文件 | Claude |
| 2026-07-11 | `outputs_cmp/*`、`logs/n50_*`、`outputs/normalization_profiles/normalization_n200.json` | 同步/导入 | 从云端回收增量包 `vrp_cloud_delta_core_20260711_193257.tgz` 并解压到仓库，补齐 N50/N100/N200 对比证据文件 | bytedance + Claude |
| 2026-07-11 | `COMPARATIVE_EXPERIMENT_PLAN_AND_TRACKER_20260710.md` | 修改 | 回填 N100 固定口径 Ours-rw1 的训练+eval500结果（service=0.846, raw_cost=9604.049），并更新当前待办 | Claude |
| 2026-07-11 | `COMPARATIVE_EXPERIMENT_PLAN_AND_TRACKER_20260710.md` | 修改 | 新增 N50（rw1固定口径、pomo=2消融）与 N100 legacy(e12) 的 eval500 回填，并更新当前待办 | Claude |
| 2026-07-11 | `N200_LONGTERM_PLAN_AND_PROGRESS_20260708.md` | 修改 | 回填 N200 PhaseC/PhaseB 的 500样本正式评估结果，更新当前阶段、champion与下一步待办 | Claude |
| 2026-07-11 | `COMPARATIVE_EXPERIMENT_PLAN_AND_TRACKER_20260710.md` | 修改 | 回填 N200 双checkpoint正式评估与N50 legacy评估结果，更新当前待办状态（N200结果已回填） | Claude |
| 2026-07-10 | `compare_drl_vs_gurobi.py` | 修改 | 增加 checkpoint 覆盖、state_kwargs 对齐与结果目录参数，修复 DRL 对比默认口径不一致风险并输出运行配置到 JSON | Claude |
| 2026-07-10 | `baseline_utils.py` | 修改 | 基线结果摘要新增业务洁净相关字段与 rate 统计，统一 GA/SA 与 DRL 的主指标口径 | Claude |
| 2026-07-10 | `COMPARATIVE_EXPERIMENT_PLAN_AND_TRACKER_20260710.md` | 修改 | 新增 N100 对比补齐清单（候选 checkpoint、统一评估、基线执行、参数一致性复核） | Claude |
| 2026-07-10 | `COMPARATIVE_EXPERIMENT_PLAN_AND_TRACKER_20260710.md` | 修改 | 按真实训练参数口径重构执行方案：补齐 N50/N100/N200 批量命令、N200 的 GA/SA、Gurobi 重评估与汇总兼容步骤 | Claude |
| 2026-07-10 | `ga_mcvrppdtw.py` | 修改 | 将 `graph_size` 参数扩展到 N200（choices 新增 200），用于补齐 N200 对比试验 | Claude |
| 2026-07-10 | `sa_mcvrppdtw.py` | 修改 | 将 `graph_size` 参数扩展到 N200（choices 新增 200），用于补齐 N200 对比试验 | Claude |
| 2026-07-10 | `evaluate_gurobi_real_cost.py` | 修改 | 将 `graph_size` 参数扩展到 N200（choices 新增 200），用于统一 Gurobi distance 重评估口径 | Claude |
| 2026-07-10 | `COMPARATIVE_EXPERIMENT_PLAN_AND_TRACKER_20260710.md` | 修改 | 更新 N200 对比矩阵为全量（含 GA/SA），补充一键补齐命令与统一执行序列 | Claude |
| 2026-07-10 | `COMPARATIVE_EXPERIMENT_PLAN_AND_TRACKER_20260710.md` | 新增 | 新增对比试验总控看板（N50/N100/N200），固化主对比矩阵、消融矩阵与统一命令模板，保持与当前主线参数口径一致 | Claude |
| 2026-07-10 | `N200_LONGTERM_PLAN_AND_PROGRESS_20260708.md` | 修改 | 回填 Phase B 最终对照结果（rw0_base vs warmup1），更新 champion 与下一阶段为 Phase C 中程验证 | Claude |
| 2026-07-10 | `N200_LONGTERM_PLAN_AND_PROGRESS_20260708.md` | 修改 | 标记 `n200_phaseB_rw0_base` 进入进行中状态，便于隔夜运行与结果回填 | Claude |
| 2026-07-09 | `N200_LONGTERM_PLAN_AND_PROGRESS_20260708.md` | 修改 | 回填 `pomo-size=2` 实验结果（FAIL 回退），并将主线转入 Phase B 基线确认与路线结案判定 | Claude |
| 2026-07-09 | `N200_LONGTERM_PLAN_AND_PROGRESS_20260708.md` | 修改 | 回填 `reject-init-bias=-3.0` 实验结果（FAIL 回退），更新下一候选为 `pomo-size=2` | Claude |
| 2026-07-09 | `N200_LONGTERM_PLAN_AND_PROGRESS_20260708.md` | 修改 | 回填 `reject-init-bias=-2.0` 实验结果（FAIL 回退），更新下一候选为 `reject_init_bias=-3.0` | Claude |
| 2026-07-09 | `N200_LONGTERM_PLAN_AND_PROGRESS_20260708.md` | 修改 | 回填 `rw0_base_rerun1` 复跑结果（PASS），确认基线稳定可复现并更新下一步候选为 `reject_init_bias=-2.0` | Claude |
| 2026-07-09 | `N200_LONGTERM_PLAN_AND_PROGRESS_20260708.md` | 修改 | 回填 `alpha-reject=620` 实验结果（FAIL 回退），触发“连续3次FAIL复盘诊断”并更新下一步队列 | Claude |
| 2026-07-09 | `N200_LONGTERM_PLAN_AND_PROGRESS_20260708.md` | 修改 | 回填 `urgency-bias=0.10` 实验结果（FAIL 回退），更新当前进行中实验与下一里程碑 | Claude |
| 2026-07-08 | `N200_LONGTERM_PLAN_AND_PROGRESS_20260708.md` | 新增 | 新增 N200 长期计划与进度看板（含时间戳、阶段进度、PASS/FAIL 门槛与 Harness 管控方案） | Claude |
| 2026-07-06 | `run_training_optimized.py` | 修改 | 新增 N=200 训练配置（PHASE_CONFIGS）用于扩展实验起步 | Claude |
| 2026-07-06 | `BASELINES.md` | 新增 | 新增可复现封版版本记录（commit/tag/回退命令/归档映射） | Claude |
| 2026-07-06 | `state_mcvrptw_v2.py` | 修改 | 热路径优化：静态几何缓存复用、小规模 open completion 快路径、减少 mask 临时张量分配 | Claude + bytedance |
| 2026-07-06 | `FILE_CONTROL.md` | 新增 | 新增项目文件控制台账，统一记录新增文件/目录用途与变更 | Claude |
| 2026-07-06 | 仓库外归档目录 | 迁移/清理 | 将 `_cleanup_hold_*`、`_keep_n100_*`、`cloud_pull_opt51`、`artifacts` 迁出仓库根目录并清理临时目录 | bytedance |

---

## 5. 后续登记模板

复制以下模板追加：

```markdown
| YYYY-MM-DD | `<path>` | 新增/修改/删除/迁移 | `<用途与原因，是否影响复现>` | `<operator>` |
```
