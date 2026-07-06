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
