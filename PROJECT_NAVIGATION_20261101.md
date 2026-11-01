# 项目文件导航（2026-11-01）

## 当前研究入口

- 核心模型：根目录 problem_mcvrptw_v2.py、state_mcvrptw_v2.py、run_training_optimized.py、evaluate_model.py、nets/、utils/。
- 修复版实验：experiments/sensitivity_20261030/corrected_entry.py 安装成本v4；依赖 timing_candidate.py 及 experiments/reject_ablation_20261029/ 的策略v3与独立审计。该入口与根目录旧成本评价不可互换。
- 最新证据：experiments/sensitivity_20261030/v4_pair1234_review/；单种子阶段性结果，未完成三种子结论。
- 运行记录：experiments/sensitivity_20261030/ACTIVE_RUN.md；自动定时跟进已删除。剩余种子4321/8765已串行安排，已核实4321学习组进入训练。首对额外乘车超限双精度回放0.020803秒，低于预声明0.36秒容差。

## 论文与图表

- wtc2027_zh/：WTC中文工作稿；cictp2027_en/：CICTP英文工作稿。
- paper_revision_20261014/：既有LaTeX稿；model_architecture_20261028/：模型架构图和核查。
- 上述稿件须按最终修复成本、多种子结果更新；现有占位不能当已完成实验。

## 历史与回收

- recovered_cloud_20261031/：云端独有的16个历史代码变体，不直接覆盖主线。
- artifacts/cloud_recovery_20261031/raw/：先前已回收并校验的小文件，保留来源路径。
- artifacts/cloud_recovery_20261101/：迁移后旧GPU清单、用途筛选、关键模型及哈希验收；旧版成本v3权重仅用于历史复现。
- experiments/reject_ablation_20261029/：含旧成本诊断及已停止的正式实验；部分4321对照未完成，不作为完整消融。
- 各种zip/tgz、临时构建和.chart-data目录保留原位，不混入发布包。未做破坏性清理。

## GitHub整理

分支 codex/cloud-recovery-20261031 保留历史回收；research_snapshot_20261101/ 是按本地源码哈希冻结的可运行代码快照，包含核心依赖、实验入口和归一化配置。与仓库根目录旧版本明确区分，运行时从快照根目录执行。模型、原始大日志与私有论文不纳入源码快照。发布结果只含已观察的阶段性数据，不宣称最终统计优势。

## 2026-11-02敏感性完成归档

- artifacts/sensitivity_v4_final_20261102/：最终完整证据包、匹配模型、场景/路线/审计和逐文件校验；265文件通过，17×500最终审计通过。
- experiments/sensitivity_20261030/v4_results_review/：结果表、分类服务率、条件配对区间、四联图及论文可用性报告。
- GitHub对应research_snapshot_20261102/与research_results/sensitivity_v4_seed1234_20261102/；单训练种子结果，不代表消融或其余种子完成。
