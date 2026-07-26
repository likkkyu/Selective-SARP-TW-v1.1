# N200 PhaseC 同步审计记录（2026-07-26）

## 1) 目的

补齐 N200 PhaseC 主实验在本地项目目录中的可追溯证据，并明确“云端不存在”的缺失项，避免后续论文复现时口径不清。

- 云端地址：`root@connect.westb.seetacloud.com:30479`
- 云端根目录：`/root/autodl-tmp/project-vrp-v6-4-main`
- 本地根目录：`/Users/bytedance/Downloads/project-vrp-v6-4-main`

---

## 2) 已确认同步到本地主项目目录的文件

- `outputs_n200_phaseC_rw1_e10/pomo_n200_optimized/model_best.pt`
- `outputs_n200_phaseC_rw1_e10/pomo_n200_optimized/model_best_objective.pt`
- `outputs_n200_phaseC_rw1_e10/pomo_n200_optimized/model_best_service.pt`
- `outputs_cmp/n200_phaseC_rw1_e10_eval500.json`
- `outputs_cmp/n200_phasec_eval2000_20260725_214810.json`
- `outputs_cmp/n200_stability_eval500_20260725_214810/eval500_seed10007.json`
- `outputs_cmp/n200_stability_eval500_20260725_214810/eval500_seed10037.json`
- `outputs_cmp/n200_stability_eval500_20260725_214810/eval500_seed10067.json`
- `outputs_cmp/n200_stability_eval500_20260725_214810/eval500_seed10099.json`
- `outputs_cmp/n200_stability_eval500_20260725_214810/eval500_seed99999.json`
- `outputs_cmp/service_closure_report_20260725_214810.json`
- `logs/service_closure_eval_20260725_214810.nohup.log`

---

## 3) 缺失项（云端核查后确认不存在）

以下文件在本地缺失，并已通过云端目录检查确认“源端即不存在”或未产出：

- `outputs_n200_phaseC_rw1_e10/pomo_n200_optimized/training_log.json`
- `outputs_n200_phaseC_rw1_e10.log`

说明：
- `outputs_n200_phaseC_rw1_e10/pomo_n200_optimized/` 云端仅存在 3 个 checkpoint（`model_best.pt` / `model_best_objective.pt` / `model_best_service.pt`）。
- 因源端缺失，当前无法补拉上述 2 个文件；后续论文复现说明中应明确该限制。

---

## 4) 当前结论

N200 PhaseC 的关键评估与 checkpoint 证据已落地到当前本地主项目目录；
但训练过程日志链路不完整（缺 `training_log.json` 与外层 `phaseC stdout log`），需在论文“可复现性与局限”中如实标注。