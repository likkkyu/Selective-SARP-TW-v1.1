# N25/N50 eval2000 补seed结果汇总（2026-07-28）

- 记录时间：`2026-07-28`
- 统一协议：`decode=greedy, num_samples=2000, batch_size=32, seed∈{10007,10037,10067,10099,99999}, max_open=6, min_orders_per_dispatch=4, delivery_viability=true`
- 说明：
  - N25 的 4 个新增 seed 来自本次云端运行目录：`outputs_cmp/n25_eval2000_multiseed_20260728_175848`
  - N50 的 2 个新增 seed 来自本次云端运行目录：`outputs_cmp/n50_eval2000_multiseed_add_20260728_193951`
  - N50 的历史 3-seed 来自本地目录：`outputs_cmp/n50_stability_eval2000_20260723_004500`

---

## 1) N25（checkpoint: `outputs_cloud/n25_sr_s1_warmup0_20260724_201223/.../model_best_service.pt`）

### 1.1 新增 4-seed（10007/10037/10067/10099）

| seed | service_rate_mean | rejected_rate_mean | unfulfilled_rate_mean | total_cost_raw_mean |
|---:|---:|---:|---:|---:|
| 10007 | 0.853 | 0.147 | 0.000 | 2247.390 |
| 10037 | 0.852 | 0.148 | 0.000 | 2261.271 |
| 10067 | 0.853 | 0.147 | 0.000 | 2255.499 |
| 10099 | 0.852 | 0.148 | 0.000 | 2267.278 |

4-seed 均值：
- `service_rate_mean = 0.8525`（std≈0.0005）
- `rejected_rate_mean = 0.1475`（std≈0.0005）
- `unfulfilled_rate_mean = 0.0000`
- `total_cost_raw_mean = 2257.860`（std≈7.295）

审计一致性（4 个 seed）：
- `partition_consistent_samples = 2000/2000`
- `core_aggregate_match_samples = 2000/2000`

### 1.2 与既有 seed=99999 合并（5-seed）

> 既有文件：`outputs_cmp/n25_sr_s1_warmup0_eval2000_20260724_201223.json`

合并 5-seed 后：
- `service_rate_mean = 0.852604`
- `rejected_rate_mean = 0.147396`
- `unfulfilled_rate_mean = 0.000000`
- `total_cost_raw_mean = 2256.270`

结论（N25）：
- 5-seed 下结果稳定，且持续 `clean=true`（unfulfilled=0）。
- 相对 N25 fixed baseline（`service=0.831, raw=2582.804`）优势仍明确。

---

## 2) N50（checkpoint: `outputs_cloud/n50_short_train_candidate_v3_rw0_cleanrepair_plus8_lr1e5/.../model_best_service.pt`）

### 2.1 新增 2-seed（10067/10099）

| seed | service_rate_mean | rejected_rate_mean | unfulfilled_rate_mean | total_cost_raw_mean |
|---:|---:|---:|---:|---:|
| 10067 | 0.889 | 0.111 | 0.000 | 3462.350 |
| 10099 | 0.890 | 0.110 | 0.000 | 3430.253 |

2-seed 均值：
- `service_rate_mean = 0.8895`（std≈0.0005）
- `rejected_rate_mean = 0.1105`（std≈0.0005）
- `unfulfilled_rate_mean = 0.0000`
- `total_cost_raw_mean = 3446.302`（std≈16.049）

审计一致性（2 个 seed）：
- `partition_consistent_samples = 2000/2000`
- `core_aggregate_match_samples = 2000/2000`

### 2.2 与既有 3-seed 合并（5-seed）

> 既有 3-seed 文件：
> - `outputs_cmp/n50_stability_eval2000_20260723_004500/eval2000_seed10007.json`
> - `outputs_cmp/n50_stability_eval2000_20260723_004500/eval2000_seed10037.json`
> - `outputs_cmp/n50_stability_eval2000_20260723_004500/eval2000_seed99999.json`

合并 5-seed 后：
- `service_rate_mean = 0.888660`
- `rejected_rate_mean = 0.111340`
- `unfulfilled_rate_mean = 0.000000`
- `total_cost_raw_mean = 3463.079`

结论（N50）：
- N50 在 eval2000 多seed 下继续稳定，且保持 clean（unfulfilled=0）。
- 主结论不变：该 N50 候选具备封版稳定性。

---

## 3) 备注

- 本汇总用于实验台账回填与证据链归档。
- N25/N50 新增 seed 的原始 JSON 建议按既定同步命令拉回本地后再进行一次文件级核验。