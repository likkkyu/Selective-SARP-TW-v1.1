# N100/N200 checkpoint 选择偏差复核（eval500, seed=99999）

- 运行时间戳：`20260727_120637`
- 协议：`decode=greedy, num_samples=500, batch_size=32, max_open=6, min_orders_per_dispatch=4, delivery_viability=true`
- 云端输出目录：
  - `outputs_cmp/n100_ckpt_biascheck_eval500_20260727_120637`
  - `outputs_cmp/n200_ckpt_biascheck_eval500_20260727_120637`

> 说明：本汇总依据本次运行终端日志整理，作为选择偏差审计记录。

## 1) N100 结果（四 checkpoint）

| checkpoint | service | rejected | unfulfilled | raw cost |
|---|---:|---:|---:|---:|
| model_best_service | 0.846 | 0.154 | 0.000 | 9603.894 |
| model_best | 0.789 | 0.211 | 0.000 | 12800.965 |
| model_final | 0.846 | 0.154 | 0.000 | 9603.894 |
| model_best_objective | 0.846 | 0.154 | 0.000 | 9603.894 |

审计一致性：四个 checkpoint 均 `partition/core=500/500`。

**结论（N100）：**
- `model_best_service == model_final == model_best_objective`（指标一致）；
- `model_best` 明显退化；
- 当前主 checkpoint 选择不存在“挑选偏差导致结论反转”问题。

## 2) N200 结果（三 checkpoint，model_final 缺失）

| checkpoint | service | rejected | unfulfilled | raw cost |
|---|---:|---:|---:|---:|
| model_best_service | 0.865 | 0.107 | 0.028 | 17660.884 |
| model_best | 0.860 | 0.140 | 0.000 | 17134.827 |
| model_best_objective | 0.865 | 0.107 | 0.028 | 17660.884 |

补充：`model_final.pt` 缺失（日志显示 `skip missing model_final.pt`）。

审计一致性：三个 checkpoint 均 `partition/core=500/500`。

**结论（N200）：**
- `model_best_service == model_best_objective`，当前主结论稳定；
- `model_best` 提供“更低 raw / 零 unfulfilled”替代解，但代价是更高 reject 与更低 service；
- 若维持“服务率优先”主口径，当前 checkpoint 选择合理；若改为“clean 优先”，需单独声明口径变化。

## 3) 论文写作建议（与主表一致）

1. 在主文中声明 checkpoint 选择规则（服务率优先），并给出本偏差复核结论；
2. 在附录保留上表，说明 N200 的 trade-off（service vs clean）；
3. 避免在主表混用不同优先级口径（否则会造成选择偏差争议）。
