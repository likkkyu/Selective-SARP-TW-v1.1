# Closure 收口索引（服务率封版）

- Run TS: `20260725_214810`
- 目标：在不改代码的前提下，补齐 N100/N200 的多 seed 稳定性与 eval2000 收口证据。
- 统一口径：`evaluate_model.py --decode greedy --seed fixed --max_open=6 --min_orders_per_dispatch=4 --enable_delivery_viability`。

---

## 1) 关键结果（基于本地已回收 JSON）

### N100（checkpoint=`outputs_cmp/n100_ours_rw1_fixed/pomo_n100_optimized/model_best_service.pt`）

- eval500 seeds:
  - 10007: service=0.855
  - 10037: service=0.852
  - 10067: service=0.849
  - 10099: service=0.842
  - 99999: service=0.846
- eval500 5-seed 均值（report）：`0.84896`（min=0.84226, max=0.85470）
- eval2000 (seed=99999): `service=0.84383`, `raw=9735.041`, `unfulfilled=0.00048`

### N200（checkpoint=`outputs_n200_phaseC_rw1_e10/pomo_n200_optimized/model_best_service.pt`）

- eval500 seeds:
  - 10007: service=0.854
  - 10037: service=0.862
  - 10067: service=0.863
  - 10099: service=0.857
  - 99999: service=0.865
- eval500 5-seed 均值（report）：`0.860116`（min=0.85352, max=0.86499）
- eval2000 (seed=99999): `service=0.86476`, `raw=17658.945`, `unfulfilled=0.02751`

---

## 2) 封版判定（当前口径）

- 若采用“服务率优先，接受边界波动”的封版口径：
  - N100：可接受（边界结果）
  - N200：可接受（稳定）
- 审计一致性：云端日志显示 partition/core 统计一致（500/500 或 2000/2000）。

---

## 3) 相关文件（应全部位于当前项目目录）

见：`outputs_cmp/CLOSURE_SERVICE_20260725_214810_PULL_LIST.txt`

该清单包含：
- N100 eval500 5 个 seed JSON
- N200 eval500 5 个 seed JSON
- N100 eval2000 JSON
- N200 eval2000 JSON
- 汇总报告 JSON
- 任务 nohup 日志

---

## 4) 本地验收建议

```bash
python3 - <<'PY'
from pathlib import Path
root=Path('/Users/bytedance/Downloads/project-vrp-v6-4-main')
files=[p.strip() for p in open(root/'outputs_cmp/CLOSURE_SERVICE_20260725_214810_PULL_LIST.txt','r',encoding='utf-8') if p.strip()]
missing=[f for f in files if not (root/f).exists()]
print('OK: all closure files are under current project folder.' if not missing else 'MISSING:\n'+'\n'.join(missing))
PY
```

若存在缺失，按 pull list 重新 rsync 拉取即可。
