# Baselines & Version Management

> 目的：记录可复现封版版本（代码 commit / tag / 对应实验资产），用于后续 N=200 扩展时快速回退与对比。

## 1) N100 稳定基线（封版）

- 名称：`baseline-n100-stable-20260706`
- commit：`2fa5eaa`
- 分支：`reject-split-doc-sync`
- 对应核心结果：
  - 2000 样本评测：`service_rate_mean=0.893175`, `rejected_rate_mean=0.106825`, `unfulfilled_rate_mean=0.0`
  - 代表性样本：`sample_index=1789`
- 对应离线归档：
  - `~/Downloads/project-vrp-archives/_keep_n100_20260706_211736/analysis_bundle/analysis_bundle_n100_20260706_202322/`

## 2) N200 扩展起点

- 名称：`n200-exp-start`
- 基于：`baseline-n100-stable-20260706`
- 目标：在不破坏 N100 基线的前提下，增加 N=200 训练配置并开展可行性训练

## 3) 回退命令

```bash
git fetch --tags
git switch reject-split-doc-sync
git reset --hard baseline-n100-stable-20260706
```

## 4) 版本管理约定

1. 每次“可复现里程碑”必须打 tag。  
2. 做新规模实验（如 N=200）前，先从稳定基线切分支。  
3. 大体积产物不进 git（模型、日志、评测包放归档目录）。
