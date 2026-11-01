# 2026-11-01有用资产回收

迁移后旧GPU共盘点1435个科研代码/配置/结果/权重文件。与本地按SHA256比对后，代码与小型证据已有本地副本，本次新增下载12个关键checkpoint，全部逐文件大小与SHA256通过。约381.83MiB。raw/保留云端相对路径。

用途：N50历史基线，以及旧成本v3正式/诊断的best和final权重，用于复现成本修复前行为。service_first4321属于提前停止的未完成训练，不得当作完整20轮结果。全部权重不作为当前成本v4最终模型。

selected.json记录选择及理由，verification.json是验收，remote_inventory.json为来源清单。重复压缩包、短跑权重和逐轮checkpoint未继续下载；源文件未删除。未选择文件不等于建议删除，历史小文件仍在20261031/raw。

GitHub发布源码、用途索引与摘要，不上传checkpoint。分支codex/cloud-recovery-20261031，源码整理提交818e74a；当前源码以research_snapshot_20261101/SOURCE_MANIFEST.json冻结，保留根目录旧版本。
