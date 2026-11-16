# 数据来源与使用边界

## 优先来源

1. **Shenzhen Urban Data Release V2**，Desheng Zhang（Rutgers University）与 Fan Zhang（深圳先进技术研究院等）。官方说明页：<https://people.cs.rutgers.edu/~dz220/Data.html>。出租车字段包含车辆编号、时间、经纬度、载客状态和速度，适合恢复乘客 OD 并校准交通速度。官方 V2 出租车下载地址在 2026-11-12 实测返回 404；说明页仍可访问。
2. **OpenStreetMap** 当前道路，经 Overpass API 获得。正式稿若与 2013/2014 轨迹配对，将改用 ohsome API 的相近年份历史快照。OSM 派生数据遵守 ODbL 和署名要求。
3. **LaDe**（Cainiao-AI），用于货物订单距离、时间窗与服务时长的统计校准，不搬运其他城市坐标到深圳。<https://huggingface.co/datasets/Cainiao-AI/LaDe>。

## 可复核镜像与替代资料

- `Wadesha/taxi-archive` 公开镜像按小时保存 2014-10-22 深圳出租车记录，字段为 `trajectory_id,date,lon,lat,passenger,speed(km/h)`。该镜像不是原发布机构；使用前应核对内容哈希、字段分布、坐标和原始许可，论文仍引用原发布者及 UrbanCPS 数据论文。镜像链接：<https://github.com/Wadesha/taxi-archive>。
- Zenodo `PassengerRequestDataset`（DOI: 10.5281/zenodo.8287367）提供福田—罗湖道路图和合成乘客请求，可作为道路图/需求分布的独立稳健性数据，但不能称为实测乘客 OD。

## 仓库策略

- 不提交原始出租车、公交或货车轨迹。
- 只提交解析代码、协议、哈希、字段/质量摘要、不可逆聚合速度和派生实验结果。
- 坐标必须先通过范围、顺序和道路吸附偏移检查；不以“看起来落在深圳”代替 WGS84/GCJ-02 判定。

