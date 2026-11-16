"""Recompute pilot report from downloaded machine records."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=Path(__file__).resolve().parent
rows=[]
for mode in ['learned','service_first']:
 p=BASE/'evidence'/f'pilot_{mode}'
 m=json.loads((p/'manifest.json').read_text())
 assert m['status']=='completed'
 for f,h in m['sha256'].items():
  assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h, f
 t=json.loads((p/'timing.json').read_text())
 v=json.loads((p/'outputs/pomo_n50_optimized/training_log.json').read_text())['val'][-1]
 rows.append((mode,m,t,v))
text=['# N50 拒单时机消融短跑报告','',
'GPU: RTX PRO 6000 Blackwell；2组，各2轮，每轮64训练例、32验证例，batch16、POMO1、seed1234。源码哈希已核验；两个运行均完成。', '',
'|策略|总秒数|训练秒/轮|验证秒/轮|末轮服务率|末轮未完成率|',
'|---|---:|---:|---:|---:|---:|']
for mode,m,t,v in rows:
 text.append(f"|{mode}|{m['elapsed_seconds']:.2f}|{t['avg_train_s']:.2f}|{t['avg_validate_s']:.2f}|{v['service_rate']:.4f}|{v['unfulfilled_rate']:.4f}|")
text += ['', '学习组末轮全部拒单，规则组存在5.5%未完成订单。短跑证明执行链条可用，不支持核心机制有效或完整可行的结论。不能把这些结果隐藏或替换为预期收益。', '',
'优先下一步：2组×1种子，10轮×1024训练例，验证128例；观察学习组是否恢复服务及规则组的终止残留。两组配方相同，不单独给一组调参。正式因果比较须有独立约束审计、共同测试集和3个独立训练seed。', '',
'## 根据本次速度线性估时']
for label,epochs,train,val,seeds in [('诊断',10,1024,128,1),('正式候选',20,2048,500,3)]:
 hours=sum((t['avg_train_s']*train/64+t['avg_validate_s']*val/32)*epochs*seeds for _,_,t,_ in rows)/3600
 text.append(f'- {label}：两组，{epochs}轮×{train}例，验证{val}例，{seeds}种子：约{hours:.2f}小时；乘1.5预留约{hours*1.5:.2f}小时。')
text += ['', '估时包含按样本数线性外推的训练/验证；不含独立测试与诊断开发，策略改变后解码耗时可能明显变化。训练是否收敛未知。正式任务尚未启动。', '',
'云端副本：`/root/autodl-tmp/reject_ablation_20261029`；checkpoint保留云端，原始日志与配置已回收本地evidence。当前测试只覆盖规则、掩码及梯度，现有验证不是完整独立可行性证明。']
(BASE/'REPORT.md').write_text('\n'.join(text)+'\n')
print('\n'.join(text))
