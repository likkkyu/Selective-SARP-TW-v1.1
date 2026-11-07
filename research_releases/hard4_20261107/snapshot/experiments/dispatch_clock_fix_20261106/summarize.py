"""Complete descriptive diagnostic summary; no test-set inference or seed selection."""
import csv,hashlib,json,statistics
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
def main():
 regression=json.loads((P/'regression_v2/report.json').read_text());scope=json.loads((P/'mask_scope_v1/report.json').read_text());rejections=json.loads((P/'rejection_trace_v1/report.json').read_text());smoke=json.loads((P/'baseline_smoke_v1/report.json').read_text())
 assert regression['status']==scope['summary']['status']==smoke['status']=='passed' and rejections['status']=='completed'
 rows=[];evaluations=0
 for size in [4,16]:
  old=json.loads((P/f'old_dev{size}_v1/report.json').read_text());new=json.loads((P/f'fixed_dev{size}_v1/report.json').read_text())
  assert old['status']==new['status']=='completed' and old['scenario_sha256']==new['scenario_sha256'] and old['state_kwargs']==new['state_kwargs'] and old['profile']==new['profile']
  for a,b in zip(old['models'],new['models']):
   assert a['model']==b['model'] and a['checkpoint_sha256']==b['checkpoint_sha256'] and a['samples']==b['samples']==size
   assert a['hard_invalid']==b['hard_invalid']==a['unfulfilled']==b['unfulfilled']==0
   rows.append(dict(development_samples=size,model=a['model'],policy=a['policy'],training_seed=a['seed'],old_service_pct=a['service_rate']*100,fixed_service_pct=b['service_rate']*100,difference_pp=(b['service_rate']-a['service_rate'])*100,old_objective=a['objective'],fixed_objective=b['objective'],old_raw_cost=a['raw_cost'],fixed_raw_cost=b['raw_cost']))
  evaluations+=size*12
 plan=json.loads((P/'DEV16_PLAN.json').read_text());assert all(hashlib.sha256((P/k).read_bytes()).hexdigest()==v for k,v in plan['source_sha256'].items())
 frozen=json.loads((P/'PATCH_MANIFEST.json').read_text());assert hashlib.sha256((ROOT/'state_mcvrptw_v2.py').read_bytes()).hexdigest()==frozen['base_module_sha256']
 with (P/'development_comparison.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 groups=[]
 for policy in ['learned','service_first']:
  selected=[x for x in rows if x['development_samples']==16 and x['policy']==policy]
  groups.append(dict(policy=policy,independent_training_seeds=3,old_mean_pct=statistics.mean(x['old_service_pct'] for x in selected),fixed_mean_pct=statistics.mean(x['fixed_service_pct'] for x in selected),difference_mean_pp=statistics.mean(x['difference_pp'] for x in selected)))
 summary=dict(status='diagnostic_completed_business_constraint_decision_pending',patch_regression='passed',clock_version=1,physical_and_cost_versions=[4,1],boundary_cases=len(regression['boundary_cases']),recorded_prefixes_unblocked=regression['unblocked_counterexamples'],mask_scope=scope['summary'],paired_model_instance_evaluations=evaluations,distinct_base_instances=20,new_development_instances=16,hard_invalid=0,unfulfilled=0,groups_on_new_dev16=groups,rejection_diagnosis=rejections['summary'],baseline_smoke=dict(SA_evaluations=96,GA_evaluations=96,scope='functional gate only, not final ranking'),production_root_and_snapshot_unchanged=True,no_weight_update=True,no_gpu=True,no_automation=True,next_decision='clarify four-order construction target vs business hard constraint before changing its mask; no automatic retraining')
 (P/'SUMMARY.json').write_text(json.dumps(summary,indent=2))
 lines=['# 共享出发时钟隔离修复与已有模型诊断','', '本轮完成计时修复及CPU开发诊断，未改生产代码或网络权重，未租GPU、未建定时任务。保留全部原版与修复版结果。','', '## 修复与正确性','',f"12组边界测试通过；32个已记录反例前缀恢复可用。729个抽查状态中376个在途状态完整掩码保持一致；车场不移除原合法服务动作。无关闭commitment、容量、乘车或安全规则。4单处理保持原样。",'', '## 六个旧检查点在新16例开发集的配对结果','', '|模型|原环境服务率/%|计时修复后/%|差/百分点|','|---|---:|---:|---:|']
 for x in [r for r in rows if r['development_samples']==16]:lines.append(f"|{x['model']}|{x['old_service_pct']:.3f}|{x['fixed_service_pct']:.3f}|{x['difference_pp']:+.3f}|")
 lines+=['', '另4例既有开发集同样完成六模型前后复评，见CSV；两组不能混写成单一新确认集。20个基础实例×6模型×2环境共240次模型—实例评价均通过独立硬约束审计、残留0。240不是独立训练重复；每种策略仍是3个冻结训练种子。无重新选检查点、换种子或学习权重更新。','', '旧权重在更正掩码后出现性能下降，不得回退错误计时以保持原成绩，也不能因此断言正确问题天然更难。可行动作扩大并不保证固定贪心策略单调变好。','', '## 拒单机制定位','', f"新16例、六模型轨迹共{rejections['summary']['rejected']}次拒单，全部发生时拒单是环境唯一允许的动作；其中{rejections['summary']['inroute_below4_no_service']}次在途、已完成不足4单且无服务动作。{rejections['summary']['unused_vehicle_could_serve']}次在保留已访问/拒绝状态及车辆数、假设允许先返场换车的诊断状态下，仍有可服务节点。以上是动作事件数，不能当独立订单或训练重复，假设换车也不是已运行的新策略结果。",'', '这表明下降包含4单硬构造门槛与新动作选择的连锁影响，不能全部归为学习拒单头主动拒单，也不能直接据此决定重训。网络此前选择何种路线仍会影响进入死锁的机会；上述计数并未分离所有因果贡献。','', '## 传统基线与当前停止点','', '修复入口已完成SA/GA各96候选、1例1种子的功能预检，候选均经过独立审计、数值有限、残留0；它不是充分搜索或正式算法排名。其余正式预算和最终确认数据暂不启动。','', '论文与当前物理协议把4单作为构造目标；状态代码却将其用于禁止返场/小批新派遣。改变这一处理会改变三类方法的实际策略规则，因此已向用户提出明确选择：隔离取消硬掩码但保留目标统计，或坚持业务硬约束并同步调整MILP及全部协议。用户回复前不做依赖该选择的实验，不启动GPU重训。','', '如选择目标统计，下一轮应单列“仅计时修复”与“计时修复＋取消4单硬门槛”，先复用20个开发实例和同六权重验证，再决定训练预算。不能将两项修改混作一项消融。若选择硬约束，则当前不足4单的SCIP/插入方案不能直接当新业务定义下的有效解，需要统一重建。','', '## 文件','', '- clock_patch.py、generated_get_mask_v1.py、clock_patch.diff、PATCH_MANIFEST.json：隔离补丁及来源。','- regression_v2/：最终回归；regression_v1保留早期测试将预期fail-closed误当错误的记录目录，未覆盖。','- mask_scope_v1/：动作范围核查。','- old_dev4_v1/、fixed_dev4_v1/、old_dev16_v1/、fixed_dev16_v1/：完整模型逐例输出。','- DEV16_PLAN.json、development_n50_seed660611.pt及元数据：预定新开发集协议。','- rejection_trace_v1/：逐次拒单条件和换车可用性诊断。','- baseline_smoke_v1/：修复版传统搜索入口预检。','- development_comparison.csv、SUMMARY.json：汇总。']
 (P/'REVIEW.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
