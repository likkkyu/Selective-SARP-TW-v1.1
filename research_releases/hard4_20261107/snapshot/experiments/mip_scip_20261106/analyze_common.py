"""Summarize all common instances, preserving unsuccessful and null improvements."""
import csv,hashlib,json,statistics,sys
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[1]

def main():
 out=P/(sys.argv[1] if len(sys.argv)>1 else 'common_n50_15s_v1')
 status=json.loads((out/'status.json').read_text());plan=json.loads((out/'plan.json').read_text())
 if status['status']!='completed' or sorted(status['completed'])!=plan['instances'] or status['errors']:raise RuntimeError('Run incomplete; do not produce final comparison')
 for name,sha in plan['source_sha256'].items():
  if hashlib.sha256((P/name).read_bytes()).hexdigest()!=sha:raise RuntimeError('Source changed after freeze: '+name)
 rows=[json.loads((out/f'instance_{i:02d}.json').read_text()) for i in plan['instances']]
 records=[]
 for r in rows:
  for method,cost,a in [('insertion',r['insertion_start'],r['insertion_audit']),('SCIP_insertion_start',r['scip']['replayed'],r['scip']['audit'])]:
   assert a['unfulfilled']==0 and not any(v['kind']!='min_dispatch' for v in a['violations'])
   records.append(dict(method=method,instance=r['instance'],service_rate=a['completed']/50,objective=cost['objective'],raw_cost=cost['raw_cost'],vehicles=a['vehicles'],dispatch_target_shortfall=any(v['kind']=='min_dispatch' for v in a['violations'])))
 baseline=list(csv.DictReader((ROOT/'experiments/paper_delivery_20261104/confirmation_v1/instance_seed_results.csv').open()))
 for method in ['learned','service_first','SA','GA']:
  for i in plan['instances']:
   group=[r for r in baseline if r['method']==method and int(r['instance'])==i]
   assert len(group)==3
   records.append(dict(method=method,instance=i,service_rate=statistics.mean(float(x['service_rate']) for x in group),objective=statistics.mean(float(x['normalized_objective']) for x in group),raw_cost=statistics.mean(float(x['raw_cost']) for x in group),vehicles='',dispatch_target_shortfall=statistics.mean(x['dispatch_target_shortfall']=='True' for x in group)))
 summary={}
 for method in ['learned','service_first','SA','GA','insertion','SCIP_insertion_start']:
  selected=[r for r in records if r['method']==method]
  summary[method]={key:statistics.mean(r[key] for r in selected) for key in ['service_rate','objective','raw_cost','dispatch_target_shortfall']}
 scips=[r['scip'] for r in rows]
 result=dict(status='completed',scope=plan['scope'],n_instances=len(rows),summary=summary,hard_invalid=0,unfulfilled=0,scip_optimal=sum(r['status']=='optimal' for r in scips),scip_timelimit=sum(r['status']=='timelimit' for r in scips),improved_over_insertion=sum(r['scip']['objective']<r['insertion_start']['objective']-.02 for r in rows),all_served_instances=sum(r['completed']==50 for r in scips),mean_scip_reported_gap=statistics.mean(r['gap'] for r in scips),mean_gap_relative_primal=statistics.mean(r['relative_gap_to_primal'] for r in scips),objective_range=[min(r['objective'] for r in scips),max(r['objective'] for r in scips)],dual_bound_range=[min(r['dual_bound'] for r in scips),max(r['dual_bound'] for r in scips)],max_objective_replay_error=max(r['objective_replay_error'] for r in scips),max_tensor_objective_error=max(r['tensor_objective_error'] for r in scips),max_tensor_raw_error=max(r['tensor_raw_error'] for r in scips),total_local_wall_seconds=status['wall_seconds'],mean_insertion_seconds=statistics.mean(r['insertion_start_seconds'] for r in rows),mean_model_build_seconds=statistics.mean(r['build_seconds'] for r in scips),mean_scip_seconds=statistics.mean(r['solve_seconds'] for r in scips))
 (out/'analysis.json').write_text(json.dumps(result,indent=2))
 with (out/'instance_comparison.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
 lines=['# SCIP共同实例比较','',f"共{len(rows)}个已使用过的N50共同实例；每例SCIP限时{plan['SCIP_seconds_per_instance']}秒，1线程；确定性插入初解独立列出。不是新确认集。",'', '|方法|服务率/%|目标J|综合原始成本|4单构造目标不足方案/%|','|---|---:|---:|---:|---:|']
 for method,r in summary.items():lines.append(f"|{method}|{100*r['service_rate']:.3f}|{r['objective']:.3f}|{r['raw_cost']:.3f}|{100*r['dispatch_target_shortfall']:.3f}|")
 lines+=['',f"SCIP最优状态{result['scip_optimal']}例，限时{result['scip_timelimit']}例；相对插入初解目标改进超过0.02的有{result['improved_over_insertion']}例。全部服务的SCIP方案{result['all_served_instances']}例。插入与SCIP共{2*len(rows)}个最终方案均通过独立物理审计、残留0。",'',f"平均(上界−下界)/|上界|为{100*result['mean_gap_relative_primal']:.2f}%；这是按原始目标上界归一的间隙，区别于SCIP getGap()的分母约定。N50限时结果不等于已证目标最优。",'', '神经/SA/GA每实例为三个冻结种子的均值；SCIP只用一个固定求解器种子、插入初解为确定性策略。这里不把不同种子数解释为等计算预算，不报告公平加速倍数或新显著性检验。', '', '若插入基线优于学习策略，必须更新论文主表及贡献判断；不能删除此基线、只比较冷启动全部拒单结果，或把插入收益归因于SCIP后续搜索。物理模型与成本一致，各算法构造掩码/搜索表示不同。', '', '本轮用于补充MIP参照及检验已有模型的性能主张。若上界大幅优于学习方案，先检查神经构造限制、派遣使用和拒单行为，再决定是否改方法；不为了恢复优势反复改变测试或初解。']
 (out/'REVIEW.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
