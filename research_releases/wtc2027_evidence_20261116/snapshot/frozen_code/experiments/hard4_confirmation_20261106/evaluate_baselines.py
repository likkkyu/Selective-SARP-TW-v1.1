"""One shared-instance evaluator: strict insertion, bounded LNS and hard4 SCIP.

Preserves both algorithm budgets and actual end-to-end times. All returned plans
are checked on original samples, with the same independent tensor C4 scorer.
"""
from common import *
import argparse,json,hashlib,time,signal,sys
P=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/hard4_lns_20261106'))
from route_engine import RouteEngine,split_actions
from search import solve
from strict_insertion import make_start
from scip_boundary_v2 import BoundarySafeSCIP


def verify(sample,actions):
 n=len(sample['loc'])//2;a=audit(sample,actions,audit_config(C,n,(n+5)//6))
 if a['violations'] or a['unfulfilled']:raise RuntimeError(a)
 engine=RouteEngine(sample,C,PROFILE);routes=split_actions(actions,n);cost=engine.cost(routes)
 if cost is None:raise RuntimeError('Independent route engine rejects plan')
 with torch.no_grad():j,d=MCVRPPDTW.get_costs(sample_to_batch(sample),torch.tensor([actions],dtype=torch.long),return_details=True)
 if abs(float(j)-cost['objective'])>.02 or abs(float(d['total_cost_raw'])-cost['raw_cost'])>.02:raise RuntimeError('Common C4 replay mismatch')
 types=sample['node_type'][:n].tolist();completed={node-n-1 for node in actions if n<node<=2*n};pax=sum(t==1 for t in types);ps=sum(types[i]==1 for i in completed)
 return dict(audit=a,cost=cost,completed_per_trip=[len(r)//2 for r in routes],passenger_total=pax,passenger_completed=ps,cargo_total=n-pax,cargo_completed=a['completed']-ps,tensor_objective_error=abs(float(j)-cost['objective']),tensor_raw_error=abs(float(d['total_cost_raw'])-cost['raw_cost']))


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
 plan=json.loads(Path(args.plan).read_text());source=Path(plan['scenario']);out=Path(args.output);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
 for path,digest in plan['source_sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
 assert hashlib.sha256(source.read_bytes()).hexdigest()==plan['scenario_sha256']
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Baseline evaluator finite total budget')));signal.alarm(plan['baseline_timeout_seconds'])
 bundle=torch.load(source,weights_only=False);data=bundle['data'][:plan['samples']];assert len(data)==plan['samples'] and bundle['meta']['graph_size']==50
 Config.set_normalization_profile(50,PROFILE);Config.PASSENGER_MAX_EXCESS_RIDE_TIME_MINUTES=bundle['meta']['excess_minutes'];C['PASSENGER_MAX_EXCESS_RIDE_TIME_MINUTES']=bundle['meta']['excess_minutes'];assert Config.PASSENGER_MAX_RIDE_TIME_MINUTES==bundle['meta']['total_ride_minutes']
 report=dict(status='running',scope=plan['scope'],rows=[],cost_version=4,hard_minimum=4,torch_threads=torch.get_num_threads(),scenario_sha256=plan['scenario_sha256'])
 def save():report['seconds']=time.monotonic()-started;(out/'report.json').write_text(json.dumps(report,indent=2))
 save()
 try:
  for i,sample in enumerate(data):
   begin=time.monotonic();initial=make_start(sample);init_seconds=time.monotonic()-begin
   begin=time.monotonic();initial_check=verify(sample,initial['actions']);initial_audit_seconds=time.monotonic()-begin
   entry=dict(instance=i,initialization_seconds=init_seconds,initial_audit_seconds=initial_audit_seconds,initial=dict(actions=initial['actions'],**initial_check),LNS=[])
   (out/f'instance_{i}.json').write_text(json.dumps(entry,indent=2))
   for seed in plan['search_seeds']:
    engine=RouteEngine(sample,C,PROFILE)
    result=solve(engine,split_actions(initial['actions'],50),seed,plan['lns_seconds'],plan['lns_max_iterations'])
    begin=time.monotonic();checked=verify(sample,result['actions']);audit_seconds=time.monotonic()-begin
    assert checked['cost']['objective']<=initial_check['cost']['objective']+1e-8
    anytime={}
    for deadline in [1.,3.,10.]:
     available=[r for r in result['history'] if init_seconds+r['seconds']<=deadline]
     anytime[str(deadline)]=dict(objective=available[-1]['objective'],completed=available[-1]['completed']) if available else None
    entry['LNS'].append(dict(result,seed=seed,initialization_seconds=init_seconds,total_algorithm_seconds=init_seconds+result['seconds'],final_audit_seconds=audit_seconds,end_to_end_deadline_checkpoints=anytime,**checked));(out/f'instance_{i}.json').write_text(json.dumps(entry,indent=2))
    print('LNS',i+1,'/',len(data),'seed',seed,'served',checked['audit']['completed'],'J',round(checked['cost']['objective'],4),'total_algorithm_s',round(init_seconds+result['seconds'],3),flush=True)
   begin=time.monotonic();solver=BoundarySafeSCIP(C,PROFILE,vehicles=9,time_limit=plan['scip_seconds'],seed=plan['scip_seed'],threads=1,verbose=False)
   result=solver.solve(sample,warm_actions=initial['actions'],logfile=out/f'scip_{i}.log');external_seconds=time.monotonic()-begin
   assert result['actions'] is not None
   begin=time.monotonic();checked=verify(sample,result['actions']);audit_seconds=time.monotonic()-begin
   entry['SCIP']=dict(result=result,external_solver_seconds=external_seconds,total_algorithm_seconds=init_seconds+external_seconds,final_audit_seconds=audit_seconds,**checked)
   (out/f'instance_{i}.json').write_text(json.dumps(entry,indent=2));report['rows'].append(dict(instance=i,initial_completed=initial_check['audit']['completed'],LNS_completed=[r['audit']['completed'] for r in entry['LNS']],SCIP_completed=checked['audit']['completed'],hard_invalid=0,unfulfilled=0));save()
   print('SCIP',i+1,'/',len(data),'served',checked['audit']['completed'],'status',result['status'],'J',round(checked['cost']['objective'],4),'gap',round(result['gap'],4),flush=True)
  report['status']='completed'
 except BaseException as exc:report.update(status='failed',error=repr(exc));raise
 finally:save();signal.alarm(0)
if __name__=='__main__':main()
