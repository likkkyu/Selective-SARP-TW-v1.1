"""General batch-one, unchanged-checkpoint hard4 deployment evaluator.

No reference trajectory required on new data. Full final audits never exempt
minimum dispatch size. Cache resets per instance are charged to inference time.
"""
import argparse,gc,hashlib,json,signal,sys,time,math
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
sys.path.insert(0,str(ROOT/'experiments/dispatch_clock_fix_20261106'))
from bootstrap import *
P=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/hard4_speed_20261106'))
sys.path.insert(0,str(ROOT/'experiments/hard4_lns_20261106'))
from route_engine import RouteEngine,split_actions
from search import solve
sys.path.insert(0,str(ROOT/'experiments/neural_diagnosis_20261107'))


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--scenario',required=True);parser.add_argument('--output',required=True);parser.add_argument('--limit',type=int);parser.add_argument('--reference');parser.add_argument('--timeout',type=int,default=3600);parser.add_argument('--plan');parser.add_argument('--guard',choices=['legacy','physical','fast'],required=True);parser.add_argument('--depot',choices=['native','service'],required=True);args=parser.parse_args()
 if args.guard!='fast' or args.depot!='native' or not args.plan:raise ValueError('Frozen fast/native common confirmation requires plan')
 if args.timeout<=0:raise ValueError('Finite timeout required')
 out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Bounded neural evaluation timeout')));signal.alarm(args.timeout)
 from clock_patch import install as clock
 from timing_candidate import install as timing
 from completion_guard import install as guard,STATS
 clock();timing();guard()
 from admission_targeted import install as admission_install,clear as admission_clear,STATS as ADMISSION_STATS
 admission_install()
 if args.guard=='physical':
  from physical_guard import install as physical_install,clear as admission_clear,STATS as ADMISSION_STATS
  physical_install()
 if args.guard=='fast':
  from physical_guard_fast import install as fast_install,clear as admission_clear,STATS as ADMISSION_STATS
  fast_install()
 from route_audit import audit
 from run_v3 import install_policy
 from depot_noop import install as noop
 from evaluate_model import build_model_from_checkpoint
 from torch.utils.data import default_collate
 sys.path.insert(0,str(ROOT/'experiments/mip_scip_20261106'))
 from scip_selective import parameters,audit_config
 scenario=Path(args.scenario).resolve();scenario_sha=hashlib.sha256(scenario.read_bytes()).hexdigest();bundle=torch.load(scenario,map_location='cpu',weights_only=False)
 assert bundle['meta']['graph_size']==50
 Config.PASSENGER_MAX_EXCESS_RIDE_TIME_MINUTES=bundle['meta']['excess_minutes'];assert Config.PASSENGER_MAX_RIDE_TIME_MINUTES==bundle['meta']['total_ride_minutes']
 data=bundle['data'][:args.limit] if args.limit else bundle['data'];assert data
 profile=json.loads((SNAP/'outputs/normalization_profiles/normalization_n50.json').read_text());Config.set_normalization_profile(50,profile)
 model_specs=json.loads((ROOT/'experiments/paper_delivery_20261104/confirmation_v1/plan.json').read_text())['models']
 if args.plan:
  plan=json.loads(Path(args.plan).read_text());assert plan['scenario_sha256']==scenario_sha and plan['models']==model_specs and len(data)==plan['samples']
  for path,digest in plan['source_sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
 cfg=audit_config(parameters(Config),50,9)
 report=dict(status='running',scope='unchanged trained checkpoints evaluated under unified hard4 deployment rules; not retraining or pure training-component ablation',scenario=str(scenario),scenario_sha256=scenario_sha,samples=len(data),batch_size=1,device='cpu',torch_threads=torch.get_num_threads(),versions=dict(clock=1,admission=5 if args.guard=='fast' else 4 if args.guard=='physical' else 3,cost=4,guard=1),physical_config=parameters(Config),normalization_profile=profile,state_kwargs=KW,models=[],timing='algorithm includes collate, cache reset and complete model forward/safe construction; loading and final audit/replay reported separately')
 report.update(prototype_guard=args.guard,depot_policy=args.depot,scope='new common confirmation of frozen fast/native deployment and matched 0.5s LNS; unchanged six weights; not retraining or pure training-component ablation')
 report.update(legacy_mask_used=args.guard=='legacy',legacy_deadlock_termination=args.guard=='legacy',scalar_backend='CPU float32 faithful transition' if args.guard=='fast' else 'actual State virtual replay')
 def save():report['seconds']=time.monotonic()-started;(out/'report.json').write_text(json.dumps(report,indent=2))
 save()
 try:
  for spec in model_specs:
   label=f"{spec['policy']}_{spec['seed']}";load_begin=time.monotonic();path=Path(spec['checkpoint']);assert hashlib.sha256(path.read_bytes()).hexdigest()==spec['sha256']
   ckpt=torch.load(path,map_location='cpu',weights_only=False);assert ckpt['normalization_profile']==profile and ckpt['args']['cost_semantics_version']==4
   install_policy(spec['policy']);noop()
   if args.depot=='service':
    from depot_service import install as depot_install
    depot_install()
   model=build_model_from_checkpoint(ckpt,torch.device('cpu'));model.load_state_dict(ckpt['model_state_dict']);model.eval();model.set_decode_type('greedy');load_seconds=time.monotonic()-load_begin;rows=[]
   reference=json.loads(Path(args.reference,f'{label}.json').read_text())['rows'] if args.reference else None
   print('MODEL',label,'batch1 samples',len(data),'load_seconds',round(load_seconds,2),flush=True)
   for i,sample in enumerate(data):
    begin=time.monotonic();admission_clear();STATS['recoveries']=STATS['failed_recoveries']=0
    batch=default_collate([sample])
    with torch.no_grad():cost,ll,pi=model(batch,return_pi=True,state_kwargs=KW)
    inference_seconds=time.monotonic()-begin
    hybrid_engine=RouteEngine(sample,parameters(Config),profile)
    hybrid=solve(hybrid_engine,split_actions(pi[0].tolist(),50),plan['hybrid_search_seed'],plan['hybrid_search_seconds'],plan['lns_max_iterations'])
    integrated_seconds=time.monotonic()-begin
    assert torch.isfinite(cost).all() and torch.isfinite(ll).all()
    begin=time.monotonic();actions=pi[0].tolist();a=audit(sample,actions,cfg)
    assert not a['violations'] and a['unfulfilled']==0 and STATS['failed_recoveries']==0,(label,i,a)
    engine=RouteEngine(sample,parameters(Config),profile);routes=split_actions(actions,50);independent=engine.cost(routes);assert independent is not None
    with torch.no_grad():again,details=MCVRPPDTW.get_costs(batch,pi,return_details=True)
    assert torch.allclose(cost,again) and abs(float(cost[0])-independent['objective'])<.02 and abs(float(details['total_cost_raw'][0])-independent['raw_cost'])<.02
    ids={node-51 for node in actions if 51<=node<=100};types=sample['node_type'][:50].tolist();pax=sum(t==1 for t in types);ps=sum(types[j]==1 for j in ids)
    row=dict(instance=i,actions=actions,audit=a,objective=float(cost[0]),raw_cost=float(details['total_cost_raw'][0]),independent_cost=independent,completed_per_trip=[len(r)//2 for r in routes],inference_seconds=inference_seconds,audit_replay_seconds=time.monotonic()-begin,passenger_total=pax,passenger_completed=ps,cargo_total=50-pax,cargo_completed=a['completed']-ps,admission_stats=dict(ADMISSION_STATS),guard_stats=dict(STATS))
    if reference is not None:
     row['reference_batch_actions_equal']=actions==reference[i]['actions'];row['reference_objective_difference']=float(cost[0])-reference[i]['objective']
     old=reference[i]['actions'];short,long=(actions,old) if len(actions)<=len(old) else (old,actions)
     row['reference_equal_up_to_terminal_depot_padding']=long[:len(short)]==short and all(x==0 for x in long[len(short):])
     # Batch arithmetic can change tied greedy choices. Report rather than
     # silently force a reference route; equivalence is not assumed.
    h=hybrid['actions'];ha=audit(sample,h,cfg);assert not ha['violations'] and not ha['unfulfilled']
    hc=RouteEngine(sample,parameters(Config),profile).cost(split_actions(h,50));assert hc is not None and hc['objective']<=independent['objective']+1e-8
    with torch.no_grad():hj,hd=MCVRPPDTW.get_costs(batch,torch.tensor([h],dtype=torch.long),return_details=True)
    assert abs(float(hj)-hc['objective'])<.02 and abs(float(hd['total_cost_raw'])-hc['raw_cost'])<.02
    row['hybrid']=dict(hybrid,audit=ha,independent_cost=hc,integrated_algorithm_seconds=integrated_seconds,tensor_objective_error=abs(float(hj)-hc['objective']),tensor_raw_error=abs(float(hd['total_cost_raw'])-hc['raw_cost']))
    rows.append(row);(out/f'{label}.json').write_text(json.dumps(dict(model=spec,load_seconds=load_seconds,rows=rows),indent=2))
    print('PROGRESS',label,i+1,'/',len(data),'served',a['completed'],'J',round(float(cost[0]),4),'algorithm_s',round(inference_seconds,3),'hybrid_served',ha['completed'],'hybrid_total_s',round(integrated_seconds,3),'hard_valid',True,flush=True)
   summary=dict(model=label,policy=spec['policy'],seed=spec['seed'],checkpoint_sha256=spec['sha256'],samples=len(rows),service_rate=sum(r['audit']['completed'] for r in rows)/50/len(rows),objective=sum(r['objective'] for r in rows)/len(rows),load_seconds=load_seconds,total_inference_seconds=sum(r['inference_seconds'] for r in rows),hard_invalid=0,unfulfilled=0)
   report['models'].append(summary);save();del model,ckpt;gc.collect()
  report['status']='completed'
 except BaseException as exc:report.update(status='failed',error=repr(exc));raise
 finally:save();signal.alarm(0)
if __name__=='__main__':main()
