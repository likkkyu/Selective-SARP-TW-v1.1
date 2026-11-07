"""Paired CPU development evaluation of six unchanged checkpoint weights."""
import argparse,gc,hashlib,json,math,signal,time
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"dispatch_clock_fix_20261106"))
from bootstrap import *
P=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['admission'],required=True);p.add_argument('--scenario',required=True);p.add_argument('--output',required=True);p.add_argument('--batch-size',type=int,default=4);p.add_argument('--reference',required=True);args=p.parse_args()
 out=P/args.output;out.mkdir(exist_ok=False);started=time.monotonic()
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('900s six-model development CPU limit')));signal.alarm(900)
 if args.mode=='admission':
  from clock_patch import install as clock
  clock()
 from timing_candidate import install as timing
 from completion_guard import install as guard,STATS
 timing();guard()
 from admission_targeted import install as admission_install,clear as admission_clear,STATS as ADMISSION_STATS
 admission_install()
 from route_audit import audit
 from run_v3 import install_policy
 from depot_noop import install as noop
 from evaluate_model import build_model_from_checkpoint
 from torch.utils.data import default_collate
 scenario=Path(args.scenario).resolve();bundle=torch.load(scenario,map_location='cpu',weights_only=False);data=bundle['data'];assert bundle['meta']['graph_size']==50
 Config.PASSENGER_MAX_EXCESS_RIDE_TIME_MINUTES=bundle['meta']['excess_minutes'];assert Config.PASSENGER_MAX_RIDE_TIME_MINUTES==bundle['meta']['total_ride_minutes']
 profile=json.loads((SNAP/'outputs/normalization_profiles/normalization_n50.json').read_text());Config.set_normalization_profile(50,profile)
 models=json.loads((ROOT/'experiments/paper_delivery_20261104/confirmation_v1/plan.json').read_text())['models']
 cfg=dict(start=Config.OPERATION_START,end=Config.OPERATION_END,area=Config.AREA_SIZE,speed=Config.VEHICLE_SPEED,trip=Config.MAX_TRIP_TIME,service=Config.SERVICE_TIME,min_dispatch=4,hard_cargo=Config.HARD_CARGO_PICKUP_TIMEWINDOW,ride=Config.PASSENGER_MAX_RIDE_TIME_MINUTES/60,excess=Config.PASSENGER_MAX_EXCESS_RIDE_TIME_MINUTES/60,max_open=6,vehicles=9)
 report=dict(status='running',scope='hard-four certified construction development diagnostic; unchanged six weights; distinct from clock-only environment; not formal test or training',mode=args.mode,clock_patch_version=1,admission_version=3,cost_version=4,guard_version=1,scenario=str(scenario),scenario_sha256=hashlib.sha256(scenario.read_bytes()).hexdigest(),samples=len(data),batch_size=args.batch_size,profile=profile,state_kwargs=KW,models=[],source_hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [SNAP/'state_mcvrptw_v2.py',ROOT/'experiments/dispatch_clock_fix_20261106/clock_patch.py',P/'admission_targeted.py',P/'eval_optimized.py']})
 def save():report['seconds']=time.monotonic()-started;(out/'report.json').write_text(json.dumps(report,indent=2))
 save()
 try:
  for spec in models:
   label=f"{spec['policy']}_{spec['seed']}";path=Path(spec['checkpoint']);assert hashlib.sha256(path.read_bytes()).hexdigest()==spec['sha256']
   ckpt=torch.load(path,map_location='cpu',weights_only=False);assert ckpt['normalization_profile']==profile and ckpt['args']['cost_semantics_version']==4
   install_policy(spec['policy']);noop();model=build_model_from_checkpoint(ckpt,torch.device('cpu'));model.load_state_dict(ckpt['model_state_dict']);model.eval();model.set_decode_type('greedy')
   rows=[];model_start=time.monotonic();print('RUN',args.mode,label,len(data),'development instances',flush=True)
   admission_clear();STATS['recoveries']=STATS['failed_recoveries']=0
   for b in range(0,len(data),args.batch_size):
    batch=default_collate(data[b:b+args.batch_size])
    with torch.no_grad():cost,ll,pi=model(batch,return_pi=True,state_kwargs=KW);again,details=MCVRPPDTW.get_costs(batch,pi,return_details=True)
    assert torch.isfinite(cost).all() and torch.isfinite(ll).all() and torch.allclose(cost,again)
    for j in range(pi.shape[0]):
     i=b+j;actions=pi[j].tolist();a=audit(data[i],actions,cfg);delivered={v-51 for v in actions if 51<=v<=100};nt=data[i]['node_type'][:50].tolist();pax=sum(x==1 for x in nt);ps=sum(nt[x]==1 for x in delivered)
     row=dict(instance=i,actions=actions,audit=a,objective=float(cost[j]),raw_cost=float(details['total_cost_raw'][j]),passenger_total=pax,passenger_completed=ps,cargo_total=50-pax,cargo_completed=a['completed']-ps)
     reference_rows=json.loads((ROOT/'experiments/hard4_admission_20261106'/args.reference/f'{label}.json').read_text())['rows']
     assert actions==reference_rows[i]['actions'] and abs(row['objective']-reference_rows[i]['objective'])<.02,('Optimization changed output',label,i)
     rows.append(row);(out/f'{label}.json').write_text(json.dumps(dict(model=spec,rows=rows),indent=2))
     assert a['unfulfilled']==0 and not a['violations'],(label,i,a)
    print('PROGRESS',args.mode,label,len(rows),'/',len(data),flush=True)
   assert len(rows)==len(data) and STATS['failed_recoveries']==0
   r=dict(model=label,seed=spec['seed'],policy=spec['policy'],checkpoint_sha256=spec['sha256'],samples=len(rows),service_rate=sum(x['audit']['completed'] for x in rows)/(50*len(rows)),objective=sum(x['objective'] for x in rows)/len(rows),raw_cost=sum(x['raw_cost'] for x in rows)/len(rows),hard_invalid=0,unfulfilled=0,guard_stats=dict(STATS),admission_stats=dict(ADMISSION_STATS),seconds=time.monotonic()-model_start)
   report['models'].append(r);save();print('DONE',args.mode,label,'service',round(r['service_rate']*100,3),'%',flush=True);del model,ckpt;gc.collect()
  report['status']='completed'
 except BaseException as exc:report.update(status='failed',error=repr(exc));raise
 finally:save();signal.alarm(0)
if __name__=='__main__':main()
