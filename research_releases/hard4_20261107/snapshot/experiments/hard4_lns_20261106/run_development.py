"""Frozen bounded development pilot. No independent test-set claims."""
from bootstrap import *
from route_engine import RouteEngine,split_actions
from search import solve
from strict_insertion import make_start
import hashlib,json,time,signal,statistics
import argparse
P=Path(__file__).resolve().parent


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',default='development_v2');parser.add_argument('--scenario',default=str(ROOT/'experiments/paper_delivery_20261104/development_n50_seed614729.pt'));args=parser.parse_args()
 out=P/args.output;out.mkdir(exist_ok=False);started=time.monotonic()
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('300s total pilot cap')));signal.alarm(300)
 source=Path(args.scenario).resolve()
 plan=dict(scope='four previously used development instances; protocol selection pilot, not held-out performance evidence',scenario=str(source),scenario_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),instances=[0,1,2,3],search_seeds=[1234,4321,8765],search_seconds_after_initialization=10.,max_iterations=1000,total_timeout_seconds=300,hard_minimum_completed_pairs=4,cost_version=4,physical_config=C,normalization_profile=PROFILE,source_sha256={f:hashlib.sha256((P/f).read_bytes()).hexdigest() for f in ['route_engine.py','search.py','run_development.py']})
 (out/'plan.json').write_text(json.dumps(plan,indent=2));data=torch.load(source,weights_only=False)['data'];rows=[];status=dict(status='running',rows=rows)
 (out/'runner_frozen.py').write_text(Path(__file__).read_text())
 Config.set_normalization_profile(50,dict(PROFILE,graph_size=50))
 def save():
  status['seconds']=time.monotonic()-started;(out/'report.json').write_text(json.dumps(status,indent=2))
 save()
 try:
  for idx in plan['instances']:
   s=data[idx];begin=time.monotonic();initial=make_start(s);initial_seconds=time.monotonic()-begin;initialroutes=split_actions(initial['actions'],50)
   (out/f'initial_{idx}.json').write_text(json.dumps(dict(seconds=initial_seconds,**initial),indent=2))
   for seed in plan['search_seeds']:
    # Independent run: no route-cache or stochastic search sharing across seeds.
    engine=RouteEngine(s,C,PROFILE)
    def progress(r):print('IMPROVE instance',idx,'seed',seed,'served',r['completed'],'J',round(r['objective'],4),'time',round(r['seconds'],2),flush=True)
    result=solve(engine,initialroutes,seed,plan['search_seconds_after_initialization'],plan['max_iterations'],progress)
    a=audit(s,result['actions'],audit_config(C,50,9));assert not a['violations'] and a['unfulfilled']==0,a
    reference=scalar_cost(s,result['actions'],C,PROFILE);assert max(abs(result['cost'][k]-reference[k]) for k in reference)<1e-7
    with torch.no_grad():j,d=MCVRPPDTW.get_costs(sample_to_batch(s),torch.tensor([result['actions']],dtype=torch.long),return_details=True)
    assert abs(float(j)-result['cost']['objective'])<.02 and abs(float(d['total_cost_raw'])-result['cost']['raw_cost'])<.02
    assert result['cost']['objective']<=initial['cost']['objective']+1e-8
    row=dict(instance=idx,seed=seed,initialization_seconds=initial_seconds,total_seconds=initial_seconds+result['seconds'],initial_cost=initial['cost'],audit=a,tensor_objective_error=abs(float(j)-result['cost']['objective']),tensor_raw_error=abs(float(d['total_cost_raw'])-result['cost']['raw_cost']),**result)
    rows.append(row);save();print('DONE',len(rows),'/12','instance',idx,'seed',seed,'initial',initial['audit']['completed'],'final',a['completed'],'J',round(result['cost']['objective'],4),'iterations',result['iterations'],'seconds',round(row['total_seconds'],2),flush=True)
  status.update(status='passed',mean_service_rate=sum(r['audit']['completed'] for r in rows)/len(rows)/50,mean_initial_service_rate=sum(r['initial_cost']['reject_penalty']/C['ALPHA_REJECT'] for r in rows)/len(rows),all_hard_audits_pass=True)
  status['mean_initial_service_rate']=1-status['mean_initial_service_rate']/50
 except BaseException as exc:status.update(status='failed',error=repr(exc));raise
 finally:save();signal.alarm(0)
if __name__=='__main__':main()
