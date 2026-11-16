"""Small complete SA/GA searches; development gate, not final algorithm ranking."""
import hashlib,json,math,signal,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
sys.path.insert(0,str(ROOT/'experiments/dispatch_clock_fix_20261106'))
from bootstrap import *
P=Path(__file__).resolve().parent
from clock_patch import install as clock
clock();sys.path.insert(0,str(ALIGN))
from cpu_search_pilot import Config,source,STATS as GUARD_STATS
from cached_search import CachedSA,CachedGA
from admission_targeted import install,clear,STATS
install()

def main():
 out=P/'search_smoke_v1';out.mkdir(exist_ok=False);started=time.monotonic()
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('300s bounded two-method search smoke')));signal.alarm(300)
 scenario=ROOT/'experiments/hard4_admission_20261106/followup_dev4_seed660611.pt';sample=torch.load(scenario,map_location='cpu',weights_only=False)['data'][0];Config.set_normalization_profile(50,json.loads((SNAP/'outputs/normalization_profiles/normalization_n50.json').read_text()))
 plan=dict(scope='one development instance, one search seed; eight candidate evaluations per method; complete short searches, not formal ranking or final tuned configuration',scenario_sha256=hashlib.sha256(scenario.read_bytes()).hexdigest(),instance=0,seed=1234,hard_minimum=4,SA=dict(iterations=7,temperature=50,cooling=.995),GA=dict(population=4,generations=2,elite=1,mutation=.2),timeout_seconds=300,admission_version=3,source_sha256=hashlib.sha256((P/'admission_targeted.py').read_bytes()).hexdigest())
 (out/'plan.json').write_text(json.dumps(plan,indent=2));rows=[];status=dict(status='running',rows=rows)
 def save():status['seconds']=time.monotonic()-started;(out/'report.json').write_text(json.dumps(status,indent=2))
 save()
 try:
  for method in ['SA','GA']:
   clear();GUARD_STATS['recoveries']=GUARD_STATS['failed_recoveries']=0;records=[];begin=time.monotonic()
   common=dict(num_vehicles=9,seed=1234,eval_protocol=source.ALIGNED_PROTOCOL,hard_constraint_mode='strict',semantic_mode='drl_aligned',state_kwargs=KW,allow_reject=True,fill_missing_orders=False)
   solver=CachedSA(**common,iterations=7,initial_temperature=50,cooling_rate=.995,scoring_mode='objective') if method=='SA' else CachedGA(**common,population_size=4,generations=2,elite_size=1,mutation_rate=.2)
   def record(score,info,proposal):
    a=info['independent_audit'];assert not a['violations'] and a['unfulfilled']==0 and math.isfinite(score)
    r=dict(evaluation=len(records)+1,objective=info['objective_cost'],raw_cost=info['raw_total_cost'],actions=info['action_trace'],audit=a);records.append(r)
    (out/f'{method}_candidates.json').write_text(json.dumps(records,indent=2));print('SEARCH',method,len(records),'/8','served',a['completed'],'best_J',round(min(x['objective'] for x in records),3),flush=True)
   solver.recorder=record;best=solver.solve(sample);assert len(records)==8 and abs(best['objective_cost']-min(x['objective'] for x in records))<.02
   assert not best['independent_audit']['violations'] and best['independent_audit']['unfulfilled']==0
   rows.append(dict(method=method,evaluations=len(records),cache_hits=solver.cache_hits,cache_misses=solver.cache_misses,seconds=time.monotonic()-begin,best=best,admission_stats=dict(STATS)));save();print('DONE short search',method,best['independent_audit']['completed'],'/50',flush=True)
  status['status']='passed'
 except BaseException as exc:status.update(status='failed',error=repr(exc));raise
 finally:save();signal.alarm(0)
if __name__=='__main__':main()
