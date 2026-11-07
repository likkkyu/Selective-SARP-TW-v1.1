"""Bounded corrected-environment SA/GA integration, not a formal comparison."""
import hashlib,json,math,signal,sys,time
from bootstrap import *
from clock_patch import install

def main():
 out=P/'baseline_smoke_v1';out.mkdir(exist_ok=False);signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('180s SA/GA smoke budget')));signal.alarm(180)
 install();sys.path.insert(0,str(ALIGN))
 from cached_search import CachedSA,CachedGA
 from cpu_search_pilot import source,STATS
 Config.set_normalization_profile(50,json.loads((SNAP/'outputs/normalization_profiles/normalization_n50.json').read_text()))
 data=torch.load(P/'development_n50_seed660611.pt',map_location='cpu',weights_only=False)['data'];sample=data[0]
 common=dict(num_vehicles=9,seed=1234,eval_protocol=source.ALIGNED_PROTOCOL,hard_constraint_mode='strict',semantic_mode='drl_aligned',state_kwargs=KW,allow_reject=True,fill_missing_orders=False)
 rows=[];started=time.monotonic()
 for method in ['SA','GA']:
  st=time.monotonic();records=[];STATS['recoveries']=STATS['failed_recoveries']=0
  solver=CachedSA(**common,iterations=95,initial_temperature=50,cooling_rate=.995,scoring_mode='objective') if method=='SA' else CachedGA(**common,population_size=48,generations=2,elite_size=4,mutation_rate=.2)
  def record(score,info,proposal):
   a=info['independent_audit'];assert a['hard_valid'] and a['unfulfilled']==0 and math.isfinite(score)
   records.append(dict(evaluation=len(records)+1,objective=info['objective_cost'],audit=a,actions=info['action_trace']))
   if len(records)%24==0:print(method,'PROGRESS',len(records),'/96',flush=True)
  solver.recorder=record;solution=solver.solve(sample);assert len(records)==96 and STATS['failed_recoveries']==0
  row=dict(method=method,evaluations=96,cache_hits=solver.cache_hits,cache_misses=solver.cache_misses,seconds=time.monotonic()-st,solution=solution,guard_stats=dict(STATS))
  rows.append(row);(out/f'{method}_candidates.json').write_text(json.dumps(records,indent=2));print('PASS',method,'service',solution['independent_audit']['completed'],'/50',flush=True)
 (out/'report.json').write_text(json.dumps(dict(status='passed',scope='one development instance, one search seed, 96 logical candidates per method; smoke test only, not tuned/final rankings',clock_patch_version=1,rows=rows,seconds=time.monotonic()-started),indent=2));signal.alarm(0)
if __name__=='__main__':main()
