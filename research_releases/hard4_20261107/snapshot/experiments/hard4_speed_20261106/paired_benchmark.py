"""Same-process sequential paired timing and exact candidate-output regression."""
import json,signal,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
sys.path.insert(0,str(ROOT/'experiments/dispatch_clock_fix_20261106'))
from bootstrap import *
P=Path(__file__).resolve().parent
from clock_patch import install as clock
clock();sys.path.insert(0,str(ALIGN))
from cpu_search_pilot import FixedTwoOptSA,source,ga,Config
sys.path.insert(0,str(ROOT/'experiments/hard4_admission_20261106'))
import admission as ref
import admission_targeted as fast
base=State.get_mask

def main():
 out=P/'paired_benchmark_v1';out.mkdir(exist_ok=False);signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('240s paired performance gate')));signal.alarm(240)
 sample=torch.load(ROOT/'experiments/hard4_admission_20261106/followup_dev4_seed660611.pt',map_location='cpu',weights_only=False)['data'][0];Config.set_normalization_profile(50,json.loads((SNAP/'outputs/normalization_profiles/normalization_n50.json').read_text()))
 common=dict(num_vehicles=9,seed=1234,eval_protocol=source.ALIGNED_PROTOCOL,hard_constraint_mode='strict',semantic_mode='drl_aligned',state_kwargs=KW,allow_reject=True,fill_missing_orders=False)
 rows=[]
 for method in ['SA','GA']:
  method_rows=[]
  for label,module in [('reference',ref),('targeted',fast)]:
   State.get_mask=base
   if hasattr(State,'_hard4_admission_version'):delattr(State,'_hard4_admission_version')
   module.install();module.clear();start=time.monotonic()
   if method=='SA':
    solver=FixedTwoOptSA(iterations=1,initial_temperature=50,cooling_rate=.995,scoring_mode='objective',**common);score,info,_=solver.evaluate_cost(sample,[list(range(i,min(i+6,50))) for i in range(0,50,6)]);actions=info['action_trace'];a=info['independent_audit']
   else:
    solver=ga.GeneticAlgorithmSolver(population_size=48,generations=1,mutation_rate=.2,elite_size=4,**common);original=ga.evaluate_order_routes;captured={}
    def capture(*args,**kw):
     result=original(*args,**kw);captured['actions']=result[2][0].tolist();return result
    ga.evaluate_order_routes=capture
    try:score,info=solver.evaluate_chromosome(sample,list(range(50)))
    finally:ga.evaluate_order_routes=original
    actions=captured['actions']
    from route_audit import audit
    a=audit(sample,actions,dict(start=10.,end=16.,area=10.,speed=25.,trip=3.,service=.05,min_dispatch=4,hard_cargo=True,ride=70/60,excess=.5,max_open=6,vehicles=9))
   seconds=time.monotonic()-start;assert not a['violations'] and not a['unfulfilled'];row=dict(method=method,variant=label,seconds=seconds,objective=score,actions=actions,audit=a,stats=dict(module.STATS));method_rows.append(row);rows.append(row);print(method,label,round(seconds,3),'s',flush=True)
  assert method_rows[0]['actions']==method_rows[1]['actions'] and abs(method_rows[0]['objective']-method_rows[1]['objective'])<.02
  (out/'report.json').write_text(json.dumps(dict(status='running',rows=rows),indent=2))
 summary={method:dict(reference_seconds=next(r['seconds'] for r in rows if r['method']==method and r['variant']=='reference'),targeted_seconds=next(r['seconds'] for r in rows if r['method']==method and r['variant']=='targeted')) for method in ['SA','GA']}
 for s in summary.values():s['reference_over_targeted']=s['reference_seconds']/s['targeted_seconds']
 (out/'report.json').write_text(json.dumps(dict(status='passed',scope='two fixed candidates, one timing pair per method, same machine/thread; preliminary engineering measurement, not algorithm or publication speed claim',summary=summary,rows=rows),indent=2));signal.alarm(0);print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
