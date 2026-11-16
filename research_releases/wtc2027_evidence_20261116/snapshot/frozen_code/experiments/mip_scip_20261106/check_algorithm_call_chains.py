"""Trace actual DRL/SA/GA callers of the shared environment; no production edit."""
import hashlib,inspect,json,signal,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[1];ALIGN=ROOT/'experiments/evaluation_alignment_20261103'
sys.path.insert(0,str(ALIGN))
from cpu_search_pilot import Config,SNAP,source,ga,FixedTwoOptSA,sample_to_batch,torch
from state_mcvrptw_v2 import StateMCVRPPDTW as State
from evaluate_model import build_model_from_checkpoint
from run_v3 import install_policy
from depot_noop import install as install_noop

def main():
 out=P/'algorithm_call_chain_audit_v1';out.mkdir(exist_ok=False)
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('120s call-chain audit limit')));signal.alarm(120)
 sample=torch.load(ROOT/'experiments/paper_delivery_20261104/confirmation_n50_seed803219.pt',map_location='cpu',weights_only=False)['data'][0]
 Config.set_normalization_profile(50,json.loads((SNAP/'outputs/normalization_profiles/normalization_n50.json').read_text()))
 files=['state_mcvrptw_v2.py','problem_mcvrptw_v2.py','baseline_utils.py','nets/attention_model.py','sa_mcvrppdtw.py','ga_mcvrppdtw.py']
 hashes={f:dict(root=hashlib.sha256((ROOT/f).read_bytes()).hexdigest(),snapshot=hashlib.sha256((SNAP/f).read_bytes()).hexdigest()) for f in files}
 assert all(v['root']==v['snapshot'] for v in hashes.values())
 trainings=[]
 for path in (ROOT/'artifacts/ablation_final_20261103/raw').glob('**/manifest.json'):
  manifest=json.loads(path.read_text());trainings.append(dict(run=path.parent.name,training_state_hash=manifest['sha256']['state_mcvrptw_v2.py'],matches_current=manifest['sha256']['state_mcvrptw_v2.py']==hashes['state_mcvrptw_v2.py']['root']))
 assert len(trainings)==6 and all(r['matches_current'] for r in trainings)
 report=dict(status='running',scope='single-instance call-graph probes; not performance measurements or repair',hashes=hashes,training_state_matches=trainings,probes=[])
 original=State.get_mask;active={}
 def traced(state,*args,**kwargs):
  result=original(state,*args,**kwargs);mask=result[0] if isinstance(result,tuple) else result
  active['calls']+=1
  if active['calls']==1:
   active['first_call_stack']=[dict(function=f.function,file=f.filename,line=f.lineno) for f in inspect.stack()[1:9]]
   active['initial_node1_blocked']=bool(mask[0,0,1]);active['initial_time']=float(state.current_time[0,0]);active['initial_prev']=int(state.prev_a[0,0]);active['state_options']=dict(min_orders_per_dispatch=state.min_orders_per_dispatch,max_open=state.max_concurrent_open_orders,viability=state.enable_delivery_viability,skip_commitment=kwargs.get('skip_pickup_commitment',False))
  return result
 State.get_mask=traced
 kw=dict(max_concurrent_open_orders=6,min_orders_per_dispatch=4,enable_delivery_viability=True,enable_viability_fallback=False,relax_pickup_commitment_trip_time=False)
 common=dict(num_vehicles=9,seed=1234,eval_protocol=source.ALIGNED_PROTOCOL,hard_constraint_mode='strict',semantic_mode='drl_aligned',state_kwargs=kw,allow_reject=True,fill_missing_orders=False)
 try:
  install_policy('learned');install_noop()
  ckpt_path=Path(json.loads((ROOT/'experiments/paper_delivery_20261104/confirmation_v1/plan.json').read_text())['models'][0]['checkpoint']);ckpt=torch.load(ckpt_path,map_location='cpu',weights_only=False)
  model=build_model_from_checkpoint(ckpt,torch.device('cpu'));model.load_state_dict(ckpt['model_state_dict']);model.eval();model.set_decode_type('greedy')
  active=dict(algorithm='DRL learned1234',calls=0)
  with torch.no_grad():result=model(sample_to_batch(sample),return_pi=True,state_kwargs=kw)
  report['probes'].append(dict(active));print('PASS DRL actual forward shared-mask calls',active['calls'],flush=True)
  sa=FixedTwoOptSA(iterations=1,initial_temperature=50,cooling_rate=.995,scoring_mode='objective',**common)
  active=dict(algorithm='SA isolated fixed, objective scoring',calls=0)
  sa.evaluate_cost(sample,source.balanced_split(list(range(50)),9) if hasattr(source,'balanced_split') else [list(range(i,min(i+6,50))) for i in range(0,50,6)])
  report['probes'].append(dict(active));print('PASS SA actual scoring shared-mask calls',active['calls'],flush=True)
  g=ga.GeneticAlgorithmSolver(population_size=48,generations=1,mutation_rate=.2,elite_size=4,**common)
  active=dict(algorithm='GA objective scoring',calls=0);g.evaluate_chromosome(sample,list(range(50)))
  report['probes'].append(dict(active));print('PASS GA actual scoring shared-mask calls',active['calls'],flush=True)
  before=[[0],[1,2],[3]];old=source.SimulatedAnnealingSolver.two_opt(sa,before);fixed=sa.two_opt(before)
  assert sorted(x for r in old for x in r)!=[0,1,2,3] and sorted(x for r in fixed for x in r)==[0,1,2,3]
  report['SA_two_opt_regression']=dict(input=before,root_original_output=old,isolated_output=fixed,current_paper_entry='CachedSA -> PilotSA -> FixedTwoOptSA; corrected operator is already used')
  assert all(x['calls']>0 and x['initial_node1_blocked'] for x in report['probes'])
  report['status']='completed';(out/'report.json').write_text(json.dumps(report,indent=2));print('PASS root/snapshot/training identities and all three live call chains')
 finally:State.get_mask=original;signal.alarm(0)
if __name__=='__main__':main()
