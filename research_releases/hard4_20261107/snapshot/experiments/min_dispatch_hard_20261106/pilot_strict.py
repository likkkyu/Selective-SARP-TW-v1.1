"""Four development instances under user-confirmed hard minimum, no training."""
import hashlib,json,signal,time
from bootstrap import *
from strict_scip import HardMinDispatchSCIP
from strict_insertion import make_start

def main():
 out=P/'pilot_dev4_v2';out.mkdir(exist_ok=False);signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('180s strict development pilot')));signal.alarm(180)
 path=ROOT/'experiments/paper_delivery_20261104/development_n50_seed614729.pt';data=torch.load(path,map_location='cpu',weights_only=False)['data'];Config.set_normalization_profile(50,PROFILE)
 plan=dict(scope='strict-hard-minimum-four development pilot; no test-set inference or complete MIP proof',samples=4,scenario_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),time_limit_per_instance=5,threads=1,seed=1234,min4_hard=True,final_trip_exception=False,pre_execution_decline_allowed=True,profile=PROFILE,physical_and_cost=C,source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [P/'strict_scip.py',P/'strict_insertion.py',P/'generated_hard_warm_start.py']})
 (out/'plan.json').write_text(json.dumps(plan,indent=2));rows=[];begin=time.monotonic()
 for i,s in enumerate(data):
  initial=make_start(s);solver=HardMinDispatchSCIP(C,PROFILE,time_limit=5,verbose=False)
  result=solver.solve(s,warm_actions=initial['actions'],logfile=out/f'instance_{i}.log')
  a=audit(s,result['actions'],audit_config(C,50,9));assert not a['violations'] and a['unfulfilled']==0,a
  with torch.no_grad():j,d=MCVRPPDTW.get_costs(sample_to_batch(s),torch.tensor([result['actions']],dtype=torch.long),return_details=True)
  assert abs(float(j)-result['objective'])<.02 and abs(float(d['total_cost_raw'])-result['replayed']['raw_cost'])<.02
  rows.append(dict(instance=i,initial=initial,SCIP=result,strict_audit=a));(out/f'instance_{i}.json').write_text(json.dumps(rows[-1],indent=2));print('STRICT PASS',i,'completed',a['completed'],'per_trip',result['completed_per_trip'],'status',result['status'],flush=True)
  solver.model.freeProb()
 (out/'report.json').write_text(json.dumps(dict(status='passed',scope=plan['scope'],rows=rows,seconds=time.monotonic()-begin),indent=2));signal.alarm(0);print('PASS strict >=4 four-instance pilot',flush=True)
if __name__=='__main__':main()
