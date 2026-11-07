"""One prespecified LNS search seed as a physical-problem reference.

Neural sensitivity uses all three trained policies. LNS is an interpretation
control, not three additional training replicas or an exact monotonicity oracle.
"""
from pathlib import Path
import sys,json,time,signal,hashlib,argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/hard4_confirmation_20261106'))
from common import *
from evaluate_baselines import verify
from strict_insertion import make_start
from route_engine import RouteEngine,split_actions
from search import solve
P=Path(__file__).resolve().parent


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
 plan=json.loads(Path(args.plan).read_text());out=Path(args.output);out.mkdir(parents=True,exist_ok=False);begin=time.monotonic()
 for name,digest in plan['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
 assert hashlib.sha256(Path(plan['scenario']).read_bytes()).hexdigest()==plan['scenario_sha256']
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('1800s bounded sensitivity LNS')));signal.alarm(plan['lns_timeout_seconds'])
 bundle=torch.load(plan['scenario'],map_location='cpu',weights_only=False);data=bundle['data'];assert len(data)==plan['samples']
 Config.set_normalization_profile(50,PROFILE);Config.PASSENGER_MAX_EXCESS_RIDE_TIME_MINUTES=bundle['meta']['excess_minutes'];C['PASSENGER_MAX_EXCESS_RIDE_TIME_MINUTES']=bundle['meta']['excess_minutes']
 rows=[];report=dict(status='running',scope=plan['scope'],factor=plan['window_multiplier'],search_seed=1234,rows=rows)
 def save():report['seconds']=time.monotonic()-begin;(out/'report.json').write_text(json.dumps(report,indent=2))
 save()
 try:
  for i,sample in enumerate(data):
   pipeline_begin=time.monotonic();initial=make_start(sample);init_seconds=time.monotonic()-pipeline_begin
   engine=RouteEngine(sample,C,PROFILE);result=solve(engine,split_actions(initial['actions'],50),1234,10.,1000)
   integrated_seconds=time.monotonic()-pipeline_begin
   t=time.monotonic();checked=verify(sample,result['actions']);audit_seconds=time.monotonic()-t
   row=dict(result,**checked,instance=i,initialization_seconds=init_seconds,total_algorithm_seconds=integrated_seconds,audit_seconds=audit_seconds,initial_cost=initial['cost'])
   rows.append(row);save();print('SENS-LNS',plan['window_multiplier'],i+1,'/',len(data),'served',row['audit']['completed'],'J',round(row['cost']['objective'],4),flush=True)
  report['status']='completed'
 except BaseException as exc:report.update(status='failed',error=repr(exc));raise
 finally:save();signal.alarm(0)
if __name__=='__main__':main()
