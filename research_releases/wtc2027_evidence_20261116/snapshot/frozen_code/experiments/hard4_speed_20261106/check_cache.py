"""Cache exactness + reference trajectory on a fixed scored candidate."""
import argparse,hashlib,json,signal,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
sys.path.insert(0,str(ROOT/'experiments/dispatch_clock_fix_20261106'))
from bootstrap import *
P=Path(__file__).resolve().parent
from clock_patch import install as clock
clock();sys.path.insert(0,str(ALIGN))
from cpu_search_pilot import FixedTwoOptSA,source,ga,Config
import admission_fast as fast
from route_audit import audit

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--audit-hits',action='store_true');parser.add_argument('--output',required=True);args=parser.parse_args()
 out=P/args.output;out.mkdir(exist_ok=False);signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('180s cache regression')));signal.alarm(180)
 fast.AUDIT_CACHE_HITS=args.audit_hits;fast.install();fast.clear()
 sample=torch.load(ROOT/'experiments/hard4_admission_20261106/followup_dev4_seed660611.pt',map_location='cpu',weights_only=False)['data'][0]
 Config.set_normalization_profile(50,json.loads((SNAP/'outputs/normalization_profiles/normalization_n50.json').read_text()))
 solver=FixedTwoOptSA(iterations=1,initial_temperature=50,cooling_rate=.995,num_vehicles=9,seed=1234,eval_protocol=source.ALIGNED_PROTOCOL,hard_constraint_mode='strict',semantic_mode='drl_aligned',state_kwargs=KW,allow_reject=True,fill_missing_orders=False,scoring_mode='objective')
 started=time.monotonic();score,info,_=solver.evaluate_cost(sample,[list(range(i,min(i+6,50))) for i in range(0,50,6)]);elapsed=time.monotonic()-started
 ref=json.loads((ROOT/'experiments/hard4_admission_20261106/baseline_entry_v1/report.json').read_text())['rows'][0]
 assert info['action_trace']==ref['actions'] and abs(score-ref['objective'])<.02
 a=info['independent_audit'];assert not a['violations'] and not a['unfulfilled']
 report=dict(status='passed',audit_every_cache_hit=args.audit_hits,seconds=elapsed,reference_seconds=ref['seconds'],identical_actions=True,objective_error=abs(score-ref['objective']),audit=a,stats=dict(fast.STATS),source_sha256=hashlib.sha256((P/'admission_fast.py').read_bytes()).hexdigest(),scope='single fixed candidate, cache-only optimization; timing preliminary, not algorithm speed comparison')
 (out/'report.json').write_text(json.dumps(report,indent=2));signal.alarm(0);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
