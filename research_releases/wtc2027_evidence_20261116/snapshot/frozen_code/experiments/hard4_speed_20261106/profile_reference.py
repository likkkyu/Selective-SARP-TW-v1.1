"""One fixed SA candidate; representative certificate costs, not a benchmark."""
import cProfile,json,pstats,signal,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
sys.path.insert(0,str(ROOT/'experiments/hard4_admission_20261106'))
from check_baselines import FixedTwoOptSA,source,Config,KW,torch,SNAP,clear,STATS

def main():
 out=P/'profile_v1';out.mkdir(exist_ok=False)
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('180s profiling limit')));signal.alarm(180)
 s=torch.load(ROOT/'experiments/hard4_admission_20261106/followup_dev4_seed660611.pt',map_location='cpu',weights_only=False)['data'][0];Config.set_normalization_profile(50,json.loads((SNAP/'outputs/normalization_profiles/normalization_n50.json').read_text()));clear()
 solver=FixedTwoOptSA(iterations=1,initial_temperature=50,cooling_rate=.995,num_vehicles=9,seed=1234,eval_protocol=source.ALIGNED_PROTOCOL,hard_constraint_mode='strict',semantic_mode='drl_aligned',state_kwargs=KW,allow_reject=True,fill_missing_orders=False,scoring_mode='objective')
 profile=cProfile.Profile();t=time.monotonic();profile.enable();score,info,_=solver.evaluate_cost(s,[list(range(i,min(i+6,50))) for i in range(0,50,6)]);profile.disable();profile.dump_stats(str(out/'cpu.prof'))
 with (out/'top.txt').open('w') as stream:pstats.Stats(profile,stream=stream).strip_dirs().sort_stats('cumtime').print_stats(35)
 (out/'result.json').write_text(json.dumps(dict(scope='profiled single candidate, not normal runtime',seconds=time.monotonic()-t,score=score,actions=info['action_trace'],audit=info['independent_audit'],stats=dict(STATS)),indent=2));signal.alarm(0);print((out/'top.txt').read_text())
if __name__=='__main__':main()
