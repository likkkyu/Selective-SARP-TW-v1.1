"""Exhaustive tiny-instance oracle, independent audit and tensor cost agreement."""
import itertools,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];P=Path(__file__).resolve().parent
SNAP=ROOT/'experiments/evaluation_alignment_20261103/snapshot'
sys.path.insert(0,str(SNAP));sys.path.insert(0,str(SNAP/'experiments/reject_ablation_20261029'));sys.path.insert(0,str(SNAP/'experiments/sensitivity_20261030'))
import torch
torch.set_num_threads(1)
from problem_mcvrptw_v2 import Config,MCVRPPDTWDataset,MCVRPPDTW
from route_audit import audit
from timing_candidate import install
from baseline_utils import sample_to_batch
from scip_selective import SelectiveSCIP,parameters,scalar_cost,audit_config
from insertion_start import insertion_start
install()
C=parameters(Config);PROFILE=json.loads((SNAP/'outputs/normalization_profiles/normalization_n50.json').read_text())


def exhaustive(s,k):
    n=len(s['loc'])//2;best=None;feasible=0;checked=0
    for mask in range(1<<n):
        ids=[i+1 for i in range(n) if mask>>i&1];nodes=ids+[i+n for i in ids]
        for perm in itertools.permutations(nodes):
            pos={v:i for i,v in enumerate(perm)}
            if any(pos[i]>pos[i+n] for i in ids):continue
            balance=0;cut=[]
            for i,node in enumerate(perm):
                balance+=1 if node<=n else -1
                if balance==0 and i<len(perm)-1:cut.append(i)
            for cuts in range(1<<len(cut)):
                chosen={cut[i] for i in range(len(cut)) if cuts>>i&1}
                if (1 if nodes else 0)+len(chosen)>k:continue
                actions=[]
                for i,node in enumerate(perm):
                    actions.append(node)
                    if i in chosen:actions.append(0)
                if nodes:actions.append(0)
                actions.extend([2*n+1]*(n-len(ids)));checked+=1
                a=audit(s,actions,audit_config(C,n,k))
                if a['unfulfilled'] or any(v['kind']!='min_dispatch' for v in a['violations']):continue
                feasible+=1;score=scalar_cost(s,actions,C,PROFILE)
                if best is None or score['objective']<best['objective']:best=dict(score,actions=actions)
    return dict(best=best,checked=checked,feasible=feasible)


def main():
    out=P/(sys.argv[1] if len(sys.argv)>1 else 'validation_v1');out.mkdir(exist_ok=False)
    tests=[]
    for seed,k in ((760101,1),(760102,2),(760103,1),(760104,2)):
        s=MCVRPPDTWDataset(num_samples=1,graph_size=3,seed=seed,passenger_ratio_override=.6).data[0]
        tests.append((f'generated_{seed}_K{k}',s,k))
    # Structured cases force passenger rejection, capacity splitting, waiting,
    # and cargo lateness above both instance-level piecewise breakpoints.
    def toy(types,loc,windows,dp,dc):
        n=len(types)//2
        return dict(depot=torch.tensor([0.,0.]),loc=torch.tensor(loc),time_windows=torch.tensor(windows),node_type=torch.tensor(types),demand_passenger=torch.tensor(dp),demand_cargo=torch.tensor(dc),n_orders=torch.tensor(n))
    tests.append(('cargo_tiers',toy([0,0,0,0],[[.1,0],[.2,0],[.7,0],[.9,0]],[[10.,11.],[10.,11.],[10.,10.05],[10.,10.05]],[0.,0.,0.,0.],[.6,.6,-.6,-.6]),2))
    tests.append(('passenger_capacity',toy([1,1,1,1],[[.1,0],[.1,.1],[.3,0],[.3,.1]],[[10.,10.1],[10.,10.1],[10.,15.],[10.,15.]],[.8,.8,-.8,-.8],[0.,0.,0.,0.]),1))
    tests.append(('impossible_pickup',toy([1,1],[[1.,1.],[.9,.9]],[[10.,10.01],[10.,15.]],[.1,-.1],[0.,0.]),1))
    report=dict(scope='implementation validation; not a publication performance benchmark',status='running',rows=[])
    for label,s,k in tests:
        n=len(s['loc'])//2;Config.set_normalization_profile(n,dict(PROFILE,graph_size=n))
        oracle=exhaustive(s,k)
        solver=SelectiveSCIP(C,PROFILE,vehicles=k,time_limit=30,gap=0,verbose=False)
        result=solver.solve(s,logfile=out/f'{label}.log')
        a=audit(s,result['actions'],audit_config(C,n,k))
        assert a['unfulfilled']==0 and not any(v['kind']!='min_dispatch' for v in a['violations']),a
        with torch.no_grad():j,details=MCVRPPDTW.get_costs(sample_to_batch(s),torch.tensor([result['actions']],dtype=torch.long),return_details=True)
        tensor_error=abs(float(j.item())-result['replayed']['objective']);raw_error=abs(float(details['total_cost_raw'].item())-result['replayed']['raw_cost'])
        assert result['status']=='optimal',result
        err=abs(result['objective']-oracle['best']['objective'])
        assert err<.02,(label,err,result,oracle)
        assert tensor_error<.02 and raw_error<.02,(tensor_error,raw_error)
        warm=insertion_start(s,C,PROFILE,k,audit)
        warm_solver=SelectiveSCIP(C,PROFILE,vehicles=k,time_limit=30,gap=0,verbose=False)
        warmed=warm_solver.solve(s,warm_actions=warm)
        # addSol may decline an already stored duplicate (e.g. all-reject).
        assert warmed['status']=='optimal' and abs(warmed['objective']-oracle['best']['objective'])<.02,warmed
        row=dict(label=label,orders=n,oracle=oracle,scip=result,warm_scip=warmed,audit=a,oracle_error=err,tensor_objective_error=tensor_error,tensor_raw_error=raw_error)
        report['rows'].append(row);(out/'report.json').write_text(json.dumps(report,indent=2));print('PASS',label,'completed',a['completed'],'objective',round(result['objective'],6),'oracle_error',err,flush=True)
    report['status']='passed';(out/'report.json').write_text(json.dumps(report,indent=2))
    print('PASS all',len(tests),'tiny exact-oracle/audit/tensor cases',flush=True)
if __name__=='__main__':main()
