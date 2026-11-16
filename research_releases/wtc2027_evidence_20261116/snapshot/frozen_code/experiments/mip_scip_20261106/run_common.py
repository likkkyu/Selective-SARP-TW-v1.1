"""Bounded common-set SCIP comparison. Output directory is immutable per run."""
import argparse,concurrent.futures,hashlib,json,os,platform,subprocess,sys,time
from pathlib import Path
from validate import ROOT,P,SNAP,C,PROFILE
import torch
from route_audit import audit
from baseline_utils import sample_to_batch
from problem_mcvrptw_v2 import Config,MCVRPPDTW
from scip_selective import SelectiveSCIP,scalar_cost,audit_config
from insertion_start import insertion_start
DATA=ROOT/'experiments/paper_delivery_20261104/confirmation_n50_seed803219.pt'

def worker(task):
    i,out,budget=task;out=Path(out);torch.set_num_threads(1);Config.set_normalization_profile(50,PROFILE)
    sample=torch.load(DATA,map_location='cpu',weights_only=False)['data'][i]
    before=time.monotonic();actions=insertion_start(sample,C,PROFILE,9,audit);insert_seconds=time.monotonic()-before
    initial_audit=audit(sample,actions,audit_config(C,50,9))
    assert initial_audit['unfulfilled']==0 and not any(v['kind']!='min_dispatch' for v in initial_audit['violations']),initial_audit
    result=dict(instance=i,insertion_start_seconds=insert_seconds,insertion_start=scalar_cost(sample,actions,C,PROFILE),insertion_audit=initial_audit,insertion_actions=actions)
    # Incremental local evidence is saved before handing control to SCIP.
    target=out/f'instance_{i:02d}.json';target.write_text(json.dumps(result,indent=2))
    solver=SelectiveSCIP(C,PROFILE,time_limit=budget,verbose=False)
    solved=solver.solve(sample,warm_actions=actions,logfile=out/f'instance_{i:02d}.log')
    a=audit(sample,solved['actions'],audit_config(C,50,9))
    assert a['unfulfilled']==0 and not any(v['kind']!='min_dispatch' for v in a['violations']),a
    with torch.no_grad():obj,details=MCVRPPDTW.get_costs(sample_to_batch(sample),torch.tensor([solved['actions']],dtype=torch.long),return_details=True)
    solved.update(audit=a,tensor_objective_error=abs(float(obj.item())-solved['replayed']['objective']),tensor_raw_error=abs(float(details['total_cost_raw'].item())-solved['replayed']['raw_cost']))
    assert solved['tensor_objective_error']<.05 and solved['tensor_raw_error']<.05,solved
    assert solved['objective']<=result['insertion_start']['objective']+.02
    result.update(scip=solved,total_seconds=time.monotonic()-before,status='completed');target.write_text(json.dumps(result,indent=2))
    solver.model.freeProb()
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--samples',type=int,default=32);p.add_argument('--seconds',type=float,default=30);p.add_argument('--workers',type=int,default=1);args=p.parse_args()
    if not 1<=args.samples<=32 or args.seconds<=0 or not 1<=args.workers<=2:raise ValueError('Invalid pilot/common budget')
    out=P/args.output;out.mkdir(exist_ok=False)
    plan=dict(scope='additional MIP comparison on the existing previously read common set; no model training or tuning; not a new untouched confirmation set',instances=list(range(args.samples)),scenario_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),SCIP_seconds_per_instance=args.seconds,SCIP_threads=1,concurrent_workers=args.workers,solver_seed=1234,gap_limit=0.,initialization='deterministic earliest-deadline cheapest paired insertion with independent hard audit; no neural warm start',profile=PROFILE,hardware=dict(platform=platform.platform(),processor=platform.processor(),logical_cpu_count=os.cpu_count()),source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (P/'scip_selective.py',P/'insertion_start.py',Path(__file__))},comparison_boundary='same physical audit/cost/data as archived neural/SA/GA; distinct search spaces and timing boundaries; no fair acceleration ratio inferred')
    (out/'plan.json').write_text(json.dumps(plan,indent=2))
    status=dict(status='running',completed=[],errors=[],plan=plan);start=time.monotonic()
    def save():status['wall_seconds']=time.monotonic()-start;(out/'status.json').write_text(json.dumps(status,indent=2))
    save();print('START',args.samples,'N50 instances; SCIP budget',args.seconds,'s each;',args.workers,'CPU workers; no GPU',flush=True)
    if args.workers==1:
        for i in plan['instances']:
            try:
                row=worker((i,str(out),args.seconds));status['completed'].append(i);r=row['scip'];print('PROGRESS',len(status['completed']),'/',args.samples,'instance',i,'served',r['completed'],'initial',row['insertion_audit']['completed'],'status',r['status'],'J',round(r['objective'],3),'bound',round(r['dual_bound'],3),'gap',r['gap'],flush=True)
            except Exception as exc:
                status['status']='failed';status['errors'].append(dict(instance=i,error=repr(exc)));save();raise
            save()
        status['status']='completed';save();print('COMPLETE',len(status['completed']),'wall_seconds',status['wall_seconds'],flush=True);return
    def fresh_worker(i):
        command=[sys.executable,str(ROOT/'experiments/evaluation_alignment_20261103/cpu_entry.py'),str(Path(__file__).resolve()),'--worker',str(i),str(out),str(args.seconds)]
        with (out/f'instance_{i:02d}_worker.log').open('w') as stream:
            subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=args.seconds+180)
        return json.loads((out/f'instance_{i:02d}.json').read_text())
    # Fresh cpu_entry processes set Torch threads before project imports. Native
    # multiprocessing spawn imported Torch too early on this macOS/OMP build.
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(fresh_worker,i):i for i in plan['instances']}
        for future in concurrent.futures.as_completed(pending):
            i=pending[future]
            try:
                row=future.result();status['completed'].append(i);r=row['scip'];print('PROGRESS',len(status['completed']),'/',args.samples,'instance',i,'served',r['completed'],'initial',row['insertion_audit']['completed'],'status',r['status'],'J',round(r['objective'],3),'bound',round(r['dual_bound'],3),'gap',r['gap'],flush=True)
            except Exception as exc:
                status['status']='failed';status['errors'].append(dict(instance=i,error=repr(exc)));print('FAIL',i,repr(exc),flush=True)
                for f in pending:f.cancel()
                save();raise
            save()
    status['status']='completed';save();print('COMPLETE',len(status['completed']),'wall_seconds',status['wall_seconds'],flush=True)
if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker':worker((int(sys.argv[2]),sys.argv[3],float(sys.argv[4])))
    else:main()
