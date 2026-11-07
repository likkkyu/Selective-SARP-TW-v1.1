"""Complete-case paired-instance analysis. Preserve all frozen arms and seeds."""
from pathlib import Path
import csv,hashlib,json,math
import numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parents[1]


def main():
    out=P/'confirmation_v1';plan=json.loads((out/'plan.json').read_text());n=plan['samples']
    for f,h in plan['source_sha256'].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h,f
    assert hashlib.sha256(Path(plan['scenario']).read_bytes()).hexdigest()==plan['scenario_sha256']
    groups={};valid=0;seeds=[]
    assert json.loads((out/'neural/report.json').read_text())['status']=='completed'
    assert json.loads((out/'controls/report.json').read_text())['status']=='completed'
    for spec in plan['models']:
        d=json.loads((out/'neural'/f"{spec['policy']}_{spec['seed']}.json").read_text());assert d['model']==spec and len(d['rows'])==n
        for phase in ['raw','hybrid']:
            label=f"{spec['policy']}_{phase}";values=[]
            for i,r in enumerate(d['rows']):
                assert r['instance']==i
                if phase=='hybrid':
                    r=r['hybrid'];j=r['cost']['objective'];sec=r['integrated_algorithm_seconds']
                else:j=r['objective'];sec=r['inference_seconds']
                assert not r['audit']['violations'] and not r['audit']['unfulfilled'] and math.isfinite(j) and math.isfinite(sec)
                values.append([r['audit']['completed']*2.,j,sec]);valid+=1
            groups.setdefault(label,[]).append(np.array(values))
            seeds.append(dict(method=label,training_seed=spec['seed'],service_percent=float(np.mean(values,axis=0)[0]),mean_J=float(np.mean(values,axis=0)[1])))
    for phase in ['initial','short','strong']:
        values=[]
        for i in range(n):
            r=json.loads((out/'controls'/f'instance_{i}.json').read_text())[phase]
            assert not r['audit']['violations'] and not r['audit']['unfulfilled']
            sec=r['algorithm_seconds'] if phase=='initial' else r['integrated_algorithm_seconds']
            values.append([r['audit']['completed']*2.,r['cost']['objective'],sec]);valid+=1
        groups[f'insertion_{phase}']=[np.array(values)]
    means={k:np.mean(v,axis=0) for k,v in groups.items()}
    rng=np.random.default_rng(8162049);indices=rng.integers(0,n,size=(10000,n));comparisons=[]
    for a,b in [('learned_raw','service_first_raw'),('learned_hybrid','service_first_hybrid'),('learned_hybrid','insertion_short'),('learned_hybrid','insertion_strong'),('learned_raw','insertion_initial')]:
        delta=means[a]-means[b];boot=delta[indices].mean(axis=1)
        comparisons.append(dict(A=a,B=b,interpretation='A minus B; positive service favors A, negative J/time favors A',estimate=delta.mean(axis=0).tolist(),conditional_instance_bootstrap_CI95=np.quantile(boot,[.025,.975],axis=0).T.tolist()))
    assert valid==480
    summary=[dict(method=k,service_percent=float(v[:,0].mean()),mean_J=float(v[:,1].mean()),mean_algorithm_seconds=float(v[:,2].mean())) for k,v in means.items()]
    report=dict(status='passed',samples=n,fully_audited_plans=valid,columns=['service_percentage_points','objective_J','algorithm_seconds'],summary=summary,seed_specific=seeds,comparisons=comparisons,protocol=plan['scope'],replication=plan['statistics'],search_replication=plan['search_replication'],not_pure_training_ablation=True)
    (out/'analysis.json').write_text(json.dumps(report,indent=2))
    with (out/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    for row in summary:print(row)
    print('ALL',valid,'plans complete, audited and retained')


if __name__=='__main__':main()
