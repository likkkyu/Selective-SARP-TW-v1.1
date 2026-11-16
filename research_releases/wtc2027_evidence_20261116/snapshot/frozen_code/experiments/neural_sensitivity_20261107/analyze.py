"""Complete fixed-policy paired sensitivity, raw + hybrid; no selective endpoints."""
from pathlib import Path
import csv,hashlib,json,statistics
import numpy as np
import torch
P=Path(__file__).resolve().parent;ROOT=P.parents[1]


def main():
    out=P/'window_v1';protocol=json.loads((out/'protocol.json').read_text());n=protocol['samples'];assert n==32
    for f,h in protocol['source_sha256'].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h,f
    assert hashlib.sha256(Path(protocol['base_scenario']).read_bytes()).hexdigest()==protocol['base_sha256']
    data=torch.load(protocol['base_scenario'],weights_only=False)['data'];raw=[]
    def extract(method,seed,factor,i,r):
        a=r['audit'];assert not a['violations'] and not a['unfulfilled'];c=r.get('independent_cost',r.get('cost'));assert c is not None
        accepted={node-51 for node in r['actions'] if 51<=node<=100};assert len(accepted)==a['completed']
        types=data[i]['node_type'][:50].tolist();pt=sum(t==1 for t in types);ps=sum(types[j]==1 for j in accepted)
        return dict(method=method,repeat_seed=seed,window_multiplier=factor,instance=i,completed=a['completed'],service_pct=2*a['completed'],objective=c['objective'],non_rejection_objective=c['objective']-c['reject_penalty'],raw_cost=c['raw_cost'],energy_fee=c['energy'],distance_km=c['distance'],passenger_delay_fee=c['passenger_delay_cost'],cargo_delay_fee=c['cargo_delay_cost'],rejection_penalty=c['reject_penalty'],vehicles=a['vehicles'],passenger_total=pt,passenger_completed=ps,cargo_total=50-pt,cargo_completed=a['completed']-ps)
    base=Path(protocol['base_outputs_reused']['neural']);control=Path(protocol['base_outputs_reused']['LNS']);assert json.loads((base.parent/'analysis.json').read_text())['status']=='passed'
    for factor,label in [(.5,'tw05'),(1.,None),(1.5,'tw15')]:
        directory=base if label is None else out/f'{label}_neural'
        assert json.loads((directory/'report.json').read_text())['status']=='completed'
        for spec in protocol['models']:
            d=json.loads((directory/f"learned_{spec['seed']}.json").read_text());assert d['model']==spec and len(d['rows'])==n
            for i,r in enumerate(d['rows']):
                assert r['instance']==i
                raw.append(extract('learned_raw',spec['seed'],factor,i,r));raw.append(extract('learned_hybrid',spec['seed'],factor,i,r['hybrid']))
        if label is None:
            assert json.loads((control/'report.json').read_text())['status']=='completed'
            rows=[json.loads((control/f'instance_{i}.json').read_text())['strong'] for i in range(n)]
        else:
            d=json.loads((out/f'{label}_lns/report.json').read_text());assert d['status']=='completed' and len(d['rows'])==n and d['factor']==factor;rows=d['rows']
        for i,r in enumerate(rows):raw.append(extract('LNS',1234,factor,i,r))
    assert len(raw)==672
    metrics=['service_pct','objective','non_rejection_objective','raw_cost','energy_fee','distance_km','passenger_delay_fee','cargo_delay_fee','rejection_penalty','vehicles']
    methods=['learned_raw','learned_hybrid','LNS'];factors=[.5,1.,1.5];matrix={}
    for method in methods:
        matrix[method]={}
        for factor in factors:
            matrix[method][factor]={}
            for key in metrics:
                values=[]
                for i in range(n):
                    records=[r for r in raw if r['method']==method and r['window_multiplier']==factor and r['instance']==i];assert len(records)==(1 if method=='LNS' else 3)
                    values.append(statistics.mean(r[key] for r in records))
                matrix[method][factor][key]=np.array(values)
    indices=np.random.default_rng(8162049).integers(0,n,(10000,n))
    def describe(x):
        lo,hi=np.quantile(x[indices].mean(axis=1),[.025,.975]);return dict(mean=float(x.mean()),conditional_instance_CI95=[float(lo),float(hi)])
    summary={m:{str(f):{k:describe(v) for k,v in matrix[m][f].items()} for f in factors} for m in methods}
    effects={m:{k:describe(matrix[m][1.5][k]-matrix[m][.5][k]) for k in metrics} for m in methods}
    per_seed=[];classes=[]
    for method in methods[:2]:
        for seed in [1234,4321,8765]:
            scores={f:statistics.mean(r['service_pct'] for r in raw if r['method']==method and r['repeat_seed']==seed and r['window_multiplier']==f) for f in factors}
            assert all(any(r['completed']>0 for r in raw if r['method']==method and r['repeat_seed']==seed and r['window_multiplier']==f) for f in factors),'All-reject regime requires separate diagnosis; preserve records'
            per_seed.append(dict(method=method,training_seed=seed,service_percent_by_factor=scores,wide_minus_narrow_pp=scores[1.5]-scores[.5]))
    for method in methods:
        for f in factors:
            r=[x for x in raw if x['method']==method and x['window_multiplier']==f]
            classes.append(dict(method=method,window_multiplier=f,passenger_service_percent=100*sum(x['passenger_completed'] for x in r)/sum(x['passenger_total'] for x in r),cargo_service_percent=100*sum(x['cargo_completed'] for x in r)/sum(x['cargo_total'] for x in r)))
    report=dict(status='passed',scope=protocol['scope'],independent_paired_instances=32,neural_raw_plans=288,neural_hybrid_plans=288,LNS_plans=96,all_final_plans=672,all_hard_valid=True,unfulfilled=0,summary=summary,wide_minus_narrow=effects,per_training_seed=per_seed,class_coverage=classes,statistics=protocol['statistics'],interpretation='Same request identities and only passenger pickup latest changes. Wider windows cannot worsen the mathematical optimum, but finite raw/hybrid/LNS algorithms may respond nonmonotonically. No real-world causal or optimality claims.')
    (out/'analysis.json').write_text(json.dumps(report,indent=2))
    with (out/'raw_window_plans.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(raw[0]));writer.writeheader();writer.writerows(raw)
    print('PASS',672,'complete plans; wide minus narrow service effects',[(m,effects[m]['service_pct']) for m in methods],flush=True)


if __name__=='__main__':main()
