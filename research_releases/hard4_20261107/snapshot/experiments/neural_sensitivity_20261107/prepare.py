"""Exact same-order passenger pickup-window interventions; no regenerated demand."""
from pathlib import Path
import argparse,copy,hashlib,json,time
import torch
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    parent=ROOT/'experiments/neural_confirmation_20261107/confirmation_v1';mainplan=json.loads((parent/'plan.json').read_text())
    assert json.loads((parent/'FINAL_REVIEW.json').read_text())['status']=='passed'
    if not args.preflight:
        assert json.loads((P/'preflight_v1/REVIEW.json').read_text())['status']=='passed'
    source=ROOT/'experiments/paper_delivery_20261104/development_n50_seed614729.pt' if args.preflight else Path(mainplan['scenario'])
    if not args.preflight:assert sha(source)==mainplan['scenario_sha256']
    base=torch.load(source,map_location='cpu',weights_only=False)
    if args.preflight:base=copy.deepcopy(base);base['data']=base['data'][:1]
    n_samples=len(base['data']);assert n_samples==(1 if args.preflight else 32)
    out=P/('preflight_v1' if args.preflight else 'window_v1');out.mkdir(exist_ok=False)
    models=[m for m in mainplan['models'] if m['policy']=='learned'];assert len(models)==3
    sources=dict(mainplan['source_sha256'])
    for f in P.glob('*.py'):sources[str(f.relative_to(ROOT))]=sha(f)
    for f,h in sources.items():assert sha(ROOT/f)==h,f
    protocol=dict(scope='Paired fixed-policy passenger pickup-window response: same old trained learned checkpoints, CPU float32 certificate v5/native, raw and 0.5s hand-LNS plans. Not retraining or real-world causal inference.',
      created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),preflight=args.preflight,base_scenario=str(source),base_sha256=sha(source),samples=n_samples,models=models,
      factors=[.5,1.,1.5],factor_choice='Previously chosen 0.5/1.5 endpoints; no response-directed factor tuning.',
      base_outputs_reused=dict(neural=str(parent/'neural'),LNS=str(parent/'controls'),LNS_key='strong'),
      source_sha256=sources,physical_config=mainplan['physical_config'],normalization_profile=mainplan['normalization_profile'],
      statistics='Same 32 base-instance IDs across all conditions and checkpoints. Three training seeds separately; fixed-checkpoint instance bootstrap 10000/seed8162049. No p-values or independent-request replication. Raw and hybrid responses both retained.',
      resource_budget=dict(neural_seconds_per_endpoint=1800,LNS_seconds_per_endpoint=1800,total_timeout_upper_bound=7200,expected_CPU_minutes=22,no_GPU=True,no_retraining=True),
      planned_final_plans=dict(neural_raw=288,neural_hybrid=288,LNS=96,total_including_reused_base=672),
      safety='Only passenger pickup latest changes; all hard4 constraints and C4 costs reaudited; no discarded cases or replacement seeds. Finite search may respond nonmonotonically despite nested physical feasibility.',endpoints=[])
    (out/'protocol_before_transform.json').write_text(json.dumps(protocol,indent=2));changes=[]
    for factor,label in [(.5,'tw05'),(1.5,'tw15')]:
        bundle=copy.deepcopy(base);counts=[]
        for i,(old,new) in enumerate(zip(base['data'],bundle['data'])):
            n=len(old['loc'])//2;assert n==50;mask=old['node_type'][:n]==1;width=old['time_windows'][:n,1]-old['time_windows'][:n,0]
            expected=old['time_windows'].clone();expected[:n,1][mask]=torch.minimum(old['time_windows'][:n,0][mask]+factor*width[mask],torch.full_like(width[mask],16.))
            new['time_windows']=expected
            for key in old:
                if key!='time_windows':assert torch.equal(old[key],new[key]) if hasattr(old[key],'shape') else old[key]==new[key]
            assert torch.equal(old['time_windows'][:,0],expected[:,0]) and torch.equal(old['time_windows'][n:],expected[n:])
            assert torch.equal(old['time_windows'][:n][~mask],expected[:n][~mask])
            diff=expected[:n,1]-old['time_windows'][:n,1]
            assert bool((diff[mask]<=0).all()) if factor<1 else bool((diff[mask]>=0).all())
            count=int((diff.abs()>1e-7).sum());assert count>0;counts.append(count)
        bundle['meta'].update(pickup_window_multiplier=factor,pairing='Only passenger pickup latest changes; exact same request IDs and all other fields')
        scenario=out/f'{label}.pt';torch.save(bundle,scenario)
        endpoint=dict(scope=protocol['scope'],scenario=str(scenario.resolve()),scenario_sha256=sha(scenario),samples=n_samples,models=models,source_sha256=sources,window_multiplier=factor,hybrid_search_seed=1234,hybrid_search_seconds=.5,lns_max_iterations=1000,neural_timeout_seconds=1800,lns_timeout_seconds=1800)
        (out/f'{label}_plan.json').write_text(json.dumps(endpoint,indent=2));protocol['endpoints'].append(endpoint)
        changes.append(dict(factor=factor,total_changed=sum(counts),per_instance=counts,other_fields_identical=True,delivery_and_cargo_windows_identical=True))
    (out/'PARAMETER_EFFECT_CHECK.json').write_text(json.dumps(dict(status='passed',rows=changes,scope='Parameter activation only, not effect result'),indent=2))
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));print('PREPARED',n_samples,'same-order cases:',changes,flush=True)


if __name__=='__main__':main()
