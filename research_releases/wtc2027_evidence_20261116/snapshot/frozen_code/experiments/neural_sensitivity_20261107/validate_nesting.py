"""Physical-domain nesting test, not inference or evidence of monotone heuristics."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2];P=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/hard4_confirmation_20261106'))
from common import *
from evaluate_baselines import verify
from route_engine import RouteEngine,split_actions


def main():
    out=P/'window_v1';protocol=json.loads((out/'protocol.json').read_text())
    assert C==protocol['physical_config'] and PROFILE==protocol['normalization_profile']
    base=torch.load(protocol['base_scenario'],weights_only=False)['data'];narrow=torch.load(out/'tw05.pt',weights_only=False)['data'];wide=torch.load(out/'tw15.pt',weights_only=False)['data'];Config.set_normalization_profile(50,PROFILE)
    neural=Path(protocol['base_outputs_reused']['neural']);control=Path(protocol['base_outputs_reused']['LNS']);plans=[]
    for spec in protocol['models']:
        d=json.loads((neural/f"learned_{spec['seed']}.json").read_text())
        for i,r in enumerate(d['rows']):
            for stage,row in [('raw',r),('hybrid',r['hybrid'])]:plans.append((f"learned_{stage}_{spec['seed']}",i,row['actions']))
    for i in range(32):plans.append(('LNS',i,json.loads((control/f'instance_{i}.json').read_text())['strong']['actions']))
    assert len(plans)==224;rows=[]
    for method,i,actions in plans:
        routes=split_actions(actions,50);engine=RouteEngine(base[i],C,PROFILE);original=engine.cost(routes);assert original is not None
        # Reject token selects an order by deadline. Canonicalize the SAME accepted
        # set as complete tours followed by rejection, avoiding deadline-induced
        # changes to token identity when windows change.
        canonical=engine.actions(routes);w=verify(wide[i],canonical)
        assert abs(w['cost']['objective']-original['objective'])<1e-8
        nc=RouteEngine(narrow[i],C,PROFILE).cost(routes)
        rows.append(dict(method=method,instance=i,wide_same_acceptance_and_routes_valid=True,wide_objective_difference=w['cost']['objective']-original['objective'],narrow_original_routes_valid=nc is not None))
    report=dict(status='passed',scope='224 fixed baseline plans under changed windows; not new policy output, not optimality or causal proof.',wide_hard_valid=224,narrow_original_routes_valid=sum(r['narrow_original_routes_valid'] for r in rows),wide_same_cost=True,canonicalization='Identical accepted IDs and pickup/delivery tours, followed by all unaccepted requests rejected before physical execution.',rows=rows)
    (out/'NESTED_FEASIBILITY_CHECK.json').write_text(json.dumps(report,indent=2));print('PASS widened fixed plans',224,'/224; original routes still valid under narrow:',report['narrow_original_routes_valid'],flush=True)


if __name__=='__main__':main()
