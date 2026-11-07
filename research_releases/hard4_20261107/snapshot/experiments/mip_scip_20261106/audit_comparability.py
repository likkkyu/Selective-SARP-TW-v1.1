"""Post-run independent protocol audit; never changes a route or model setting."""
import hashlib,json,math,signal,sys,time
from pathlib import Path
from validate import ROOT,P,SNAP,C,PROFILE,Config,torch,MCVRPPDTW,sample_to_batch,audit
from scip_selective import audit_config,scalar_cost
sys.path.insert(0,str(SNAP/'experiments/sensitivity_20261030'))
from completion_guard import install as install_guard
from state_mcvrptw_v2 import StateMCVRPPDTW as State
install_guard()
Q=ROOT/'experiments/paper_delivery_20261104/confirmation_v1';RUN=P/'common_n50_15s_v1';OUT=P/'comparability_audit_v2'

def mask_replay(sample,actions,kwargs):
    state=State.initialize(sample_to_batch(sample),**kwargs);n=len(sample['loc'])//2
    for step,node in enumerate(actions):
        node=int(node)
        if bool(state.all_finished()):
            if any(int(v)!=0 for v in actions[step:]):return dict(valid=False,step=step,node=node,reason='nonpadding action after termination')
            return dict(valid=True,steps=step,padding=len(actions)-step)
        mask,debug=state.get_mask(return_debug=True)
        if bool(mask[0,0,node]):
            tests={}
            for label,kw,extra in [('min_dispatch_1',dict(min_orders_per_dispatch=1),{}),('no_delivery_viability',dict(enable_delivery_viability=False),{}),('skip_commitment',{},dict(skip_pickup_commitment=True)),('min1_no_viability_skip_commitment',dict(min_orders_per_dispatch=1,enable_delivery_viability=False),dict(skip_pickup_commitment=True))]:
                alt=state._replace(**kw).get_mask(**extra);tests[label]=not bool(alt[0,0,node])
            return dict(valid=False,step=step,node=node,previous=int(state.prev_a.item()),state_time=float(state.current_time.item()),trip_start=float(state.trip_start_time.item()),vehicles=float(state.used_vehicles.item()),served_in_trip=int(state.served_orders_since_dispatch.item()),open_orders=int(state.get_open_started_count().item()),counterfactual_action_allowed=tests,debug={k:v.detach().cpu().tolist() if hasattr(v,'detach') else v for k,v in debug.items() if k!='pickup_tw_slack_hours'},prefix=actions[:step+1])
        state=state.update(torch.tensor([node]),current_mask=mask)
    return dict(valid=bool(state.all_finished()),steps=len(actions),reason='finished' if state.all_finished() else 'trace not finished')

def main():
    OUT.mkdir(exist_ok=False);started=time.monotonic()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Comparability audit 360s limit')));signal.alarm(360)
    data_path=ROOT/'experiments/paper_delivery_20261104/confirmation_n50_seed803219.pt';bundle=torch.load(data_path,map_location='cpu',weights_only=False);data=bundle['data'];Config.set_normalization_profile(50,PROFILE)
    prior=json.loads((Q/'plan.json').read_text());new=json.loads((RUN/'plan.json').read_text());deps=json.loads((RUN/'dependencies.json').read_text())
    assert hashlib.sha256(data_path.read_bytes()).hexdigest()==prior['scenario_sha256']==new['scenario_sha256']
    assert all(hashlib.sha256((P/name).read_bytes()).hexdigest()==sha for name,sha in new['source_sha256'].items())
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha for name,sha in deps['dependencies'].items())
    report=dict(scope='post-run parameter/data/cost/physical and neural-state replay audit; not new performance testing',status='running',data_sha256=new['scenario_sha256'],dependency_hashes_match=True,models=[],new_routes=[],reference_state_checks=[])
    def save():report['elapsed_seconds']=time.monotonic()-started;(OUT/'report.json').write_text(json.dumps(report,indent=2))
    save();kwargs=None
    for model in prior['models']:
        label=f"{model['policy']}_{model['seed']}";folder=Q/'models'/label
        cfg=json.loads((folder/'effective_config.json').read_text());scenario=json.loads((folder/'scenario.json').read_text());restore=json.loads((folder/'normalization_restore.json').read_text())
        assert scenario['sha256']==new['scenario_sha256'] and restore['profile']==PROFILE
        assert all(cfg[k]==v for k,v in C.items()),label
        assert all(cfg[k] is True for k in ['HARD_MAX_TRIP_TIME','HARD_OPERATION_END','HARD_PASSENGER_PICKUP_TIMEWINDOW','HARD_PASSENGER_MAX_RIDE_TIME','HARD_VEHICLE_LIMIT'])
        traces=[json.loads(line) for line in (folder/'routes.jsonl').read_text().splitlines()];assert len(traces)==32
        for i,row in enumerate(traces):
            expected={k:v.detach().cpu().tolist() if hasattr(v,'detach') else v for k,v in data[i].items()}
            for k in ('loc','depot','time_windows','node_type','demand_passenger','demand_cargo'):assert row['sample'][k]==expected[k],(label,i,k)
            if kwargs is None:kwargs=row['state_kwargs']
            assert kwargs==row['state_kwargs']
            a=audit(data[i],row['actions'],audit_config(C,50,9));assert a['unfulfilled']==0 and not any(v['kind']!='min_dispatch' for v in a['violations'])
        report['models'].append(dict(model=label,samples=32,data_exact_match=True,parameters_exact_match=True,normalization_exact_match=True,hard_audits=32,state_kwargs=kwargs));save()
        # Positive controls: replay two actual model outputs, all six models.
        for i in (0,1):
            r=mask_replay(data[i],traces[i]['actions'],kwargs);report['reference_state_checks'].append(dict(model=label,instance=i,result=r));save()
        print('PARAMETER/DATA PASS',label,flush=True)
    # Both deterministic insertion and SCIP final actions must be re-audited.
    for i in new['instances']:
        row=json.loads((RUN/f'instance_{i:02d}.json').read_text())
        checks={}
        for kind,actions in [('insertion',row['insertion_actions']),('SCIP',row['scip']['actions'])]:
            a=audit(data[i],actions,audit_config(C,50,9));assert a['unfulfilled']==0 and not any(v['kind']!='min_dispatch' for v in a['violations'])
            replay=scalar_cost(data[i],actions,C,PROFILE)
            with torch.no_grad():j,details=MCVRPPDTW.get_costs(sample_to_batch(data[i]),torch.tensor([actions],dtype=torch.long),return_details=True)
            error=abs(float(j.item())-replay['objective']);rawerr=abs(float(details['total_cost_raw'].item())-replay['raw_cost'])
            assert error<.02 and rawerr<.02
            checks[kind]=dict(audit=a,scalar_cost=replay,tensor_objective_error=error,tensor_raw_error=rawerr)
        # SCIP extracts trips sorted by first-node ID, which may change their
        # concatenation order. Verify unordered trips, not identical sequences.
        def trips(actions):
            routes=[];current=[]
            for v in actions:
                if v==0:
                    if current:routes.append(tuple(current));current=[]
                elif v<=100:current.append(v)
            assert not current
            return sorted(routes)
        checks['same_unordered_trips']=trips(row['insertion_actions'])==trips(row['scip']['actions'])
        checks['neural_state']=mask_replay(data[i],row['scip']['actions'],kwargs)
        checks['insertion_neural_state']=mask_replay(data[i],row['insertion_actions'],kwargs)
        report['new_routes'].append(dict(instance=i,checks=checks));save()
        print('REPLAY',i,'physical PASS','neural_mask',checks['neural_state']['valid'],flush=True)
    report['status']='completed';report['summary']=dict(models=6,model_instance_exact_matches=192,new_physical_routes=64,new_unfulfilled=0,neural_reference_prefixes_passed=sum(x['result']['valid'] for x in report['reference_state_checks']),neural_reference_prefixes_tested=len(report['reference_state_checks']),SCIP_sequences_legal_in_neural_state=sum(x['checks']['neural_state']['valid'] for x in report['new_routes']),SCIP_sequences_blocked=sum(not x['checks']['neural_state']['valid'] for x in report['new_routes']))
    save();signal.alarm(0);print(json.dumps(report['summary']),flush=True)
if __name__=='__main__':main()
