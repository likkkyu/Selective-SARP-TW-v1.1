"""Analytic two-trip cargo-tier regression. Run after timed window work ends."""
from pathlib import Path
import hashlib,json,math,signal,sys,time
ROOT=Path(__file__).resolve().parents[3];P=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/hard4_confirmation_20261106'))
from common import *
from scip_boundary_v2 import BoundarySafeSCIP


def main():
    busy=ROOT/'experiments/neural_sensitivity_20261107/window_v1'
    assert (busy/'exit_status.txt').exists(),'Wait for timed sensitivity to terminate; do not compete for CPU'
    assert (busy/'exit_status.txt').read_text().strip()=='0','Review sensitivity failure first'
    out=P/'evidence';out.mkdir(exist_ok=False);start=time.monotonic()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('60s analytic multi-trip cap')));signal.alarm(60)
    n=8;loc=[[0.,0.]]*4+[[1.,1.]]*4+[[1.,0.]]*4+[[0.,1.]]*4
    sample=dict(depot=torch.tensor([.5,.5]),loc=torch.tensor(loc),time_windows=torch.tensor([[14.,14.16]]*8+[[14.4,14.4]]*8),node_type=torch.zeros(16,dtype=torch.long),demand_passenger=torch.zeros(16),demand_cargo=torch.tensor([.05]*8+[-.05]*8),n_orders=torch.tensor(8))
    actions=[1,2,3,4,9,10,11,12,0,5,6,7,8,13,14,15,16,0]
    plan=dict(scope='Analytic two-trip representation and global cargo-tier regression; not a performance benchmark, not an extra neural seed or full enumeration of N8',physical_config=C,normalization_profile=PROFILE,cold_seconds=5,warm_seconds=2,total_timeout_seconds=60,source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),ROOT/'experiments/hard4_confirmation_20261106/scip_boundary_v2.py',ROOT/'experiments/min_dispatch_hard_20261106/strict_scip.py',ROOT/'experiments/mip_scip_20261106/scip_selective.py']})
    (out/'plan.json').write_text(json.dumps(plan,indent=2));torch.save(sample,out/'case.pt')
    try:
        Config.set_normalization_profile(8,dict(PROFILE,graph_size=8));a=audit(sample,actions,audit_config(C,8,2));assert not a['violations'] and not a['unfulfilled'] and a['completed']==8 and a['vehicles']==2
        scalar=scalar_cost(sample,actions,C,PROFILE)
        s=lists(sample);cross=math.dist(s['loc'][0],s['loc'][4])*C['AREA_SIZE']/C['VEHICLE_SPEED'];width=float(s['time_windows'][0][1])-14.;assert cross>width
        # Both groups must use separate trips: their simultaneous pickup windows
        # cannot be connected. Four sequential pickups precede the 10km drive;
        # early delivery-and-return would miss the remaining pickup window.
        drive=C['AREA_SIZE']/C['VEHICLE_SPEED'];assert 2*drive>width
        first_arrival=14.+4*C['SERVICE_TIME']+drive
        minutes_per_group=sum(max(0.,first_arrival+j*C['SERVICE_TIME']-s['time_windows'][8+j][1])*60 for j in range(4))
        minutes=2*minutes_per_group;b1=C['CARGO_DELAY_TIER1_MIN'];b2=C['CARGO_DELAY_TIER2_MIN']
        tier=lambda m:min(m,b1)*C['CARGO_DELAY_COST']+max(0.,min(m,b2)-b1)*C['CARGO_DELAY_COST_30_60']+max(0.,m-b2)*C['CARGO_DELAY_COST_60_PLUS']
        global_cargo=tier(minutes);wrong_per_trip=2*tier(minutes_per_group);assert abs(global_cargo-wrong_per_trip)>10
        depot_leg=math.dist(s['depot'],s['loc'][0])*C['AREA_SIZE'];payload=4*s['demand_cargo'][0]*C['CARGO_CAPACITY']*C['CARGO_UNIT_WEIGHT_KG']
        energy=2*C['ENERGY_BASE']*C['ELECTRICITY_PRICE']*(2*depot_leg+C['AREA_SIZE']*(1+payload/C['ENERGY_WEIGHT_REF']))
        optimum=C['ALPHA_ENERGY']*energy/PROFILE['energy_cost_raw']+C['ALPHA_DELAY']*global_cargo/PROFILE['cargo_delay_cost_raw']+C['ALPHA_VEHICLE']*2*C['VEHICLE_COST']/PROFILE['vehicle_cost_norm']
        assert optimum<575 and abs(scalar['objective']-optimum)<1e-8 and abs(scalar['cargo_delay_cost']-global_cargo)<1e-8
        with torch.no_grad():j,d=MCVRPPDTW.get_costs(sample_to_batch(sample),torch.tensor([actions]),return_details=True)
        assert abs(float(j)-optimum)<.02 and abs(float(d['total_cost_raw'])-scalar['raw_cost'])<.02
        cold=BoundarySafeSCIP(C,PROFILE,vehicles=2,time_limit=5,threads=1,verbose=False).solve(sample,logfile=out/'cold.log')
        warm=BoundarySafeSCIP(C,PROFILE,vehicles=2,time_limit=2,threads=1,verbose=False).solve(sample,warm_actions=actions,logfile=out/'oracle_warm.log')
        for result in [cold,warm]:
            assert result['actions'] is not None and result['objective']>=optimum-.02 and result['dual_bound']<=optimum+.02
            assert not result['original_instance_audit']['violations'] and not result['original_instance_audit']['unfulfilled']
        assert abs(warm['objective']-optimum)<.02
        report=dict(status='passed',scope=plan['scope'],analytic_proof='Pickup groups separated by more than pickup-window width; hence each trip can accept only one four-order group. Hard4 permits accepted totals 0/4/8. The feasible all-eight plan costs less than a single rejection, so all eight are optimal. Earliest four pickups then direct travel and four deliveries minimize lateness, travel and dispatch for each group; symmetric coordinates/demands make within-group permutations equivalent.',analytic_objective=optimum,global_cargo_minutes=minutes,global_cargo_fee=global_cargo,incorrect_sum_per_trip_tier_fee=wrong_per_trip,actions=actions,audit=a,scalar_cost=scalar,tensor_objective_error=abs(float(j)-optimum),cold_solver=cold,oracle_warm_solver=warm,warm_objective_error=abs(warm['objective']-optimum),seconds=time.monotonic()-start)
        (out/'report.json').write_text(json.dumps(report,indent=2));print('PASS two-trip global tier regression',global_cargo,'versus incorrect',wrong_per_trip,'analytic J',optimum,flush=True)
    except BaseException as exc:
        (out/'failure.json').write_text(json.dumps(dict(status='failed',error=repr(exc)),indent=2));raise
    finally:signal.alarm(0)


if __name__=='__main__':main()
