"""Analytic upper bound, not an empirical maximum or proof of heuristic optimality."""
from pathlib import Path
import hashlib,json,math
import torch
P=Path(__file__).resolve().parent


def main():
    plan=json.loads((P/'confirmation_v1/plan.json').read_text());c=plan['physical_config'];p=plan['normalization_profile'];n=50;k=9;tol=1e-4
    assert all(c[x]>=0 for x in ['ALPHA_ENERGY','ALPHA_DELAY','ALPHA_VEHICLE','PASSENGER_DELAY_COST','CARGO_DELAY_COST','CARGO_DELAY_COST_30_60','CARGO_DELAY_COST_60_PLUS','VEHICLE_COST','ENERGY_BASE','ELECTRICITY_PRICE'])
    data=torch.load(plan['scenario'],weights_only=False)['data'];assert len(data)==32
    assert hashlib.sha256(Path(plan['scenario']).read_bytes()).hexdigest()==plan['scenario_sha256']
    for s in data:
        assert len(s['loc'])==100
        assert bool(((s['loc']>=0)&(s['loc']<=1)).all()) and bool(((s['depot']>=0)&(s['depot']<=1)).all())
        assert bool((s['time_windows'][n:,1]>=c['OPERATION_START']).all())
    max_leg=c['AREA_SIZE']*math.sqrt(2)
    edges=2*n+k # m distinct visited nodes across K closed trips imply m+K edges.
    max_payload=(1+tol)*(c['PASSENGER_CAPACITY']*c['PASSENGER_WEIGHT_KG']+c['CARGO_CAPACITY']*c['CARGO_UNIT_WEIGHT_KG'])
    energy=c['ALPHA_ENERGY']*edges*max_leg*c['ENERGY_BASE']*(1+max_payload/c['ENERGY_WEIGHT_REF'])*c['ELECTRICITY_PRICE']/p['energy_cost_raw']
    max_delay=(c['OPERATION_END']+tol-c['OPERATION_START'])*60
    passenger=c['ALPHA_DELAY']*n*max_delay*c['PASSENGER_DELAY_COST']/p['passenger_delivery_delay_cost_raw']
    rate=max(c['CARGO_DELAY_COST'],c['CARGO_DELAY_COST_30_60'],c['CARGO_DELAY_COST_60_PLUS'])
    cargo=c['ALPHA_DELAY']*n*max_delay*rate/p['cargo_delay_cost_raw']
    vehicle=c['ALPHA_VEHICLE']*k*c['VEHICLE_COST']/max(1.,p['vehicle_cost_norm'])
    overtime=c['ALPHA_TRIP_OVERTIME']*k*tol # Numerical tolerance allowance; exact feasibility gives zero.
    upper=energy+passenger+cargo+vehicle+overtime;assert upper<c['ALPHA_REJECT']
    report=dict(status='passed',scope='Sufficient analytic bound for N50 instances in the declared domain; not observed maximum, not arbitrary N/parameters, not proof of greedy-policy optimality.',
      conditions=['Unit-square locations and depot','All delivery deadlines at or after operation start','Nonnegative objective weights','Each pickup/delivery node visited at most once','At most nine closed trips','Every accepted order completed; no unfulfilled requests','All audited physical limits; at most 1e-4h numeric end/trip tolerance'],
      checked_input_instances=32,scenario_sha256=plan['scenario_sha256'],n=n,k=k,
      conservative_components=dict(normalized_energy=energy,normalized_passenger_delay=passenger,normalized_cargo_delay=cargo,normalized_dispatch=vehicle,numerical_overtime_allowance=overtime),
      non_rejection_objective_upper_bound=upper,rejection_penalty=c['ALPHA_REJECT'],margin=c['ALPHA_REJECT']-upper,
      proof='Write J=575r+G, 0<=G<=U<575. If r_A<=r_B-1 then J_A-J_B<=-575+U<0. Thus every globally optimal feasible solution first minimizes rejection count and then minimizes G. Bounding all 50 orders as passengers AND as cargo double-counts alternatives intentionally, making the upper bound conservative.',
      implication='Current objective is lexicographic service coverage then operational penalties within this domain, not an economic price-vs-acceptance tradeoff. Feasible route grouping and order decisions still matter; service-first greedy need not be optimal.',
      training_decision='Do not lower rejection penalty based on new test outcomes or relabel old weights. A new economic-tradeoff study would need a separately motivated objective, conditioning/training design and independent protocol.')
    (P/'OBJECTIVE_DOMINANCE_AUDIT.json').write_text(json.dumps(report,indent=2));print('Conservative non-rejection J bound:',upper,'< penalty',c['ALPHA_REJECT'],'margin',c['ALPHA_REJECT']-upper)


if __name__=='__main__':main()
