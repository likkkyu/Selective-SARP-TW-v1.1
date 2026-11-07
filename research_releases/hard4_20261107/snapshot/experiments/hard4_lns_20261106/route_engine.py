"""Independent scalar route evaluator. No neural masks or learned parameters.

Route costs are additive except cargo delay: sum minutes BEFORE tier pricing.
Each engine owns one immutable instance/config and its route cache.
"""
import math
from collections import OrderedDict


class RouteEngine:
    def __init__(self, sample, config, profile, vehicles=None):
        self.s = {k: v.detach().cpu().tolist() if hasattr(v, 'detach') else v for k,v in sample.items()}
        self.c = dict(config); self.profile = dict(profile)
        self.n = len(self.s['loc'])//2; self.k = vehicles or math.ceil(self.n/6)
        xy = [self.s['depot']] + self.s['loc']
        self.distance = [[math.dist(a,b)*config['AREA_SIZE'] for b in xy] for a in xy]
        self.cache = OrderedDict(); self.evaluations = 0; self.hits = 0

    def route(self, nodes):
        key = tuple(nodes)
        if key in self.cache:
            self.hits += 1; self.cache.move_to_end(key); return self.cache[key]
        self.evaluations += 1
        result = self._route(key)
        self.cache[key] = result
        if len(self.cache)>100000: self.cache.popitem(last=False)
        return result

    def _route(self, nodes):
        if not nodes: return dict(orders=(), energy=0., distance=0., passenger_minutes=0., cargo_minutes=0.)
        c=self.c; s=self.s; n=self.n; tol=1e-4
        if any(not isinstance(x,int) or not 1<=x<=2*n for x in nodes): return None
        t=c['OPERATION_START']; depart=max(t,s['time_windows'][nodes[0]-1][0]-self.distance[0][nodes[0]]/c['VEHICLE_SPEED'])
        t=depart; cur=0; lp=lc=energy=distance=pd=cd=0.; picked=set(); done=set(); opened={}
        for node in nodes:
            j=node-1; order=j%n+1; pickup=node<=n
            km=self.distance[cur][node]; arrival=t+km/c['VEHICLE_SPEED']
            energy+=km*c['ENERGY_BASE']*(1+(lp*c['PASSENGER_CAPACITY']*c['PASSENGER_WEIGHT_KG']+lc*c['CARGO_CAPACITY']*c['CARGO_UNIT_WEIGHT_KG'])/c['ENERGY_WEIGHT_REF'])*c['ELECTRICITY_PRICE']; distance+=km
            start=max(arrival,s['time_windows'][j][0]); finish=start+c['SERVICE_TIME']
            if pickup:
                if order in picked: return None
                if (s['node_type'][j]==1 or c['HARD_CARGO_PICKUP_TIMEWINDOW']) and start>s['time_windows'][j][1]+tol: return None
                picked.add(order); opened[order]=finish
            else:
                if order not in opened or order in done: return None
                if s['node_type'][j]==1:
                    ride=start-opened[order]
                    if ride>c['PASSENGER_MAX_RIDE_TIME_MINUTES']/60+tol or ride-self.distance[order][node]/c['VEHICLE_SPEED']>c['PASSENGER_MAX_EXCESS_RIDE_TIME_MINUTES']/60+tol: return None
                    pd+=max(0.,arrival-s['time_windows'][j][1])*60
                else: cd+=max(0.,arrival-s['time_windows'][j][1])*60
                del opened[order]; done.add(order)
            lp+=s['demand_passenger'][j]; lc+=s['demand_cargo'][j]
            if not -tol<=lp<=1+tol or not -tol<=lc<=1+tol or len(opened)>6 or finish>c['OPERATION_END']+tol: return None
            cur=node; t=finish
        if opened or picked!=done: return None
        km=self.distance[cur][0]; returned=t+km/c['VEHICLE_SPEED']
        if returned>c['OPERATION_END']+tol or returned-depart>c['MAX_TRIP_TIME']+tol: return None
        energy+=km*c['ENERGY_BASE']*(1+(lp*c['PASSENGER_CAPACITY']*c['PASSENGER_WEIGHT_KG']+lc*c['CARGO_CAPACITY']*c['CARGO_UNIT_WEIGHT_KG'])/c['ENERGY_WEIGHT_REF'])*c['ELECTRICITY_PRICE']; distance+=km
        return dict(orders=tuple(sorted(done)),energy=energy,distance=distance,passenger_minutes=pd,cargo_minutes=cd)

    def cost(self, routes, strict=True):
        active=[tuple(r) for r in routes if r]; metrics=[self.route(r) for r in active]
        if len(active)>self.k or any(m is None for m in metrics): return None
        ids=[x for m in metrics for x in m['orders']]
        if len(set(ids))!=len(ids) or (strict and any(len(m['orders'])<4 for m in metrics)): return None
        c=self.c; p=self.profile; energy=sum(m['energy'] for m in metrics)
        cp=sum(m['passenger_minutes'] for m in metrics)*c['PASSENGER_DELAY_COST']; minutes=sum(m['cargo_minutes'] for m in metrics)
        b1=c['CARGO_DELAY_TIER1_MIN']; b2=c['CARGO_DELAY_TIER2_MIN']
        cc=min(minutes,b1)*c['CARGO_DELAY_COST']+max(0.,min(minutes,b2)-b1)*c['CARGO_DELAY_COST_30_60']+max(0.,minutes-b2)*c['CARGO_DELAY_COST_60_PLUS']
        vc=len(active)*c['VEHICLE_COST']; rp=(self.n-len(ids))*c['ALPHA_REJECT']
        objective=c['ALPHA_ENERGY']*energy/p['energy_cost_raw']+c['ALPHA_DELAY']*(cp/p['passenger_delivery_delay_cost_raw']+cc/p['cargo_delay_cost_raw'])+c['ALPHA_VEHICLE']*vc/max(1.,p['vehicle_cost_norm'])+rp
        return dict(objective=objective,raw_cost=energy+cp+cc+vc+rp,energy=energy,distance=sum(m['distance'] for m in metrics),passenger_delay_cost=cp,cargo_delay_cost=cc,vehicle_cost=vc,reject_penalty=rp,completed=len(ids),vehicles=len(active))

    def actions(self, routes):
        served=sum(len(r)//2 for r in routes)
        return [x for r in routes if r for x in list(r)+[0]]+[2*self.n+1]*(self.n-served)


def split_actions(actions,n):
    routes=[]; cur=[]
    for node in actions:
        if node==0:
            if cur: routes.append(tuple(cur)); cur=[]
        elif node<=2*n: cur.append(node)
    if cur: raise ValueError('Unclosed route')
    return routes
