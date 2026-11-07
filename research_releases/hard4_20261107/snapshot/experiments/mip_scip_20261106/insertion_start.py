"""Deterministic audited insertion start; no neural model, mask or test tuning."""
from scip_selective import lists,scalar_cost,audit_config

def insertion_start(sample,c,profile,vehicles,audit):
    s=lists(sample);n=len(s['loc'])//2;routes=[];accepted=set()
    for order in sorted(range(1,n+1),key=lambda i:(s['time_windows'][i-1][1],i)):
        best=None
        for r in range(len(routes)+(len(routes)<vehicles)):
            old=routes[r] if r<len(routes) else []
            for p in range(len(old)+1):
                first=old[:p]+[order]+old[p:]
                for q in range(p+1,len(first)+1):
                    new=first[:q]+[order+n]+first[q:]
                    a=audit(sample,new+[0],audit_config(c,n,1))
                    if any(v['kind']!='min_dispatch' for v in a['violations']):continue
                    trial=[list(x) for x in routes]
                    if r==len(routes):trial.append(new)
                    else:trial[r]=new
                    actions=[v for route in trial for v in route+[0]]+[2*n+1]*(n-len(accepted)-1)
                    score=scalar_cost(sample,actions,c,profile)['objective']
                    if best is None or score<best[0]:best=(score,trial)
        if best is not None:routes=best[1];accepted.add(order)
    return [v for route in routes for v in route+[0]]+[2*n+1]*(n-len(accepted))
