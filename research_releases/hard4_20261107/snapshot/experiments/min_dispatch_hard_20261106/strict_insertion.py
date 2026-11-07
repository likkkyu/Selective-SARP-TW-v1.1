"""Repair static insertion plan before acceptance/execution; reject unplaced IDs."""
from bootstrap import *
from insertion_start import insertion_start

def make_start(sample):
 s=lists(sample);n=len(s['loc'])//2;k=(n+5)//6
 old=insertion_start(sample,C,PROFILE,k,audit);routes=[];cur=[]
 for node in old:
  if node==0:
   if cur:routes.append(cur);cur=[]
  elif node<=2*n:cur.append(node)
 assert not cur
 while any(len(r)//2<4 for r in routes):
  idx=min((i for i,r in enumerate(routes) if len(r)//2<4),key=lambda i:(len(routes[i]),i))
  dropped=routes.pop(idx);orders=[x for x in dropped if x<=n]
  for order in orders:
   best=None
   for r,route in enumerate(routes):
    for p in range(len(route)+1):
     first=route[:p]+[order]+route[p:]
     for q in range(p+1,len(first)+1):
      new=first[:q]+[order+n]+first[q:];a=audit(sample,new+[0],audit_config(C,n,1))
      if any(v['kind']!='min_dispatch' for v in a['violations']):continue
      trial=[x if j!=r else new for j,x in enumerate(routes)];served=sum(len(x)//2 for x in trial)
      actions=[v for route2 in trial for v in route2+[0]]+[2*n+1]*(n-served)
      score=scalar_cost(sample,actions,C,PROFILE)['objective']
      if best is None or score<best[0]:best=(score,r,new)
   if best is not None:routes[best[1]]=best[2]
 # Unassigned requests are declined before executing this static plan. Reject
 # tokens at the end only serialize IDs for the legacy earliest-remaining audit.
 accepted={node for r in routes for node in r if node<=n};rejected=set(range(1,n+1))-accepted
 actions=[v for r in routes for v in r+[0]]+[2*n+1]*len(rejected)
 a=audit(sample,actions,audit_config(C,n,k));assert not a['violations'] and a['unfulfilled']==0,a
 return dict(actions=actions,accepted_order_ids=sorted(accepted),declined_before_execution_order_ids=sorted(rejected),audit=a,completed_per_trip=[len(r)//2 for r in routes],cost=scalar_cost(sample,actions,C,PROFILE),scope='static plan finalized before vehicle execution; rejected IDs were never picked up')
