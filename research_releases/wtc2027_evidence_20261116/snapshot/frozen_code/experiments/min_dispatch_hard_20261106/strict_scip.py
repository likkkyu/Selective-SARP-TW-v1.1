"""User-confirmed >=4 completed requests on EVERY dispatched trip."""
import hashlib,inspect,textwrap,time,math
from pathlib import Path
from bootstrap import SelectiveSCIP
P=Path(__file__).resolve().parent
class HardMinDispatchSCIP(SelectiveSCIP):
 def build(self,sample):
  m=super().build(sample);begin=time.monotonic();n=self.n
  served={i:m.addVar(f'completed_since_dispatch_{i}',lb=0,ub=n) for i in range(1,2*n+1)}
  for (i,j),v in self.x.items():
   if i==0:
    # First node is a pickup; no request has yet been completed.
    m.addCons(served[j]<=n*(1-v))
   elif j==0:m.addCons(served[i]>=4*v)
   else:
    delta=served[j]-served[i]-int(j>n)
    m.addCons(delta<=(n+1)*(1-v));m.addCons(delta>=-(n+1)*(1-v))
  self.variables['trip_completed']=served;self.build_seconds+=time.monotonic()-begin
  self.original_variables=m.getNVars();self.original_constraints=m.getNConss();return m
 def solve(self,*args,**kwargs):
  result=super().solve(*args,**kwargs);result['min_dispatch_is_hard']=True;result['min_dispatch']=4
  result['physical_constraints']='all dispatched trips complete >=4 requests, including final trip; no residual; max_open6; earliest-service timing'
  if result.get('actions') is not None:
   completed=0;counts=[]
   for node in result['actions']:
    if node==0:
     counts.append(completed);completed=0
    elif self.n<node<=2*self.n:completed+=1
   if completed or any(x<4 for x in counts):raise RuntimeError('Extracted trip violates hard minimum')
   result['completed_per_trip']=counts
  return result
# Reuse the existing complete primal mapping, extending only its new count vars.
source=textwrap.dedent(inspect.getsource(SelectiveSCIP.add_route_start))
anchor="    if not m.checkSol(sol,printreason=False,completely=True):"
assert source.count(anchor)==1
extra='''    completed_since_dispatch=0
    for node in actions:
        if node==0:
            completed_since_dispatch=0
        elif 1<=node<=2*n:
            completed_since_dispatch+=int(node>n)
            m.setSolVal(sol,v['trip_completed'][node],completed_since_dispatch)
'''
source=source.replace(anchor,extra+anchor)
file=P/'generated_hard_warm_start.py'
if file.exists():assert file.read_text()==source
else:file.write_text(source)
namespace=dict(SelectiveSCIP.add_route_start.__globals__)
exec(compile(source,str(file),'exec'),namespace)
HardMinDispatchSCIP.add_route_start=namespace['add_route_start']
