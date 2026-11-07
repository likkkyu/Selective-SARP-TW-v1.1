"""Verify real-action relaxation only at depot on observed development trajectories."""
import json,signal,time
from bootstrap import *
from clock_patch import install

def main():
 out=P/'mask_scope_v1';out.mkdir(exist_ok=False);signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('180s mask scope budget')));signal.alarm(180)
 old=State.get_mask;install();fixed_raw=State.get_mask
 from completion_guard import install as guard
 guard()
 data=torch.load(ROOT/'experiments/paper_delivery_20261104/development_n50_seed614729.pt',map_location='cpu',weights_only=False)['data'];rows=[];start=time.monotonic()
 for mode in ['old','fixed']:
  for path in sorted((P/f'{mode}_dev4_v1').glob('*.json')):
   if path.name=='report.json':continue
   trace=json.loads(path.read_text())
   for row in trace['rows']:
    state=State.initialize(sample_to_batch(data[row['instance']]),**KW)
    for step,node in enumerate(row['actions']):
     if state.all_finished():break
     if step%10==0 or int(state.prev_a.item())==0:
      a=old(state);b=fixed_raw(state);depot=int(state.prev_a.item())==0
      if depot:assert not (b[:,:,1:101]&~a[:,:,1:101]).any(),(mode,path.name,row['instance'],step)
      else:assert torch.equal(a,b),(mode,path.name,row['instance'],step)
      rows.append(dict(mode=mode,model=path.stem,instance=row['instance'],step=step,at_depot=depot,old_service_blocked=int(a[:,:,1:101].sum()),new_service_blocked=int(b[:,:,1:101].sum()),inroute_full_mask_equal=bool(torch.equal(a,b)) if not depot else None))
     mask=state.get_mask();assert not bool(mask[0,0,node]),(mode,path.name,row['instance'],step,node)
     state=state.update(torch.tensor([node]),current_mask=mask)
 summary=dict(status='passed',states_checked=len(rows),inroute_states=sum(not r['at_depot'] for r in rows),depot_states=sum(r['at_depot'] for r in rows),all_observed_inroute_masks_identical=True,no_previously_legal_service_removed_at_depot=True,old_and_fixed_development_traces_executable_in_fixed_environment=True,seconds=time.monotonic()-start,scope='observed trajectories and sampled states; not proof for all inputs')
 (out/'report.json').write_text(json.dumps(dict(summary=summary,rows=rows),indent=2));signal.alarm(0);print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
