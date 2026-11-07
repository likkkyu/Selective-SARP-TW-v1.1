"""Isolated candidate-clock fix. Install before completion_guard, never edit snapshot."""
import difflib,hashlib,inspect,json,textwrap
from pathlib import Path
VERSION=1
P=Path(__file__).resolve().parent

def install():
    import state_mcvrptw_v2 as module
    State=module.StateMCVRPPDTW
    if getattr(State,'_candidate_dispatch_clock_version',None)==VERSION:return
    if hasattr(State,'_before_completion_guard'):raise RuntimeError('Install clock patch BEFORE completion_guard')
    original=State.get_mask
    old=textwrap.dedent(inspect.getsource(original));new=old
    before='trip_start_after_pickup = torch.where(self.prev_a == 0, self.current_time, self.trip_start_time).squeeze(1)'
    after='''# Candidate-specific departure: depot waiting is outside the trip.
        candidate_departure = torch.maximum(
            self.current_time,
            time_windows_active[:, 1:n_orders + 1, 0] - travel_time[:, 1:n_orders + 1],
        )
        trip_start_after_pickup = torch.where(
            self.prev_a == 0, candidate_departure, self.trip_start_time.expand_as(candidate_departure)
        )'''
    replacements=[(before,after,1),
      ('candidate_tensor = candidate_indices','candidate_tensor = candidate_indices\n                candidate_trip_start = batch_trip_start_after_pickup[candidate_tensor]',1),
      ('candidate_finish_with_return - batch_trip_start_after_pickup','candidate_finish_with_return - candidate_trip_start',1),
      ('grouped_finish_with_return_lb - batch_trip_start_after_pickup','grouped_finish_with_return_lb - candidate_trip_start[candidate_indices_by_pos]',1),
      ('for candidate_pos, candidate_idx in enumerate(candidate_index_list):','for candidate_pos, candidate_idx in enumerate(candidate_index_list):\n                    candidate_trip_start_time = batch_trip_start_after_pickup[candidate_idx]',1),
      ('batch_trip_start_after_pickup,','candidate_trip_start_time,',3)]
    for a,b,count in replacements:
        if new.count(a)!=count:raise RuntimeError(f'Frozen source drift: expected {count} occurrences of {a!r}, got {new.count(a)}')
        new=new.replace(a,b)
    dest=P/'generated_get_mask_v1.py'
    if dest.exists():
        if dest.read_text()!=new:raise RuntimeError('Refuse overwrite different generated patch')
    else:dest.write_text(new)
    ns=dict(vars(module));exec(compile(new,str(dest),'exec'),ns)
    State._before_candidate_dispatch_clock=original;State.get_mask=ns['get_mask'];State._candidate_dispatch_clock_version=VERSION
    record=dict(version=VERSION,base_module=str(Path(inspect.getfile(original)).resolve()),base_module_sha256=hashlib.sha256(Path(inspect.getfile(original)).read_bytes()).hexdigest(),base_method_sha256=hashlib.sha256(old.encode()).hexdigest(),patched_method_sha256=hashlib.sha256(new.encode()).hexdigest(),scope='get_mask candidate dispatch clock only; no update/cost/weights/min_dispatch/viability change')
    (P/'PATCH_MANIFEST.json').write_text(json.dumps(record,indent=2));(P/'clock_patch.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='frozen/get_mask',tofile='isolated/get_mask_clock_v1')))
