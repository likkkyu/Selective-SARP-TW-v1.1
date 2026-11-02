import os,signal,time,json,datetime
from pathlib import Path
p=Path.cwd();old=2105;child=2106;train=43900
cmd=Path(f'/proc/{old}/cmdline').read_bytes().replace(b'\0',b' ')
assert b'timeout --signal=TERM --kill-after=60 129600 bash remaining.sh' in cmd
assert Path(f'/proc/{child}/status').exists() and Path(f'/proc/{train}/status').exists()
assert b'formal_service_first_8765' in Path(f'/proc/{train}/cmdline').read_bytes()
started=time.time();deadline=started+43200
record={'reason':'User authorized extending this experiment; replace outer timeout only, training unchanged','previous_timeout_pid':old,'remaining_shell_pid':child,'training_pid_at_change':train,'extended_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'new_deadline_utc':datetime.datetime.fromtimestamp(deadline,datetime.timezone.utc).isoformat(),'max_additional_seconds':43200}
(p/'TIME_LIMIT_EXTENSION.json').write_text(json.dumps(record,indent=2))
# Kill only the timeout PID, never its process group or the training child.
os.kill(old,signal.SIGKILL)
for _ in range(100):
 if (p/'exit_status.txt').exists():break
 time.sleep(.1)
if (p/'exit_status.txt').exists():(p/'replaced_timeout_wrapper_exit.txt').write_text((p/'exit_status.txt').read_text());(p/'exit_status.txt').unlink()
assert Path(f'/proc/{train}/status').exists(),'Training unexpectedly stopped'
print('EXTENSION_ACTIVE',json.dumps(record),flush=True)
def active(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]!='Z'
 except FileNotFoundError:return False
while active(child) and time.time()<deadline:time.sleep(30)
if active(child):
 print('NEW_LIMIT_REACHED: terminating experiment process group',flush=True)
 os.killpg(child-1,signal.SIGTERM)
 time.sleep(60)
 if active(child):os.killpg(child-1,signal.SIGKILL)
 status=124
else:
 status=0
 for seed in [4321,8765]:
  for policy in ['learned','service_first']:
   root=p/f'formal_{policy}_{seed}'
   try:
    m=json.loads((root/'manifest.json').read_text());a=json.loads((root/'audit.json').read_text());e=json.loads((root/'eval.json').read_text())
    assert m['status']=='completed' and a['hard_valid']==a['samples']==500 and a['unfulfilled_total']==0
   except Exception as exc:
    print('INCOMPLETE_OR_FAILED',root.name,repr(exc),flush=True);status=1
(p/'exit_status.txt').write_text(str(status)+'\n');record.update(status=status,ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());(p/'EXTENDED_RUN_RESULT.json').write_text(json.dumps(record,indent=2));print('EXTENDED_RUN_EXIT',status,flush=True)
