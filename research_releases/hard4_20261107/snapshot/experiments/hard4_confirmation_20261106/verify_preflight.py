"""Audit existing preflight observations, including terminal-only batch padding."""
from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent;ROOT=P.parents[1]


def main():
 report=json.loads((P/'neural_entry_preflight_v1/report.json').read_text());assert report['status']=='completed' and len(report['models'])==6
 checks=[]
 for spec in report['models']:
  label=spec['model'];f=P/'neural_entry_preflight_v1'/f'{label}.json';row=json.loads(f.read_text())['rows'][0]
  old=json.loads((ROOT/'experiments/hard4_admission_20261106/dev4_v1'/f'{label}.json').read_text())['rows'][0]
  a=row['actions'];b=old['actions'];short,long=(a,b) if len(a)<=len(b) else (b,a)
  equivalent=long[:len(short)]==short and all(x==0 for x in long[len(short):])
  assert equivalent and row['reference_objective_difference']==0 and not row['audit']['violations'] and not row['audit']['unfulfilled']
  checks.append(dict(model=label,exact_actions_equal=a==b,equal_up_to_terminal_depot_padding=equivalent,objective_difference=row['reference_objective_difference'],observation_sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
 mip=json.loads((P/'scip_boundary_validation_v1/report.json').read_text());assert mip['status']=='passed' and len(mip['rows'])==7 and mip['invalid_three_plus_five_warm_start_rejected']
 (P/'PREFLIGHT_REVIEW.json').write_text(json.dumps(dict(status='passed',neural_scope='all six checkpoints, first fixed development instance, batch1; no universal equivalence claim',neural_checks=checks,scip_scope='six structured boundaries, real N50 warm start and hard4 rejection; arbitrary pre-operation soft delivery deadlines explicitly refused'),indent=2))
 print('PASS six-model entry: same actual actions/J, only trailing depot padding differs; SCIP boundary regressions passed')
if __name__=='__main__':main()
