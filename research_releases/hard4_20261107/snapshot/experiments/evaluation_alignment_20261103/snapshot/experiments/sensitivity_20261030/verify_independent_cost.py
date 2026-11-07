import json
from pathlib import Path
from independent_cost import replay
p=Path(__file__).parent;existing=json.loads((p/'timing_replay.json').read_text());report={}
for folder in (p/'results_1234/learned_1234').iterdir():
 if not (folder/'routes.jsonl').exists():continue
 c=json.loads((folder/'effective_config.json').read_text());values=[replay(r['sample'],r['actions'],c) for r in map(json.loads,(folder/'routes.jsonl').read_text().splitlines())];means={k:sum(x[k] for x in values)/len(values) for k in values[0]}
 diffs={k:means[k]-existing[folder.name]['candidate'][k] for k in means}
 report[folder.name]=dict(scalar=means,candidate_minus_reference={k:-v for k,v in diffs.items()})
 assert all(abs(v)<.01 for v in diffs.values()),(folder.name,diffs)
(p/'independent_cost_check.json').write_text(json.dumps(report,indent=2));print('PASS',len(report),'scenarios; independent scalar means match candidate within 0.01 units')
