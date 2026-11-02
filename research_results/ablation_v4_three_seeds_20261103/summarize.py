import json,statistics
from pathlib import Path
p=Path(__file__).parent;rows=json.loads((p/'raw_summary.json').read_text());r={(x['seed'],x['policy']):x['aggregate'] for x in rows}
for metric in ['service_rate_mean','total_cost_raw_mean','distance_mean']:
 a=[r[s,'learned'][metric] for s in [1234,4321,8765]];b=[r[s,'service_first'][metric] for s in [1234,4321,8765]];d=[x-y for x,y in zip(a,b)];scale=100 if metric=='service_rate_mean' else 1
 print(metric,{'learned_mean':statistics.mean(a)*scale,'rule_mean':statistics.mean(b)*scale,'paired_mean_difference':statistics.mean(d)*scale,'paired_sample_sd':statistics.stdev(d)*scale})
