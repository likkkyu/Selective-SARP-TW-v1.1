from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).parent
raw=json.loads((p/'sensitivity_summary.json').read_text());r={x['scenario']:x for x in raw}
with (p/'instances.csv').open() as f: records=list(csv.DictReader(f))
groups={}
for x in records:groups.setdefault(x['scenario'],[]).append(x)
for v in groups.values():v.sort(key=lambda x:int(x['instance']))
classified=[]
for name,rs in sorted(groups.items()):
 assert len(rs)==500 and all(int(x['completed'])==int(x['passenger_completed'])+int(x['cargo_completed']) for x in rs)
 a=r[name]['aggregate'];pc=sum(int(x['passenger_completed']) for x in rs);pt=sum(int(x['passenger_total']) for x in rs);cc=sum(int(x['cargo_completed']) for x in rs);ct=sum(int(x['cargo_total']) for x in rs)
 classified.append({'scenario':name,'passenger_service_pct':100*pc/pt,'cargo_service_pct':100*cc/ct,'rejection_cost_share_pct':100*a['reject_penalty_mean']/a['total_cost_raw_mean'],'non_rejection_cost':a['total_cost_raw_mean']-a['reject_penalty_mean']})
with (p/'classified_summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(classified[0]));w.writeheader();w.writerows(classified)
rng=np.random.default_rng(20261101);contrasts=[]
for low,high in [('r0.6_tw0.5_ex1','r0.6_tw1.5_ex1'),('r0.6_tw1_ex0.5','r0.6_tw1_ex1.5'),('r0.3_tw0.75_ex1','r0.3_tw1.25_ex1'),('r0.9_tw0.75_ex1','r0.9_tw1.25_ex1')]:
 a,b=groups[low],groups[high];assert all(x['base_hash']==y['base_hash'] for x,y in zip(a,b))
 delta=np.array([(int(y['completed'])-int(x['completed']))*2 for x,y in zip(a,b)])
 boot=np.array([rng.choice(delta,size=500,replace=True).mean() for _ in range(10000)])
 contrasts.append({'low':low,'high':high,'difference_pp':float(delta.mean()),'conditional_instance_bootstrap_95pct':np.quantile(boot,[.025,.975]).tolist(),'training_seeds':1,'paired_instances':500,'bootstrap_repetitions':10000,'scope':'exploratory; uncertainty across instances conditional on this fixed policy, not training variability'})
(p/'paired_contrasts.json').write_text(json.dumps(contrasts,indent=2))
fig,axes=plt.subplots(2,2,figsize=(9,6),layout='constrained')
series=[([.3,.5,.6,.7,.9],[f'r{x:g}_tw1_ex1' for x in [.3,.5,.6,.7,.9]],'Passenger request fraction'),([.5,.75,1,1.25,1.5],[f'r0.6_tw{x:g}_ex1' for x in [.5,.75,1,1.25,1.5]],'Pickup-window multiplier'),([15,22.5,30,37.5,45],[f'r0.6_tw1_ex{x:g}' for x in [.5,.75,1,1.25,1.5]],'Maximum excess ride time (min)')]
for ax,(xs,names,label) in zip(axes.flat,series):
 ax.plot(xs,[r[n]['aggregate']['service_rate_mean']*100 for n in names],marker='o',color='#0072B2');ax.set_xlabel(label);ax.set_ylabel('Service rate (%)');ax.set_ylim(80,96);ax.grid(alpha=.25)
ax=axes[1,1]
for ratio,marker,color in [(.3,'s','#0072B2'),(.6,'o','#009E73'),(.9,'^','#D55E00')]:
 xs=[.75,1,1.25];ax.plot(xs,[r[f'r{ratio:g}_tw{x:g}_ex1']['aggregate']['service_rate_mean']*100 for x in xs],marker=marker,color=color,label=f'Passenger fraction {ratio:g}')
ax.set_xlabel('Pickup-window multiplier');ax.set_ylabel('Service rate (%)');ax.set_ylim(80,96);ax.grid(alpha=.25);ax.legend(fontsize=8)
fig.suptitle('N50 frozen-policy sensitivity: one training seed, 500 instances/scenario',fontsize=11)
fig.savefig(p/'sensitivity_review.png',dpi=220);fig.savefig(p/'sensitivity_review.pdf');plt.close(fig)
print('CLASSIFIED',json.dumps(classified));print('CONTRASTS',json.dumps(contrasts))
