"""Frozen inference scenarios; not parameter-specific retraining."""
import argparse, copy, hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from problem_mcvrptw_v2 import Config, MCVRPPDTWDataset

def make(n=50, samples=4, seed=928631, ratio=None, tw=1., excess=1.):
    if not 0 < tw or not 0 < excess or (ratio is not None and not 0 <= ratio <= 1):
        raise ValueError('Invalid scenario parameter')
    data = MCVRPPDTWDataset(num_samples=samples, graph_size=n, seed=seed,
                           passenger_ratio_override=ratio).data
    for s in data:
        mask = s['node_type'][:n] == 1
        windows = s['time_windows'][:n]
        widths = windows[:, 1] - windows[:, 0]
        windows[mask, 1] = torch.minimum(windows[mask, 0] + widths[mask]*tw,
                                        torch.tensor(float(Config.OPERATION_END)))
    return {'data':data, 'meta':dict(graph_size=n, samples=samples, seed=seed,
        passenger_ratio=ratio, pickup_window_multiplier=tw, excess_multiplier=excess,
        excess_minutes=Config.PASSENGER_MAX_EXCESS_RIDE_TIME_MINUTES*excess,
        total_ride_minutes=Config.PASSENGER_MAX_RIDE_TIME_MINUTES,
        pairing='TW/excess: same base orders; ratio: regenerated distribution, not identical orders',
        scope='frozen-policy robustness; pickup deadlines only, delivery windows unchanged')}

def save(bundle, path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise FileExistsError(path)
    torch.save(bundle,path)
    meta=dict(bundle['meta'],sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    path.with_suffix('.json').write_text(json.dumps(meta,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--samples',type=int,default=500);p.add_argument('--seed',type=int,default=928631)
    p.add_argument('--graph-size',type=int,default=50);p.add_argument('--ratio',type=float)
    p.add_argument('--tw',type=float,default=1.);p.add_argument('--excess',type=float,default=1.)
    a=p.parse_args();save(make(a.graph_size,a.samples,a.seed,a.ratio,a.tw,a.excess),a.output)
