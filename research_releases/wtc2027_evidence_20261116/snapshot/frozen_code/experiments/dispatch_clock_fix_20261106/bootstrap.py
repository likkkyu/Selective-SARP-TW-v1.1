from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];P=Path(__file__).resolve().parent;ALIGN=ROOT/'experiments/evaluation_alignment_20261103';SNAP=ALIGN/'snapshot'
for d in (SNAP,SNAP/'experiments/reject_ablation_20261029',SNAP/'experiments/sensitivity_20261030'):sys.path.insert(0,str(d))
import torch
torch.set_num_threads(1)
from problem_mcvrptw_v2 import Config,MCVRPPDTW
from state_mcvrptw_v2 import StateMCVRPPDTW as State
from baseline_utils import sample_to_batch
KW=dict(max_concurrent_open_orders=6,min_orders_per_dispatch=4,enable_delivery_viability=True,enable_viability_fallback=False,relax_pickup_commitment_trip_time=False)
