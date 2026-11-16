from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/mip_scip_20261106'))
sys.path.insert(0,str(ROOT/'experiments/min_dispatch_hard_20261106'))
from bootstrap import C,PROFILE,Config,torch,MCVRPPDTW,sample_to_batch,audit,scalar_cost,audit_config,lists,SNAP
