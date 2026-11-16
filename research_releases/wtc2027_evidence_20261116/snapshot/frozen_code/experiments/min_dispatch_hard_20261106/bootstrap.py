from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];P=Path(__file__).resolve().parent
MIP=ROOT/'experiments/mip_scip_20261106';sys.path.insert(0,str(MIP))
from validate import SNAP,C,PROFILE,torch,Config,MCVRPPDTW,sample_to_batch,audit
from scip_selective import SelectiveSCIP,scalar_cost,audit_config,lists
