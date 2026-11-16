"""Isolated rejection-timing ablation; invoke from the project root.

Checkpoints require this launcher/policy for evaluation: state_dict alone does
not encode the policy. Environment masks, target selection and costs unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def apply_rule(logits, mask):
    """Serve a feasible real node; otherwise reject if environment permits.

Depot remains available alongside service nodes. When rejection is unavailable,
retain the original actions. Never unmask an environment-forbidden action.
    """
    service = (~mask[..., 1:-1]).any(-1)
    reject_available = ~mask[..., -1]
    result = logits.clone()
    result[..., -1] = result[..., -1].masked_fill(service, float('-inf'))
    force = ~service & reject_available
    result[..., :-1] = result[..., :-1].masked_fill(force.unsqueeze(-1), float('-inf'))
    # Deterministic rejection must not depend on a trained head.
    result[..., -1] = result[..., -1].masked_fill(force, 0.0)
    return result


def install_policy(mode):
    from nets.attention_model import AttentionModel
    if mode == 'learned':
        return
    original = AttentionModel._one_to_many_logits
    def controlled(self, query, step_context, glimpse_K, glimpse_V, logit_K, mask):
        return apply_rule(original(self, query, step_context, glimpse_K, glimpse_V, logit_K, mask), mask)
    AttentionModel._one_to_many_logits = controlled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', choices=['learned', 'service_first'], required=True)
    parser.add_argument('--record', required=True)
    args, rest = parser.parse_known_args()
    if any(arg.startswith('--resume') for arg in rest):
        parser.error('Paired ablation must start from scratch')
    record = Path(args.record)
    record.mkdir(parents=True, exist_ok=False)
    files = ['nets/attention_model.py', 'state_mcvrptw_v2.py', 'problem_mcvrptw_v2.py', 'run_training_optimized.py', 'experiments/reject_ablation_20261029/run.py']
    manifest = {'policy': args.policy, 'argv': rest, 'status': 'running', 'start': time.time(),
                'sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files},
                'scope': 'pilot timing only; not a final ablation result'}
    dest = record / 'manifest.json'
    dest.write_text(json.dumps(manifest, indent=2))
    try:
        install_policy(args.policy)
        import run_training_optimized as training
        sys.argv = ['run_training_optimized.py', *rest]
        training.main()
        manifest['status'] = 'completed'
    except BaseException as exc:
        manifest.update(status='failed', error=repr(exc))
        raise
    finally:
        manifest['elapsed_seconds'] = time.time() - manifest['start']
        dest.write_text(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    main()
