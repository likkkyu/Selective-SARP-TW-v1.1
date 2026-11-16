"""Evaluate tagged checkpoints with automatic policy recovery; fail closed."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import evaluate_model
from run_v3 import install_policy
original_build = evaluate_model.build_model_from_checkpoint

def build(checkpoint, *args, **kwargs):
    config = checkpoint.get('args', {})
    if config.get('ablation_policy_version') != 3:
        raise ValueError('Missing/unsupported ablation policy metadata; use a v3 checkpoint')
    install_policy(config['ablation_policy'])
    if config.get('depot_noop_fix') is not True:
        raise ValueError('Missing shared decoder fix metadata')
    from depot_noop import install
    install()
    model = original_build(checkpoint, *args, **kwargs)
    print('Restored ablation policy:', config['ablation_policy'])
    return model

evaluate_model.build_model_from_checkpoint = build
if __name__ == '__main__':
    evaluate_model.evaluate()
