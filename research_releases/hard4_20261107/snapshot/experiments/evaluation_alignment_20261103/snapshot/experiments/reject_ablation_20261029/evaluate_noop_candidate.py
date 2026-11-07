"""Inference-only intervention on v2 weights, NOT a retrained ablation result."""
from depot_noop import install
import evaluate_v2
install()
if __name__=='__main__':
    evaluate_v2.evaluate_model.evaluate()
