"""CPU-only forward/backward smoke test for road-matrix policy training."""

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import default_collate


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FORMAL = HERE / "formal_p1_v1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from problem_mcvrptw_v2 import Config
from smoke_matrix_aware_decode import MODEL_KEYS, build_model


def main():
    torch.manual_seed(20261113)
    plan = json.loads((FORMAL / "plan.json").read_text())
    checkpoint = torch.load(plan["models"][0]["checkpoint"], map_location="cpu", weights_only=False)
    model = build_model(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.train()
    model.set_decode_type("sampling")
    Config.set_normalization_profile(50, plan["normalization_profile"])

    samples = torch.load(FORMAL / "road_cases.pt", map_location="cpu", weights_only=True)["data"][:2]
    prepared = []
    for sample in samples:
        item = {key: sample[key] for key in MODEL_KEYS}
        item["road_travel_time_matrix_hours"] = (
            sample["road_distance_matrix_km"] / plan["physical_config"]["VEHICLE_SPEED"]
        )
        prepared.append(item)
    batch = default_collate(prepared)
    state_kwargs = {
        "max_concurrent_open_orders": 6,
        "min_orders_per_dispatch": 4,
        "enable_delivery_viability": True,
        "enable_viability_fallback": False,
        "relax_pickup_commitment_trip_time": False,
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    optimizer.zero_grad(set_to_none=True)
    cost, log_likelihood = model(batch, state_kwargs=state_kwargs)
    advantage = cost.detach() - cost.detach().mean()
    loss = (advantage * log_likelihood).mean()
    if not torch.isfinite(cost).all() or not torch.isfinite(log_likelihood).all() or not torch.isfinite(loss):
        raise AssertionError("non-finite forward values")
    loss.backward()
    finite_gradients = 0
    gradient_norm_sq = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            raise AssertionError("non-finite gradient")
        finite_gradients += 1
        gradient_norm_sq += float(parameter.grad.norm().item()) ** 2
    if finite_gradients == 0 or gradient_norm_sq <= 0.0:
        raise AssertionError("road-aware training smoke produced no usable gradient")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    result = {
        "status": "pass",
        "batch_size": 2,
        "costs": [float(value) for value in cost],
        "log_likelihood": [float(value) for value in log_likelihood],
        "loss": float(loss),
        "gradient_tensors": finite_gradients,
        "gradient_l2_before_clip": gradient_norm_sq ** 0.5,
        "optimizer_step_completed": True,
        "scope": "numerical smoke test only; formal test instances must not be used for training",
    }
    (HERE / "matrix_aware_train_smoke_report.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
