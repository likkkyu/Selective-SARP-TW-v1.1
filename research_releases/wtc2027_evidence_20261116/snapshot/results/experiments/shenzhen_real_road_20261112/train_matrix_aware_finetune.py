"""Finite, resumable road-matrix fine-tuning diagnostic with independent audits."""

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import default_collate


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FORMAL = HERE / "formal_p1_v1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from problem_mcvrptw_v2 import Config, MCVRPPDTW
from road_audit import audit_routes
from smoke_matrix_aware_decode import MODEL_KEYS, build_model, split_actions


STATE_KWARGS = {
    "max_concurrent_open_orders": 6,
    "min_orders_per_dispatch": 4,
    "enable_delivery_viability": True,
    "enable_viability_fallback": False,
    "relax_pickup_commitment_trip_time": False,
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def batches(indices, size):
    for start in range(0, len(indices), size):
        yield indices[start:start + size]


def select_batch(data, indices):
    return default_collate([{key: data[index][key] for key in MODEL_KEYS} for index in indices])


def evaluate(model, data, indices, batch_size, physical_config):
    model.eval()
    model.set_decode_type("greedy")
    total_cost = total_completed = total_seconds = 0.0
    invalid = forced_declines = 0
    with torch.no_grad():
        for group in batches(indices, batch_size):
            batch = select_batch(data, group)
            started = time.monotonic()
            cost, _, pi = model(batch, return_pi=True, state_kwargs=STATE_KWARGS)
            total_seconds += time.monotonic() - started
            executable, forced = MCVRPPDTW._sanitize_underfilled_dispatches(pi, 50)
            forced_declines += int(forced.sum().item())
            for row_idx, data_idx in enumerate(group):
                routes = split_actions(executable[row_idx].tolist(), 50)
                audit = audit_routes(data[data_idx], routes, physical_config)
                invalid += int(not audit["valid"])
                total_completed += audit["completed"]
            total_cost += float(cost.sum().item())
    n = len(indices)
    return {
        "instances": n,
        "mean_cost": total_cost / n,
        "mean_completed": total_completed / n,
        "service_rate_percent": 100.0 * total_completed / (50.0 * n),
        "hard_invalid": invalid,
        "forced_underfilled_declines": forced_declines,
        "decode_seconds": total_seconds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--validation-size", type=int, default=8)
    parser.add_argument("--validation-indices-file", type=Path)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--baseline", choices=("self_critical", "batch_mean"), default="self_critical")
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--model-index", type=int, default=0)
    parser.add_argument("--max-minutes", type=float, default=120.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)

    plan = json.loads((FORMAL / "plan.json").read_text())
    pool = torch.load(args.pool, map_location="cpu", weights_only=True)
    meta, data = pool["meta"], pool["data"]
    if meta["seed"] == plan["seed"] or meta["samples"] != len(data):
        raise ValueError("training pool is not independent or is incomplete")
    if not (1 <= args.validation_size < len(data)):
        raise ValueError("invalid validation size")
    if digest(args.pool) == plan["road_cases_sha256"]:
        raise ValueError("formal test cases must never be used for training")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    Config.set_normalization_profile(50, plan["normalization_profile"])
    if not (0 <= args.model_index < len(plan["models"])):
        raise ValueError(f"model-index must be in [0, {len(plan['models']) - 1}]")
    spec = plan["models"][args.model_index]
    if digest(Path(spec["checkpoint"])) != spec["sha256"]:
        raise ValueError("Initial checkpoint hash mismatch")
    checkpoint = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
    source_model_args = checkpoint.get("args", {})
    model = build_model(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    indices = list(range(len(data)))
    split_source_sha256 = None
    if args.validation_indices_file is not None:
        split_payload = json.loads(args.validation_indices_file.read_text())
        validation = split_payload.get("validation_indices", split_payload) if isinstance(split_payload, dict) else split_payload
        validation = sorted(int(index) for index in validation)
        if len(validation) != len(set(validation)) or not validation:
            raise ValueError("Frozen validation indices must be non-empty and unique")
        if validation[0] < 0 or validation[-1] >= len(data):
            raise ValueError("Frozen validation index outside pool")
        if args.validation_size != len(validation):
            raise ValueError(
                f"--validation-size={args.validation_size} differs from frozen split size {len(validation)}"
            )
        validation_set = set(validation)
        training = [index for index in indices if index not in validation_set]
        split_source_sha256 = digest(args.validation_indices_file)
    else:
        split_rng = random.Random(args.seed)
        split_rng.shuffle(indices)
        validation = sorted(indices[:args.validation_size])
        training = indices[args.validation_size:]
    protocol = {
        "status": "running",
        "scope": "road-matrix fine-tuning diagnostic; no formal P1 test exposure",
        "pool_sha256": digest(args.pool),
        "pool_seed": meta["seed"],
        "formal_test_sha256_excluded": plan["road_cases_sha256"],
        "initial_checkpoint_sha256": spec["sha256"],
        "initial_model_index": args.model_index,
        "initial_model_seed": spec["seed"],
        "training_indices": sorted(training),
        "validation_indices": validation,
        "validation_split_source_sha256": split_source_sha256,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "reinforce_baseline": args.baseline,
        "checkpoint_selection_rule": "zero hard invalid, then highest validation service rate, then lowest validation mean cost",
        "gradient_clip": 1.0,
        "seed": args.seed,
        "max_minutes": args.max_minutes,
        "state_kwargs": STATE_KWARGS,
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    report = {"protocol": protocol, "epochs": []}
    deadline = time.monotonic() + args.max_minutes * 60.0
    best_key = None
    for epoch in range(1, args.epochs + 1):
        if time.monotonic() >= deadline:
            report["status"] = "timeout"
            break
        model.train()
        model.set_decode_type("sampling")
        epoch_indices = list(training)
        random.Random(args.seed + epoch).shuffle(epoch_indices)
        losses, costs, grad_norms = [], [], []
        for batch_index, group in enumerate(batches(epoch_indices, args.batch_size), start=1):
            if time.monotonic() >= deadline:
                report["status"] = "timeout"
                break
            optimizer.zero_grad(set_to_none=True)
            batch = select_batch(data, group)
            if args.baseline == "self_critical":
                model.eval()
                model.set_decode_type("greedy")
                with torch.no_grad():
                    baseline_cost, _ = model(batch, state_kwargs=STATE_KWARGS)
                model.train()
                model.set_decode_type("sampling")
            else:
                baseline_cost = None
            cost, log_likelihood = model(batch, state_kwargs=STATE_KWARGS)
            if baseline_cost is None:
                advantage = cost.detach() - cost.detach().mean()
            else:
                advantage = cost.detach() - baseline_cost.detach()
            loss = (advantage * log_likelihood).mean()
            if not torch.isfinite(cost).all() or not torch.isfinite(loss):
                raise FloatingPointError((epoch, batch_index, cost, loss))
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError((epoch, batch_index, "gradient", grad_norm))
            optimizer.step()
            losses.append(float(loss.item()))
            costs.extend(float(value) for value in cost)
            grad_norms.append(float(grad_norm.item()))
            print(
                f"ROAD_TRAIN epoch={epoch}/{args.epochs} batch={batch_index} "
                f"mean_cost={sum(costs)/len(costs):.3f} grad={grad_norms[-1]:.3f}",
                flush=True,
            )
        validation_result = evaluate(model, data, validation, args.batch_size, plan["physical_config"])
        epoch_result = {
            "epoch": epoch,
            "train_mean_cost": sum(costs) / len(costs),
            "train_mean_loss": sum(losses) / len(losses),
            "train_mean_gradient_norm_before_clip": sum(grad_norms) / len(grad_norms),
            "validation": validation_result,
        }
        report["epochs"].append(epoch_result)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "road_matrix_semantics_version": 1,
                "args": source_model_args,
                "source_checkpoint_sha256": spec["sha256"],
                "protocol": protocol,
            },
            args.output / "last.pt",
        )
        if validation_result["hard_invalid"] != 0:
            raise AssertionError(validation_result)
        candidate_key = (
            int(validation_result["hard_invalid"]),
            -float(validation_result["service_rate_percent"]),
            float(validation_result["mean_cost"]),
        )
        if best_key is None or candidate_key < best_key:
            best_key = candidate_key
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "road_matrix_semantics_version": 1,
                    "args": source_model_args,
                    "source_checkpoint_sha256": spec["sha256"],
                    "protocol": protocol,
                    "validation": validation_result,
                },
                args.output / "best.pt",
            )
        report["status"] = "running"
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print("ROAD_VALID", json.dumps(validation_result), flush=True)
    else:
        report["status"] = "completed"
    protocol["status"] = report["status"]
    protocol["completed_epochs"] = len(report["epochs"])
    report["protocol"] = protocol
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "epochs": len(report["epochs"])}, indent=2))


if __name__ == "__main__":
    main()
