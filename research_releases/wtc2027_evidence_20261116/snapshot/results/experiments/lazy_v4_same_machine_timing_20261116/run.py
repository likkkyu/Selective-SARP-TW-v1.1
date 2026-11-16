"""Same-node timing benchmark for frozen lazy-v4 neural and insertion constructors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
for folder in reversed(
    (
        ROOT / "experiments/drl_canonical_audit_20261115",
        ROOT / "experiments/hard4_lns_20261106",
        ROOT / "experiments/min_dispatch_hard_20261106",
        ROOT / "experiments/mip_scip_20261106",
    )
):
    sys.path.insert(0, str(folder))

import torch  # noqa: E402
from torch.utils.data import default_collate  # noqa: E402

torch.set_num_threads(1)
import audit_common as common  # noqa: E402
from evaluate_lazy_v4_canonical import install_lazy_patches  # noqa: E402
from evaluate_model import build_model_from_checkpoint  # noqa: E402
from problem_mcvrptw_v2 import MCVRPPDTWDataset  # noqa: E402
from run_v3 import install_policy  # noqa: E402


EXPECTED_VERSIONS = {
    "canonical_training_version": 2,
    "cost_semantics_version": 4,
    "dispatch_clock_version": 1,
    "completion_guard_version": 1,
    "hard4_admission_version": 4,
    "lazy_certificate_decoder_version": 4,
    "depot_noop_version": 1,
}
EXPECTED_NORMALIZATION_SHA256 = "51313f12ca85cb402419ff3c15454457fc0884b96fde50b338dad54d66163316"
STATE_KWARGS = {
    "max_concurrent_open_orders": 6,
    "min_orders_per_dispatch": 4,
    "enable_delivery_viability": True,
    "enable_viability_fallback": False,
    "relax_pickup_commitment_trip_time": False,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def action_sha(actions: list[int]) -> str:
    return hashlib.sha256(json.dumps(actions, separators=(",", ":")).encode()).hexdigest()


def environment() -> dict:
    try:
        nvidia_smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except Exception as exc:  # pragma: no cover - diagnostic only
        nvidia_smi = repr(exc)
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "nvidia_smi": nvidia_smi,
        "torch_num_threads": torch.get_num_threads(),
    }


def validate_checkpoint(checkpoint: dict, normalization: dict) -> None:
    raw = checkpoint.get("args", {})
    raw = vars(raw) if hasattr(raw, "__dict__") else raw
    mismatches = {
        key: {"expected": value, "observed": raw.get(key)}
        for key, value in EXPECTED_VERSIONS.items()
        if raw.get(key) != value
    }
    if raw.get("postdecode_underfilled_sanitizer") is not False:
        mismatches["postdecode_underfilled_sanitizer"] = {
            "expected": False,
            "observed": raw.get("postdecode_underfilled_sanitizer"),
        }
    if raw.get("canonical_normalization_profile_sha256") != EXPECTED_NORMALIZATION_SHA256:
        mismatches["canonical_normalization_profile_sha256"] = {
            "expected": EXPECTED_NORMALIZATION_SHA256,
            "observed": raw.get("canonical_normalization_profile_sha256"),
        }
    if checkpoint.get("normalization_profile") != normalization:
        mismatches["normalization_profile"] = "checkpoint differs from frozen N100 profile"
    if mismatches:
        raise ValueError({"checkpoint_semantic_mismatch": mismatches})


def insertion_start_solver_free(sample, config, profile, vehicles):
    """Exact audited insertion logic without importing the unused SCIP model."""
    values = common.lists(sample)
    n = len(values["loc"]) // 2
    routes, accepted = [], set()
    for order in sorted(range(1, n + 1), key=lambda i: (values["time_windows"][i - 1][1], i)):
        best = None
        for route_index in range(len(routes) + (len(routes) < vehicles)):
            old = routes[route_index] if route_index < len(routes) else []
            for pickup_index in range(len(old) + 1):
                first = old[:pickup_index] + [order] + old[pickup_index:]
                for delivery_index in range(pickup_index + 1, len(first) + 1):
                    new = first[:delivery_index] + [order + n] + first[delivery_index:]
                    route_audit = common.audit(
                        sample, new + [0], common.audit_config(config, n, 1)
                    )
                    if any(v["kind"] != "min_dispatch" for v in route_audit["violations"]):
                        continue
                    trial = [list(route) for route in routes]
                    if route_index == len(routes):
                        trial.append(new)
                    else:
                        trial[route_index] = new
                    actions = [
                        node for route in trial for node in route + [0]
                    ] + [2 * n + 1] * (n - len(accepted) - 1)
                    score = common.scalar_cost(sample, actions, config, profile)["objective"]
                    if best is None or score < best[0]:
                        best = (score, trial)
        if best is not None:
            routes = best[1]
            accepted.add(order)
    return [node for route in routes for node in route + [0]] + [2 * n + 1] * (
        n - len(accepted)
    )


def make_strict_insertion_solver_free(sample, config, profile):
    """Exact hard-four repair used by the frozen insertion baseline."""
    values = common.lists(sample)
    n = len(values["loc"]) // 2
    vehicles = (n + 5) // 6
    old = insertion_start_solver_free(sample, config, profile, vehicles)
    routes, current = [], []
    for node in old:
        if node == 0:
            if current:
                routes.append(current)
                current = []
        elif node <= 2 * n:
            current.append(node)
    if current:
        raise RuntimeError("unclosed insertion route")
    while any(len(route) // 2 < 4 for route in routes):
        route_index = min(
            (i for i, route in enumerate(routes) if len(route) // 2 < 4),
            key=lambda i: (len(routes[i]), i),
        )
        dropped = routes.pop(route_index)
        orders = [node for node in dropped if node <= n]
        for order in orders:
            best = None
            for target_index, route in enumerate(routes):
                for pickup_index in range(len(route) + 1):
                    first = route[:pickup_index] + [order] + route[pickup_index:]
                    for delivery_index in range(pickup_index + 1, len(first) + 1):
                        new = first[:delivery_index] + [order + n] + first[delivery_index:]
                        route_audit = common.audit(
                            sample, new + [0], common.audit_config(config, n, 1)
                        )
                        if any(v["kind"] != "min_dispatch" for v in route_audit["violations"]):
                            continue
                        trial = [
                            existing if j != target_index else new
                            for j, existing in enumerate(routes)
                        ]
                        served = sum(len(existing) // 2 for existing in trial)
                        actions = [
                            node for route2 in trial for node in route2 + [0]
                        ] + [2 * n + 1] * (n - served)
                        score = common.scalar_cost(sample, actions, config, profile)["objective"]
                        if best is None or score < best[0]:
                            best = (score, target_index, new)
            if best is not None:
                routes[best[1]] = best[2]
    accepted = {node for route in routes for node in route if node <= n}
    rejected = set(range(1, n + 1)) - accepted
    actions = [node for route in routes for node in route + [0]] + [2 * n + 1] * len(
        rejected
    )
    route_audit = common.audit(sample, actions, common.audit_config(config, n, vehicles))
    if route_audit["violations"] or route_audit["unfulfilled"]:
        raise RuntimeError({"strict_insertion_audit": route_audit})
    return actions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--baseline-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("status") != "frozen_before_same_machine_timing":
        raise ValueError("timing protocol is not frozen")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    evidence_root = args.evidence_root.resolve()
    baseline_path = args.baseline_report.resolve()
    if sha256(baseline_path) != protocol["frozen_report_sha256"]["strict_insertion"]:
        raise ValueError("strict-insertion report hash mismatch")

    started = time.monotonic()
    report = {
        "status": "running",
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "environment": environment(),
        "neural": {},
        "insertion": {"rows": []},
    }

    def save() -> None:
        report["elapsed_seconds"] = time.monotonic() - started
        (output / "REPORT.json").write_text(json.dumps(report, indent=2) + "\n")

    signal.signal(
        signal.SIGALRM,
        lambda *_: (_ for _ in ()).throw(TimeoutError("same-machine benchmark timeout")),
    )
    signal.alarm(protocol["total_timeout_seconds"])
    save()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required for the same-machine timing benchmark")
        install_lazy_patches()
        n = protocol["graph_size"]
        normalization = common.profile(n)
        common.Config.set_normalization_profile(n, normalization)
        config = common.parameters(common.Config)
        final_dataset = MCVRPPDTWDataset(
            num_samples=100, graph_size=n, seed=protocol["final_dataset_seed"]
        )
        warmup_dataset = MCVRPPDTWDataset(
            num_samples=protocol["warmup"]["instances_per_method"],
            graph_size=n,
            seed=protocol["warmup"]["dataset_seed"],
        )
        device = torch.device("cuda:0")
        install_policy("learned")

        for seed in protocol["neural_seeds"]:
            seed_root = evidence_root / f"seed_{seed}"
            checkpoint_path = seed_root / "outputs/pomo_n100_optimized/model_best.pt"
            frozen_report_path = seed_root / "final_test/report.json"
            if sha256(checkpoint_path) != protocol["checkpoint_sha256"][str(seed)]:
                raise ValueError(f"checkpoint hash mismatch for seed {seed}")
            if sha256(frozen_report_path) != protocol["frozen_report_sha256"][f"neural_{seed}"]:
                raise ValueError(f"frozen report hash mismatch for seed {seed}")
            frozen = json.loads(frozen_report_path.read_text())
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            validate_checkpoint(checkpoint, normalization)
            model = build_model_from_checkpoint(checkpoint, device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            model.set_decode_type("greedy")

            def forward(sample) -> list[int]:
                batch = default_collate([sample])
                batch = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
                with torch.no_grad():
                    _, _, actions_tensor = model(batch, return_pi=True, state_kwargs=STATE_KWARGS)
                return actions_tensor[0].detach().cpu().tolist()

            for sample in warmup_dataset:
                forward(sample)
            torch.cuda.synchronize(device)
            rows = []
            for index in protocol["selection"]["selected_indices"]:
                torch.cuda.synchronize(device)
                begin = time.perf_counter()
                actions = forward(final_dataset[index])
                torch.cuda.synchronize(device)
                seconds = time.perf_counter() - begin
                expected = frozen["rows"][index]["actions_sha256"]
                observed = action_sha(actions)
                if observed != expected:
                    raise RuntimeError(
                        {"neural_action_identity_mismatch": seed, "instance": index, "expected": expected, "observed": observed}
                    )
                rows.append({"instance": index, "seconds": seconds, "actions_sha256": observed})
                print("TIMING_NEURAL", seed, len(rows), "/", len(protocol["selection"]["selected_indices"]), round(seconds, 4), flush=True)
            report["neural"][str(seed)] = {"rows": rows}
            save()
            del model, checkpoint
            torch.cuda.empty_cache()

        baseline = json.loads(baseline_path.read_text())
        for sample in warmup_dataset:
            make_strict_insertion_solver_free(sample, config, normalization)
        for index in protocol["selection"]["selected_indices"]:
            begin = time.perf_counter()
            actions = make_strict_insertion_solver_free(final_dataset[index], config, normalization)
            seconds = time.perf_counter() - begin
            observed = action_sha(actions)
            expected = action_sha(baseline["rows"][index]["insertion_actions"])
            if observed != expected:
                raise RuntimeError(
                    {"insertion_action_identity_mismatch": index, "expected": expected, "observed": observed}
                )
            report["insertion"]["rows"].append(
                {"instance": index, "seconds": seconds, "actions_sha256": observed}
            )
            save()
            print("TIMING_INSERTION", len(report["insertion"]["rows"]), "/", len(protocol["selection"]["selected_indices"]), round(seconds, 4), flush=True)

        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        save()
        signal.alarm(0)


if __name__ == "__main__":
    main()
