"""Merge independently seeded road-native development pools with provenance."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if len(args.inputs) < 2:
        raise ValueError("At least two pools are required")

    merged_data = []
    merged_cases = []
    sources = []
    reference = None
    seeds = set()
    for path in args.inputs:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        manifest = json.loads(path.with_suffix(".json").read_text())
        if digest(path) != manifest["sha256"] or len(payload["data"]) != manifest["samples"]:
            raise ValueError(f"Incomplete or hash-mismatched pool: {path}")
        meta = payload["meta"]
        if int(meta.get("generator_version", -1)) != 3:
            raise ValueError(f"Only strict road-native generator v3 pools may be merged: {path}")
        signature = (
            meta["graph_size"], meta["graph_sha256"],
            meta["formal_test_road_cases_sha256"], meta["matrix_speed_kph"],
        )
        if reference is None:
            reference = signature
        elif signature != reference:
            raise ValueError(f"Pool protocol mismatch: {path}")
        if meta["seed"] in seeds:
            raise ValueError(f"Duplicate pool seed: {meta['seed']}")
        seeds.add(meta["seed"])
        start = len(merged_data)
        merged_data.extend(payload["data"])
        for local_index, case in enumerate(manifest["cases"]):
            if case["band_fallbacks"] != 0 or case["unique_road_nodes"] != 101:
                raise ValueError(f"Non-strict case in {path} index {local_index}")
            merged_cases.append({
                **case,
                "instance": start + local_index,
                "source_pool_seed": meta["seed"],
                "source_pool_index": local_index,
            })
        sources.append({
            "path": str(path.resolve().relative_to(ROOT)),
            "sha256": digest(path),
            "seed": meta["seed"],
            "samples": meta["samples"],
        })

    meta = {
        "scope": "merged strict road-native development pool; never formal P1 testing",
        "generator_version": 3,
        "seed": sorted(seeds),
        "samples": len(merged_data),
        "graph_size": reference[0],
        "graph_sha256": reference[1],
        "formal_test_road_cases_sha256": reference[2],
        "matrix_speed_kph": reference[3],
        "source_pools": sources,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": meta, "data": merged_data}, args.output)
    manifest = {
        **meta,
        "status": "prepared",
        "output": str(args.output.resolve().relative_to(ROOT)),
        "sha256": digest(args.output),
        "cases": merged_cases,
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ("status", "samples", "sha256", "source_pools")}, indent=2))


if __name__ == "__main__":
    main()
