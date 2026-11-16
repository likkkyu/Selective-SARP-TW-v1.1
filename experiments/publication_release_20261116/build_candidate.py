#!/usr/bin/env python3
"""Build a selective, hash-audited public candidate without model weights."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import re
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEST = ROOT / "research_releases" / "wtc2027_evidence_20261116"
SNAPSHOT = DEST / "snapshot"
INVENTORY = ROOT / "wtc2027_zh/reproducibility_20261107/inventory.json"
ALLOWED_SUFFIXES = {".py", ".md", ".json", ".csv", ".txt", ".png", ".pdf", ".sh"}
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".log", ".tgz", ".gz", ".zip"}
FORBIDDEN_PARTS = {"__pycache__", ".git", ".codex", ".claude"}
SECRET_PATTERNS = {
    "private_key": re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"),
    "cloud_ssh": re.compile(r"(?:root@|connect\.[a-z]+\.seetacloud\.com|ssh\s+-p\s+\d+)", re.I),
    "github_token": re.compile(r"(?:ghp|github_pat)_[A-Za-z0-9_]{20,}"),
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_exact(source, relative):
    target = SNAPSHOT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def include_tree(source_dir, release_prefix):
    for source in sorted(source_dir.rglob("*")):
        if not source.is_file():
            continue
        rel = source.relative_to(source_dir)
        if any(part in FORBIDDEN_PARTS for part in rel.parts):
            continue
        if source.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        if source.name in {"NEXT_STAGE.md", "RUN_STATUS.md"}:
            continue
        copy_exact(source, Path(release_prefix) / rel)


def main():
    if DEST.exists():
        raise SystemExit(f"refusing to overwrite existing candidate: {DEST}")
    SNAPSHOT.mkdir(parents=True)
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    assert inventory["status"] == "complete_local_inventory"

    # Frozen source bytes used by the six completed manuscript studies.
    for item in inventory["frozen_code_files"]:
        source = ROOT / item["resolved_path"]
        assert sha256(source) == item["sha256"]
        copy_exact(source, Path("frozen_code") / item["logical_path"])

    # New manuscript evidence, excluding tensors, weights, logs and archives.
    evidence_dirs = [
        "experiments/lazy_v4_hybrid_confirmation_20261116",
        "experiments/lazy_v4_same_machine_timing_20261116",
        "experiments/lazy_v4_scip_alignment_20261116",
        "experiments/shenzhen_real_road_20261112",
    ]
    for name in evidence_dirs:
        include_tree(ROOT / name, Path("results") / name)

    for name in [
        "wtc2027_zh/reproducibility_20261107/inventory.json",
        "wtc2027_zh/reproducibility_20261107/n100_claim_audit.json",
        "wtc2027_zh/reproducibility_20261107/verify_n100_claims.py",
        "wtc2027_zh/review_20261116/REFERENCE_VERIFICATION.md",
        "wtc2027_zh/review_20261116/REFERENCE_VERIFICATION.json",
        "wtc2027_zh/review_20261116/QA_STATUS.json",
    ]:
        copy_exact(ROOT / name, Path("reproducibility") / name)

    shutil.copy2(HERE / "README_TEMPLATE.md", DEST / "README.md")
    shutil.copy2(HERE / "ODBL_ATTRIBUTION.md", DEST / "ODBL_ATTRIBUTION.md")
    shutil.copy2(HERE / "verify_release.py", DEST / "verify_release.py")
    shutil.copy2(ROOT / "LICENSE", DEST / "LICENSE")

    files = []
    absolute_path_mentions = []
    secret_hits = []
    for path in sorted(SNAPSHOT.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(DEST))
        assert path.suffix.lower() not in FORBIDDEN_SUFFIXES
        data = path.read_bytes()
        files.append({"path": rel, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        if path.suffix.lower() in {".py", ".md", ".json", ".csv", ".txt", ".sh"}:
            text = data.decode("utf-8", errors="replace")
            if re.search(r"/(?:Users/[^/]+|root/autodl-tmp)/", text):
                absolute_path_mentions.append(rel)
            for label, pattern in SECRET_PATTERNS.items():
                if pattern.search(text):
                    secret_hits.append({"path": rel, "pattern": label})
    assert not secret_hits, secret_hits

    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    models = inventory.get("checkpoints", [])
    road_lock = json.loads((ROOT / "experiments/shenzhen_real_road_20261112/road_native_model_lock_v1.json").read_text())
    manifest = {
        "status": "local_candidate_not_published",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "Code and result evidence for N100 and Shenzhen road-network validation; manuscript and all tensor/checkpoint files excluded",
        "source_git_head": git_head,
        "source_worktree_note": "candidate includes explicitly hashed frozen source bytes and may be built from a dirty working tree",
        "previous_public_release_commit": "1505564a9fc06cf226d56eae30484adfc7b42574",
        "manuscript_sha256": sha256(ROOT / "wtc2027_zh/论文正文.md"),
        "files": files,
        "omitted_model_identifiers": [
            {"policy": m.get("policy"), "training_seed": m.get("training_seed"), "sha256": m.get("sha256"), "bytes": m.get("bytes")}
            for m in models
        ] + road_lock.get("models", []),
        "secret_scan": {"status": "passed", "patterns": sorted(SECRET_PATTERNS), "hits": []},
        "absolute_path_mentions": {
            "count": len(absolute_path_mentions),
            "files": absolute_path_mentions,
            "interpretation": "retained only inside immutable protocol/result provenance; not credentials or access routes",
        },
        "licensing": {
            "project_code": "existing repository LICENSE retained",
            "osm_derived_data": "ODbL; see ODBL_ATTRIBUTION.md",
            "model_weights": "not distributed and no reuse licence asserted",
        },
        "boundary": "A local hash-audited candidate is not a public GitHub release, archival DOI, independent reproduction, field trial, or solver-optimality certificate.",
    }
    (DEST / "CANDIDATE_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "files": len(files), "bytes": sum(f["bytes"] for f in files), "absolute_path_files": len(absolute_path_mentions)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
