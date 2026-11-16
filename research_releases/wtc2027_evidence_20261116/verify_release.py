#!/usr/bin/env python3
"""Verify the immutable members and deliberate exclusions of a release candidate."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "CANDIDATE_MANIFEST.json"
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".log", ".tgz", ".gz", ".zip"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = set()
    for item in manifest["files"]:
        rel = item["path"]
        path = ROOT / rel
        assert path.is_file(), f"missing: {rel}"
        assert path.stat().st_size == item["bytes"], f"size mismatch: {rel}"
        assert sha256(path) == item["sha256"], f"hash mismatch: {rel}"
        assert path.suffix.lower() not in FORBIDDEN_SUFFIXES, f"forbidden suffix: {rel}"
        expected.add(rel)
    actual = {
        str(path.relative_to(ROOT))
        for path in (ROOT / "snapshot").rglob("*")
        if path.is_file()
    }
    assert actual == expected, {
        "unlisted": sorted(actual - expected),
        "missing_from_snapshot": sorted(expected - actual),
    }
    assert manifest["status"] in {"local_candidate_not_published", "published_git_snapshot"}
    assert manifest["secret_scan"]["status"] == "passed"
    print(json.dumps({"status": "passed", "files": len(expected)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
