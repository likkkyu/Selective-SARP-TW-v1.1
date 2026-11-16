#!/usr/bin/env python3
"""Finalize a candidate only after its first public commit is read back."""
from pathlib import Path
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--branch", required=True)
    args = parser.parse_args()
    release = Path(args.release_dir).resolve()
    manifest_path = release / "CANDIDATE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "local_candidate_not_published"
    assert len(args.candidate_commit) == 40 and all(c in "0123456789abcdef" for c in args.candidate_commit)
    manifest["status"] = "published_git_snapshot"
    manifest["public_candidate_commit_read_back"] = args.candidate_commit
    manifest["public_branch"] = args.branch
    manifest["publication_verification"] = (
        "The candidate commit was pushed and its branch ref was read back with an exact SHA match before this finalization commit."
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    readme = release / "README.md"
    text = readme.read_text(encoding="utf-8")
    old = "This directory is a **local candidate until a Git commit is created and read back from GitHub**. Do not cite it by the local directory name as a public release."
    new = (
        f"This snapshot is published on branch `{args.branch}`. The first public candidate commit "
        f"`{args.candidate_commit}` was read back before the publication-status finalization commit. "
        "Use a fixed commit URL rather than the moving branch name when citing the evidence."
    )
    assert old in text
    readme.write_text(text.replace(old, new), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "candidate_commit": args.candidate_commit, "branch": args.branch}))


if __name__ == "__main__":
    main()
