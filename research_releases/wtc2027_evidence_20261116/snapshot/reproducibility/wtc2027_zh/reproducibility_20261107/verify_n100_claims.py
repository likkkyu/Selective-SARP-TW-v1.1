"""Deterministically check headline N100 manuscript numbers against frozen JSON.

This is an internal-consistency gate only.  It does not validate experimental
design, execution, causal interpretation, or solver optimality.
"""
from pathlib import Path
import hashlib
import json


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("n100_claim_audit.json")


def load(relative):
    path = ROOT / relative
    return path, json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manuscript_path = ROOT / "wtc2027_zh/论文正文.md"
    supplement_path = ROOT / "wtc2027_zh/方法与复现补充说明.md"
    manuscript = manuscript_path.read_text()
    supplement = supplement_path.read_text()

    quality_path, quality = load("experiments/lazy_v4_hybrid_confirmation_20261116/PAIRED_ANALYSIS.json")
    timing_path, timing = load("experiments/lazy_v4_same_machine_timing_20261116/remote_evidence/formal_v2/ANALYSIS.json")
    alignment_path, alignment = load("experiments/lazy_v4_scip_alignment_20261116/HYBRID_PAIRED_ANALYSIS.json")
    dataset_path, dataset = load("experiments/lazy_v4_scip_alignment_20261116/DATASET_EQUIVALENCE.json")

    q = quality["methods"]
    qp = quality["paired"]
    tm = timing["methods"]
    am = alignment["methods"]
    ap = alignment["paired"]
    checks = []

    def expect(document, document_name, label, expected):
        checks.append({
            "id": label,
            "document": document_name,
            "expected_text": expected,
            "passed": expected in document,
        })

    # Main-paper headline values.
    expect(manuscript, "论文正文.md", "main_n100_raw_service", f'{q["neural_start_seed_averaged"]["service_rate"]*100:.2f}%')
    expect(manuscript, "论文正文.md", "main_n100_hybrid_service", f'{q["neural_plus_short_lns_seed_averaged"]["service_rate"]*100:.2f}%')
    expect(manuscript, "论文正文.md", "main_n100_insertion_hybrid_service", f'{q["insertion_plus_short_lns"]["service_rate"]*100:.2f}%')
    expect(manuscript, "论文正文.md", "main_n100_hybrid_gain", f'{qp["neural_short_minus_neural_start_service_rate"]["mean_difference"]*100:.3f}')
    expect(manuscript, "论文正文.md", "main_same_node_neural_hybrid_time", f'{tm["neural_plus_0.5s_lns_derived"]["mean_seconds"]:.2f} s')
    expect(manuscript, "论文正文.md", "main_same_node_insertion_hybrid_time", f'{tm["insertion_plus_0.5s_lns_derived"]["mean_seconds"]:.2f} s')
    expect(manuscript, "论文正文.md", "main_alignment_hybrid_service", f'{am["neural_plus_0_5s_lns_seed_averaged"]["service_rate"]*100:.2f}%')
    expect(manuscript, "论文正文.md", "main_alignment_scip_time", f'{am["scip_300s_incumbent"]["mean_pipeline_seconds_local_cpu"]:.2f} s')
    expect(manuscript, "论文正文.md", "main_alignment_solver_boundary", "未证明最优且未改善初解")

    # Supplement uses one extra decimal place for traceability.
    for label, value in [
        ("supp_raw_service", f'{q["neural_start_seed_averaged"]["service_rate"]*100:.3f}%'),
        ("supp_hybrid_service", f'{q["neural_plus_short_lns_seed_averaged"]["service_rate"]*100:.3f}%'),
        ("supp_insertion_short_service", f'{q["insertion_plus_short_lns"]["service_rate"]*100:.3f}%'),
        ("supp_strong_service", f'{q["strong_lns_10s"]["service_rate"]*100:.3f}%'),
        ("supp_neural_time", f'{tm["neural_seed_averaged"]["mean_seconds"]:.3f}'),
        ("supp_insertion_time", f'{tm["strict_insertion"]["mean_seconds"]:.3f}'),
        ("supp_alignment_hybrid_service", f'{am["neural_plus_0_5s_lns_seed_averaged"]["service_rate"]*100:.2f}%'),
        ("supp_alignment_hybrid_time", f'{am["neural_plus_0_5s_lns_seed_averaged"]["mean_end_to_end_seconds_local_cpu"]:.3f}'),
        ("supp_alignment_insertion_time", f'{am["strict_insertion"]["mean_build_seconds_local_cpu"]:.3f}'),
        ("supp_alignment_scip_time", f'{am["scip_300s_incumbent"]["mean_pipeline_seconds_local_cpu"]:.3f}'),
        ("supp_alignment_hybrid_gain", f'{ap["hybrid_minus_raw"]["service_rate"]["mean_difference"]*100:.3f}'),
        ("supp_dataset_sha", dataset["scenario_sha256"]),
    ]:
        expect(supplement, "方法与复现补充说明.md", label, value)

    source_statuses = {
        "quality": quality["status"],
        "timing": timing["status"],
        "alignment": alignment["status"],
        "dataset_equivalence": dataset["status"],
    }
    all_passed = all(c["passed"] for c in checks) and all(v == "passed" for v in source_statuses.values())
    report = {
        "status": "passed" if all_passed else "failed",
        "scope": "Headline N100 numerical and boundary-text consistency across the current manuscript, supplement, and frozen JSON evidence",
        "boundary": "This deterministic gate verifies only declared string/value consistency. It does not validate experimental design, execution, solver optimality, or publication readiness.",
        "source_statuses": source_statuses,
        "inputs": [
            {"path": str(p.relative_to(ROOT)), "sha256": sha256(p)}
            for p in [manuscript_path, supplement_path, quality_path, timing_path, alignment_path, dataset_path]
        ],
        "checks": checks,
        "passed_checks": sum(c["passed"] for c in checks),
        "total_checks": len(checks),
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "passed": report["passed_checks"], "total": report["total_checks"]}, ensure_ascii=False))
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
