"""Deterministic plumbing test for the timing analysis; no scientific data."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main() -> None:
    protocol = json.loads((HERE / "PROTOCOL.json").read_text())
    indices = protocol["selection"]["selected_indices"]
    report = {
        "status": "passed",
        "environment": {"fixture": True},
        "neural": {
            "314159": {"rows": [{"instance": i, "seconds": 1.0} for i in indices]},
            "271828": {"rows": [{"instance": i, "seconds": 2.0} for i in indices]},
            "161803": {"rows": [{"instance": i, "seconds": 3.0} for i in indices]},
        },
        "insertion": {"rows": [{"instance": i, "seconds": 4.0} for i in indices]},
    }
    with tempfile.TemporaryDirectory(prefix="timing-analysis-fixture-") as folder:
        root = Path(folder)
        report_path, output_path = root / "report.json", root / "analysis.json"
        report_path.write_text(json.dumps(report))
        subprocess.run(
            [
                sys.executable,
                str(HERE / "analyze.py"),
                "--protocol",
                str(HERE / "PROTOCOL.json"),
                "--report",
                str(report_path),
                "--output",
                str(output_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        result = json.loads(output_path.read_text())
    assert result["status"] == "passed"
    assert result["methods"]["neural_seed_averaged"]["mean_seconds"] == 2.0
    assert result["methods"]["strict_insertion"]["mean_seconds"] == 4.0
    assert abs(result["point_ratios"]["insertion_over_neural"] - 2.0) < 1e-12
    assert abs(result["point_ratios"]["insertion_short_over_neural_short"] - 1.8) < 1e-12
    assert abs(result["point_ratios"]["strong_pipeline_over_neural_short"] - 5.6) < 1e-12
    print("TIMING_ANALYSIS_FIXTURE_OK")


if __name__ == "__main__":
    main()
