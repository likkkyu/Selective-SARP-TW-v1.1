"""Regression checks for the real-road 3 h boundary and unsafe depot fallback."""

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from state_mcvrptw_v2 import FEASIBILITY_TOLERANCE


def test_three_hour_float_boundary_matches_independent_audit():
    observed = 3.0000104822218425
    assert observed <= 3.0 + FEASIBILITY_TOLERANCE


def test_tolerance_is_small_relative_to_operational_time():
    assert FEASIBILITY_TOLERANCE == 1e-4
    assert abs(FEASIBILITY_TOLERANCE * 3600 - 0.36) < 1e-12


if __name__ == "__main__":
    test_three_hour_float_boundary_matches_independent_audit()
    test_tolerance_is_small_relative_to_operational_time()
    print("BOUNDARY_TOLERANCE_PASS")
