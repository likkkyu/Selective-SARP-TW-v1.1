# Lazy-v4 same-node timing result

## Scope and protocol

This benchmark measures deployment-style N100 construction latency on one
AutoDL node without retraining or changing any frozen model. Thirty instance
indices were sampled without replacement from the closed 100-instance final
set using seed 20261116 and frozen before timing. Three separate warm-up
instances were excluded. Every timed neural and insertion output was required
to reproduce the corresponding action hash in the frozen 100-instance report.

The neural constructor used greedy batch-one inference on an NVIDIA GeForce RTX
4090 D with CUDA synchronization. Strict insertion used one CPU thread on the
same node. The container had a 16-vCPU cgroup quota on an Intel Xeon Platinum
8352S host. The result is therefore a natural-resource deployment comparison,
not a hardware-neutral complexity comparison.

## Timing result

| Method | Mean (s) | Median (s) | P95 (s) |
|---|---:|---:|---:|
| Neural seed 314159 | 9.506 | 9.413 | 10.897 |
| Neural seed 271828 | 8.661 | 8.568 | 9.771 |
| Neural seed 161803 | 9.383 | 9.145 | 11.301 |
| Neural, seed-averaged within instance | **9.183** | **9.104** | **10.213** |
| Strict insertion, one CPU thread | 53.888 | 53.766 | 56.740 |
| Neural + 0.5 s LNS, derived | **9.683** | 9.604 | 10.713 |
| Insertion + 0.5 s LNS, derived | 54.388 | 54.266 | 57.240 |
| Insertion + 10 s LNS, derived | 63.888 | 63.766 | 66.740 |

Strict insertion required 5.868 times the mean neural construction time. The
paired-instance bootstrap 95% interval for the ratio of means was
5.681--6.044. After adding the identical 0.5-second search budget, the insertion
hybrid required 5.617 times the derived neural-hybrid time (95% interval
5.446--5.777). The strong 10-second pipeline required 6.598 times the derived
neural-hybrid time (95% interval 6.398--6.783).

The hybrid times are component sums, not separately timed end-to-end runs:
the measured constructor latency plus the predeclared 0.5- or 10-second search
budget. This avoids reopening the full quality experiment merely to reproduce
a fixed wall-clock allowance.

## Joint quality--time interpretation

The full 100-instance quality experiment remains authoritative. Neural +
0.5-second LNS achieved 98.330% service with mean objective 964.699, compared
with 99.150% and 491.747 for insertion + 0.5-second LNS, and 99.990% and 8.579
for the 10-second strong-LNS reference. Thus the neural hybrid provides a much
shorter construction pipeline on the specified platform, while the traditional
pipelines retain measurably better solution quality. The evidence supports a
quality--time trade-off, not dominance in both dimensions.

## Integrity checks

- Formal exit code: 0; report status: `passed`.
- Coverage: 30/30 selected instances for each of three neural seeds and strict
  insertion.
- All 120 timed action sequences matched their frozen full-test identities.
- No model weight was updated and no quality checkpoint was reselected.
- Raw evidence bundle SHA-256:
  `e504beba5179c04b1c617d6b7b515d35bc638d19b3dc242e7be4259f434fceff`.
- Raw report SHA-256:
  `e246fe64641c8a7a0ac689e5249861e8f92d331b44de2de7e4ef60324652365d`.
- Remote analysis SHA-256:
  `f559f7278393a0b68c0d1e828246cfa99348fc7f1010d2540f86251195428b51`.

Evidence is stored under `remote_evidence/formal_v2/` and the immutable archive
under `remote_evidence/formal_v2_evidence.tgz`.

