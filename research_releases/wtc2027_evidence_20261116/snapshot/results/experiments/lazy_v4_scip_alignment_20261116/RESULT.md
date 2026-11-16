## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-11-16
- Verification Status: VERIFIED
- Version Label: lazy_v4_scip_alignment_v1

# Current N100 neural-hybrid alignment with insertion and SCIP

## Protocol

Three frozen cost-v4 lazy-certificate checkpoints (fine-tuning seeds 314159,
271828 and 161803) were evaluated on the exact 30-instance N100 set previously
used for strict insertion and one-thread, 300-second SCIP. Seed regeneration
matched the frozen scenario file exactly for all 240 tensor fields and 30
non-tensor fields. Neural results were averaged across training seeds within each
instance before paired inference. Every plan passed the independent physical
audit, left no accepted order unfinished, and completed at least four
passenger-plus-cargo orders on every dispatched trip.

The 0.5-second LNS budget and search seed were inherited unchanged from the
previously closed N100 hybrid protocol. All 90 neural starts were retained.

## Main result

| Method | Service rate | Mean objective J | Mean local-CPU time |
|---|---:|---:|---:|
| Current neural construction | 95.60% | 2534.58 | 2.05 s |
| Neural + 0.5 s LNS | 97.90% | 1211.79 | 2.55 s |
| Strict insertion | 99.27% | 424.63 | 36.06 s |
| 300 s SCIP incumbent | 99.27% | 424.63 | 344.75 s |

Short LNS increased neural service by 2.300 percentage points (paired-instance
bootstrap 95% CI 1.822--2.800) and reduced the objective by 1322.79 (95% CI
1048.08--1616.73 in magnitude). Service improved on 29/30 paired instances and
tied on one; the objective improved on all 30.

The neural hybrid retained a 1.367-point service gap to strict insertion (95%
CI 0.800--1.944 points) and a mean objective difference of 787.16 (95% CI
461.18--1125.80), while returning in 2.55 seconds rather than 36.06 seconds on
the same Apple M2 host. The observed insertion-to-neural-hybrid time ratio was
14.12. These are hardware-specific deployment observations from separate
sessions, not hardware-independent complexity ratios.

All 30 SCIP runs reached the time limit, none proved optimal, and none improved
the strict-insertion warm start. The SCIP row therefore characterizes the
failure to improve or certify a strong incumbent under a 300-second budget; its
solution quality is attributable to strict insertion, not to SCIP search.

## Paper-facing interpretation

This alignment reinforces the quality--response-time positioning. A fixed
0.5-second neighborhood stage consistently repairs the neural construction and
produces audited N100 plans in a few seconds. Strict insertion provides a
higher-quality endpoint at a larger response-time cost, while the current MILP
formulation does not improve or certify that endpoint within 300 seconds. The
evidence supports presenting the method as a fast neural-hybrid candidate in a
deployment frontier, not as a universal solution-quality winner.

This 30-instance study is a post hoc alignment evaluation on an existing SCIP
set. The primary scale evidence remains the separate, untouched 100-instance
N100 test reported in the manuscript.

## Evidence

- Frozen raw protocol: `PLAN.json`
- Exact-data gate: `DATASET_EQUIVALENCE.json`
- Frozen short-LNS protocol: `SHORT_LNS_PLAN.json`
- Raw current-model analysis: `PAIRED_ANALYSIS.json`
- Hybrid analysis: `HYBRID_PAIRED_ANALYSIS.json`
- Full short-LNS outputs: `results/SHORT_LNS_REPORT.json`
- Statistical validation: `VALIDATION_REPORT.md`
