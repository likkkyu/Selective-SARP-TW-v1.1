## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-11-16
- Verification Status: VERIFIED
- Version Label: validation_v1

## Validation Report

- **Source**: `lazy_v4_scip_alignment_20261116`
- **Overall Confidence**: CAUTION

The reported descriptive and paired results are computationally verified. The
CAUTION rating concerns confirmatory scope: current models were evaluated after
the SCIP scenario set already existed, so this alignment should complement, not
replace, the untouched 100-instance N100 study.

### Statistical Findings

| Metric | Test | Value | Effect size | Confidence |
|---|---|---:|---|---|
| Hybrid minus raw neural service | Paired instance bootstrap, 20,000 resamples | +2.300 pp; 95% CI [1.822, 2.800] | Operationally material repair | SOLID |
| Hybrid minus raw neural objective | Paired instance bootstrap | -1322.79; 95% CI [-1616.73, -1048.08] | Large relative to the starting mean | SOLID |
| Hybrid minus insertion service | Paired instance bootstrap | -1.367 pp; 95% CI [-1.944, -0.800] | Modest quality gap | SOLID |
| Hybrid minus insertion objective | Paired instance bootstrap | +787.16; 95% CI [461.18, 1125.80] | Insertion-quality advantage | SOLID |
| Local CPU response time | Deterministic descriptive comparison | 2.55 s vs 36.06 s | 14.12× observed ratio | CAUTION |

### Warnings

| Type | Detail | Affected |
|---|---|---|
| Post hoc alignment | Models were not selected on these 30 instances, but the alignment evaluation was conducted after baseline outcomes existed. | Generalization and confirmatory wording |
| Session timing | Methods used the same Apple M2 host and one-thread evaluators but ran in separate sessions. | Timing ratio |
| Solver certificate | All SCIP runs timed out; no incumbent was proven optimal and no warm start was improved. | Any “SCIP optimum” claim |
| Training uncertainty | Three fine-tuning seeds share one legacy source checkpoint; this is fine-tuning-seed variability, not three independent from-scratch training programs. | Training-replicate interpretation |

### Fallacy Scan

- **Coverage**: 11/11 fallacy types checked

| Fallacy | Severity | Detail | Recommendation |
|---|---|---|---|
| Simpson's paradox | NOTE | Training-seed directions and instance-paired aggregate results do not reverse the main hybrid-repair direction; seed-specific results remain available. | Keep seed-level table in supplementary evidence. |
| Ecological fallacy | NOTE | Inference is explicitly at the generated-instance distribution level, not at individual passenger or city level. | Preserve the stated unit of inference. |
| Berkson's paradox | NOTE | No outcome-based instance filtering occurred; all 30 frozen instances were retained. | No action required. |
| Collider bias | NOTE | No regression adjustment or conditioned mediator/collider is used. | Not applicable. |
| Base-rate neglect | NOTE | The experiment reports complete service proportions, rejected counts and paired denominators rather than diagnostic accuracy. | Not applicable. |
| Regression to the mean | NOTE | Instances were not selected for extreme neural performance, and comparison uses the same frozen instances. | No action required. |
| Survivorship bias | NOTE | All instances and all three model seeds are included; the failed pre-inference concurrent worker was transparently rerun serially with its failure record retained. | Preserve the run-status record. |
| Look-elsewhere effect | CAUTION | Several cost and service metrics are reported, but the central service/objective comparisons were predeclared and no outcome was filtered. | Treat secondary components as descriptive; do not claim multiplicity-adjusted discovery. |
| Garden of forking paths | CAUTION | LNS budget was selected before this alignment outcome, but the alignment itself was added after baseline evidence existed. | Label this study post hoc and prioritize the independent 100-instance test. |
| Correlation != causation | NOTE | The result is an algorithmic intervention on fixed starts, not an observational causal claim about real operations. | Avoid real-world causal extrapolation. |
| Reverse causality | NOTE | Not applicable to deterministic algorithm evaluation. | No action required. |

### Reproducibility

- **Method**: deterministic scenario reconstruction, frozen hashes, retained raw
  reports, independent route audit, scalar/tensor cost replay, and deterministic
  re-analysis.
- **Verdict**: REPRODUCIBLE

| Check | Original | Re-run/verification | Difference | Status |
|---|---|---|---:|---|
| Scenario data | Frozen `.pt`, SHA256 `d874...ffba` | Seed reconstruction, 240 tensor + 30 non-tensor fields | 0 | MATCH |
| Multiscale paired analysis | Stored JSON/CSV | Fresh analyzer run | Byte-identical | MATCH |
| Current-model hard audit | Three checkpoints × 30 instances | Independent route/cost audit | 0 violations | MATCH |
| Short-LNS hard audit | 90 outputs | Independent route/cost audit | 0 violations | MATCH |
| SCIP improvement | 0/30 | Re-read all instance reports | 0 | MATCH |
