# Lazy-v4 hybrid confirmation result

## Closed protocol

The experiment used the same untouched 100-instance N100 test set
(`seed=2026111901`) as the formal neural confirmation. Each frozen neural output
(three fine-tuning seeds) and the deterministic strict-insertion output received
the same 0.5-second, fixed-seed LNS budget. All instances and starts were
retained. The independent evaluator enforced pickup-before-delivery, capacity,
time-window and ride-time constraints, completion of every accepted pickup,
and at least four completed passenger-plus-cargo orders in every dispatched
trip.

## Main result

| Method | Service rate | Mean unified objective |
|---|---:|---:|
| Neural construction, seed-averaged | 96.033% | 2285.641 |
| Neural + 0.5 s LNS, seed-averaged | **98.330%** | **964.699** |
| Strict insertion | 98.960% | 600.994 |
| Strict insertion + 0.5 s LNS | 99.150% | 491.747 |
| Strict insertion + 10 s LNS | 99.990% | 8.579 |

Relative to neural construction alone, the 0.5-second LNS increased service by
2.297 percentage points (paired-instance bootstrap 95% CI 2.027--2.577) and
reduced the objective by 1320.942 (95% CI 1165.699--1481.940). Service improved
on 95 of 100 paired instances and never decreased; the objective improved on
99 instances and tied on one. All 400 short-LNS outputs had zero hard violation,
zero unfulfilled accepted order, and no dispatched trip with fewer than four
completed orders.

The hybrid neural method did not surpass the matched insertion hybrid. Its
service rate was lower by 0.820 percentage points (95% CI 0.427--1.217), and its
mean objective was higher by 472.952 (95% CI 250.625--701.035). It also remained
below the 10-second strong-LNS reference. These comparisons are retained in the
paper-facing evidence; they are not filtered or relabeled as wins.

## Paper-facing interpretation

The result supports a focused claim: a small, fixed neighborhood-search budget
substantially and reproducibly repairs the feasible neural construction under
the strict four-order trip constraint. It does not support a claim that the
neural or hybrid neural method gives the best solution quality. The appropriate
paper narrative is therefore a neural construction plus neighborhood
improvement framework with an explicit quality--computation trade-off, while
strict insertion and strong LNS remain competitive quality references.

The current wall-clock records are not sufficient for a publication-grade
cross-method speed ratio. Neural inference was timed on an AutoDL GPU node,
whereas insertion and LNS were timed on the local CPU. The 0.5-second LNS budget
is matched across starts on the same local CPU, so its quality comparison is
valid; end-to-end speed claims require a same-machine timing benchmark.

## Evidence

- Frozen protocol: `PLAN.json`
- Full per-instance result: `results/REPORT.json`
- Paired bootstrap analysis: `PAIRED_ANALYSIS.json`
- Experiment runner: `run.py`
- Analysis script: `analyze.py`
