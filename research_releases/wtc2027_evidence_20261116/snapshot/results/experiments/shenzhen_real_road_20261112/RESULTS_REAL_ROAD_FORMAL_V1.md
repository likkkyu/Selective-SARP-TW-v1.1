# Shenzhen road-native formal test (v1)

## Design boundary

- Frozen test: 128 independent N50 scenarios on the Shenzhen Futian–Luohu OSM driving graph.
- Demand: controlled synthetic pickup–delivery requests sampled directly on road nodes; not observed trips.
- Neural replication: three independently trained seeds (1234, 4321, 8765).
- Search: one fixed LNS seed (1234), 3 s per instance. No p-values are claimed across search seeds.
- Every reported route passed the independent hard-constraint audit, including the four-completed-orders minimum for every dispatched trip.

## Primary results

Values are instance means with paired-instance bootstrap 95% CIs; neural rows first average the three frozen training seeds within each instance.

| Method | Service rate, % | Objective | Vehicles | Algorithm time, s | Full-service instances* |
|---|---:|---:|---:|---:|---:|
| Neural construction | 90.44 [89.54, 91.24] | 2753.19 [2522.71, 3019.65] | 8.42 [8.33, 8.51] | 0.244 [0.239, 0.250] | 0/128 |
| Neural + 3 s LNS | 99.99 [99.97, 100.00] | 6.43 [3.42, 10.94] | 8.16 [8.10, 8.21] | 3.245 [3.239, 3.250] | 126/128 |
| Strict insertion | 97.00 [96.45, 97.52] | 865.96 [717.74, 1023.13] | 7.98 [7.89, 8.08] | 0.105 [0.103, 0.107] | 41/128 |
| Insertion + 3 s LNS | 99.97 [99.92, 100.00] | 12.41 [3.41, 25.89] | 8.12 [8.05, 8.19] | 3.105 [3.103, 3.107] | 126/128 |

## Paired interpretation

- Matched LNS improved neural construction by 9.552 percentage points (95% CI 8.724 to 10.479).
- Neural + LNS minus insertion + LNS was 0.021 percentage points (95% CI -0.016 to 0.073).
- These intervals quantify variation across the 128 frozen scenarios conditional on the fixed models and search seed; they do not quantify all possible training/search-seed variation.

## Training-seed stability

- Pure neural service rate: 90.438% ± 0.939 percentage points (SD across 3 seeds).
- Neural + LNS service rate: 99.990% ± 0.018 percentage points (SD across 3 seeds).
- Pure neural full-service counts by training seed (1234/4321/8765): 1/5/5 of 128.
- Neural + LNS full-service counts by training seed (1234/4321/8765): 128/128/126 of 128.
- *For neural rows, the table count requires the same instance to be fully served by all three training seeds; control rows have one fixed run per instance.

## Bounded MIP reference

- On the predeclared first 16 frozen test instances, hard-four SCIP used one CPU thread and 5 s per instance.
- The strict-insertion warm start and final SCIP incumbent both served 97.375% of orders with mean objective 758.166; SCIP improved the warm-start objective in 0/16 instances.
- All 16 runs reached the time limit and 0 proved optimality. Therefore these are finite-time MIP incumbents, not exact optima or lower-bound quality claims.
- Every incumbent passed the independent road-route and hard-four audit; maximum objective replay error was 4.44e-16.

## Integrity note

An earlier decoder-v1 evaluation exposed one numerical boundary deadlock. The original failure artifact is retained. Decoder v2 aligns the hard-feasibility tolerance with the independent audit and refuses unsafe depot fallback when pickup commitments remain. All final results above were recomputed from unchanged checkpoint weights on the unchanged frozen test set.
