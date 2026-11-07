# N50 hard-four-trip code and experimental evidence

This author-authorized public snapshot supports the current off-peak passenger and freight scheduling study. Every dispatched trip, including the last, must complete at least four requests; started requests cannot be cancelled. The full manuscript and all six model checkpoint files are intentionally excluded.

## Contents and scope

`snapshot/` preserves frozen source files, synthetic scenarios, archived routes, independent audit records and analyses. It contains two separate 32-instance common studies (352 and 480 recorded plans) and a three-window response study on the second set (672 recorded plans, including 224 reused baseline plans). Plan counts are not independent training replications. Three fixed training seeds and the declared search seeds are reported separately. Both favorable and unfavorable comparisons are retained; the results do not establish learned rejection as superior to the rule comparator, or SCIP optimality at the N50 time limit.

The archived `CANDIDATE_MANIFEST.json` describes assembly before publication; its historical local-candidate status is not the current publication status. Its hashes identify immutable snapshot bytes. Historical JSON may contain local absolute file paths; those paths must be relocated in an isolated replay copy, never by modifying archived evidence.

## Recompute the archived tables without weights

From `snapshot/`, using Python3 standard library only:

```bash
python3 experiments/publication_release_20261107/recompute_archived_tables.py --output archived_tables_check.json
```

This checks all member hashes, frozen source/scenario identities, tables1–3 and nine window means/class rates using instance-first aggregation. It does not perform neural inference, training, route replay, solver runs or bootstrap interval recomputation. The entry point was executed successfully in an independently extracted directory before publication.

## Models and interpretation

Six omitted checkpoints (about191MiB) are identified by policy, training seed, size and SHA256 in the manifest. They have not been publicly distributed; this snapshot alone is not a complete neural-inference reproduction. No replacement checkpoint download URL, model licence or archival DOI is asserted. Existing upstream MIT copyright is retained in `snapshot/LICENSE`; dataset/checkpoint reuse terms remain to be assigned by the author. Files are synthetic experimental data, not privately collected passenger records.

The postprocessing analysis is post hoc and retains all initializers. Intervals are conditional on fixed models/search seeds; no general training-population, universal runtime, real-world causal or optimality guarantee is made. `snapshot/experiments/publication_release_20261107/NUMERICAL_AGREEMENT_REVIEW.json` distinguishes solver/scalar precision from float32 tensor error.

Only this isolated snapshot and release documentation are added. Other branches, historical results and ongoing manuscript files are not replaced.
