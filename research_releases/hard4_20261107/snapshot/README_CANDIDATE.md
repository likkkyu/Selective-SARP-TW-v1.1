# Code and evidence candidate for author review

This local candidate has not been uploaded. It contains the exact frozen Python
sources and synthetic input/route/audit/statistical records supporting the current
N50 hard-four-trips study. Every dispatched trip, including the last, completes
at least four passenger/freight requests; pickups cannot be cancelled in execution.

The manuscript, its Word source, illustrations, model checkpoint bytes, old cloud
archives, credentials and downloaded third-party literature are excluded. Six
checkpoint identities and hashes are listed as omitted in CANDIDATE_MANIFEST.json.
Without separately supplied checkpoints this is an archived-evidence/code package,
not a self-contained neural-inference or training reproduction.

Original records are retained byte-for-byte; paths in historical protocol JSON may
refer to the author's local machine. Relocation must occur in a separate replay
copy, while validating unchanged algorithm, checkpoint and scenario hashes. Never
run a historical launcher as a monitoring command or overwrite archived results.

The first and second common studies have distinct sets of 32 synthetic instances.
The three-window experiment reuses the second set and 224 baseline plans. Neural
replication is three fixed training seeds; matched search uses a single fixed seed.
The postprocessing decomposition is post hoc, with pointwise conditional intervals;
it is not a new independent confirmation or evidence of learned-rejection superiority.

Runtime/library observations are in environment_observed.json. Raw and hybrid routes
were independently audited. Limit-success and solver warm-start feasibility are not
proofs of optimality. Existing upstream LICENSE is included without alteration;
dataset/checkpoint reuse terms and final public archive location need author decisions.

The ZIP member hashes are recorded in CANDIDATE_MANIFEST.json and SHA256SUMS.txt.
The manifest is an allow-list, not an instruction to upload the entire working tree.

For dependency-free archived table verification, extract to a new directory and run:

    python3 experiments/publication_release_20261107/recompute_archived_tables.py --output archived_tables_check.json

This checks all archive member hashes, input/source identities and the means/class
rates in tables1–3 and nine window cells. It never opens the old absolute paths.
It does not recompute bootstrap intervals, re-run physical route audits or replace
neural inference reproduction. No model weights, NumPy, PyTorch or solver are needed.
