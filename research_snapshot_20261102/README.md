# Verified sensitivity inference snapshot

Cost semantics 4; completion safety rule 1; learned checkpoint seed1234. Frozen source recovered from the completed cloud run, with local regression test added. Run from this directory. Use experiments/sensitivity_20261030/batch_v4_guarded.py or evaluate_v4_guarded.py; old evaluate.py/batch.py are retained for provenance and must not be substituted for guarded evaluation. The matched model is in the local verified backup (SHA256 in the export manifest), not stored in Git. Exported shell launchers contain historical relative paths; review arguments before execution. Environment dependencies follow environment.yml.

Current completed result: N50, 17 scenarios x 500 instances, one training seed. Every independent audit passed within declared tolerance. This is not a three-training-seed conclusion. Root repository code and historical snapshots remain separate.
