# Cost-v4 research snapshot

Run from this directory. Source hashes are recorded in SOURCE_MANIFEST.json. Entry: python experiments/sensitivity_20261030/corrected_entry.py --stage train --policy learned --record RUN_DIR [training options]. Use --stage eval for cost-v4 checkpoints and independent auditing. Environment dependencies follow repository environment.yml. Cost-v3 checkpoints must not be evaluated as cost-v4 models.

Seeds 4321/8765 were still in progress at snapshot time. This is a source archive, not proof of completed experiments. Historical sensitivity batch.py/evaluate.py currently uses the old cost entry: do not use it for v4 sensitivity without adapting and testing version restoration.
