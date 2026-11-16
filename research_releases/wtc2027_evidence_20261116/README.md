# WTC 2027 supplementary code and results (N100 + Shenzhen road validation)

This candidate snapshot extends the previously published N50 hard-four-trip evidence at commit `1505564a9fc06cf226d56eae30484adfc7b42574`. It contains the frozen-source inventory, N100 quality/timing/SCIP-alignment analyses, and the Shenzhen road-network external-validation code and per-instance results used by the current WTC manuscript.

## Scientific scope

- Every dispatched trip, including the final trip, must complete at least four passenger and freight requests in total.
- Requests may be declined before service; a picked-up request must be delivered.
- N100 evidence reports three independently fine-tuned neural seeds on 100 untouched common instances, a 30-instance same-node timing subset, and a separate 30-instance SCIP alignment set.
- Shenzhen evidence uses a real directed OpenStreetMap driving graph and synthetic requests. It is a road-network-driven semi-synthetic validation, not an observed-demand or field-operation study.
- Fixed-time LNS budgets and full pipeline times remain distinct. SCIP time-limit results are feasible references and bound interfaces, not optimality certificates.

## Contents

- `snapshot/frozen_code/`: exact frozen source bytes resolved by the manuscript reproducibility inventory.
- `snapshot/results/`: aggregate and per-instance result records, protocols, audits, analysis scripts and publication figures.
- `snapshot/reproducibility/`: claim checks, reference verification and the local evidence inventory.
- `CANDIDATE_MANIFEST.json`: SHA-256, size and provenance for every published byte.
- `verify_release.py`: standard-library-only integrity and exclusion check.

## Deliberate exclusions

Model checkpoints and all `*.pt`/`*.pth` files are excluded, as are logs, process IDs, archives, temporary files and the manuscript itself. The manifest records omitted checkpoint identifiers where available; a hash is not a download route or a reuse licence. Consequently this snapshot supports result-table recomputation and audit, but does not by itself reproduce neural inference or training.

## Verification

From this directory run:

```bash
python3 verify_release.py
```

The verifier checks every manifest member, rejects unlisted files inside `snapshot/`, and confirms that no checkpoint, log or archive suffix entered the candidate.

## Data licensing

Project code retains the repository's existing licence notice. OpenStreetMap-derived road data are separately attributed in `ODBL_ATTRIBUTION.md` and remain subject to the Open Database License. No observed passenger records or private location traces are included.

This snapshot is published on branch `codex/wtc-n100-road-release-20261116`. The first public candidate commit `752dbe53f93daf9558f16efbf7ce2b0f514dd863` was read back before the publication-status finalization commit. Use a fixed commit URL rather than the moving branch name when citing the evidence.
