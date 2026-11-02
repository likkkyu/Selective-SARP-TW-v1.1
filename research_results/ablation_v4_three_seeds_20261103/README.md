# Completed rejection-timing ablation: three training seeds

Cost version4; learned vs service_first; seeds1234/4321/8765; 20 epochs x2048 training instances; validation500; batch16/POMO1/batch_mean. Common independent test seed917531,500 instances/model. All6 models passed independent hard-constraint audits within declared tolerance, zero unfinished orders. The same500 instances evaluated by6 models are not3000 independent instances.

Mean service rates:91.691% learned vs91.233% rule. Paired difference:+0.457 percentage points, sample SD1.885pp; seed differences:+2.240,-1.516,+0.648pp. This does not demonstrate stable or significant superiority. Read ASSESSMENT.md before using these results in a paper. No seed was omitted.

Evaluation does NOT yet use completion guard1 introduced in the sensitivity study. Common-rule reevaluation is pending. This archive is a completed run backup, not a claim that the final paper experiment is fully closed.

## Contents and storage

- raw_summary.json,results.csv,paired_statistics.json,summarize.py: observed results and reproducible descriptive statistics.
- runs/: original manifests, evaluations, per-instance independent audits and training logs where saved.
- protocols/: original run/time-extension records. Historical diagnostic scope in manifests is preserved; formal run protocols identify actual intent.
- FULL_CLOUD_MANIFEST.json: complete source inventory. SELECTED_NEW.json/REUSED_EXISTING.json/LOCAL_VERIFICATION.json document selective backup; twelve alternate-selection checkpoints not used for reporting are explicitly omitted.
- Matched cost4 core/source: ../../research_snapshot_20261102; original source hashes are recorded in the run manifests and local raw snapshots. Operational shell scripts retain cloud paths and must be reviewed before reuse.
- Local complete relevant files and12 best/final checkpoints: artifacts/ablation_final_20261103/raw in the user's project. New archive337538815 bytes; SHA2561512ad3d1ed01d437b0b9de6691cd73ed5e3f844b41177d01d98d5c3400bc3a7. Large weights and progress logs are not uploaded to Git.

No additional training or evaluation was launched during archival. Model checkpoint selection follows validation business_first, not best test score. Future evaluation must keep all groups on the same cost and safety rules.
