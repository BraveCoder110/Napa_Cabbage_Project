# V4 Evidence Consistency Audit (NEW ONLY)

Place `audit_experiment_evidence_v4.py` in `Napa_Cabbage_Project/submission_preparation/`.
Run from PyCharm terminal at project root:

```powershell
python.exe submission_preparation\audit_experiment_evidence_v4.py --output submission_preparation\evidence_audit_results_v4
```

Do NOT reuse an existing output directory: the tool refuses overwrite. Standard Python library only; no training, no modification to code, datasets, or checkpoints.

Outputs:
- `evidence_checks_v4.csv`: individual PASS/WARN/FAIL results
- `recomputed_metrics_v4.csv`: global Dice and IoU recomputed from stored TP/FP/FN
- `evidence_summary_v4.json`: machine-readable findings
- `evidence_summary_v4.md`: reviewer-friendly report

This is a saved-evidence **consistency** audit. It does not rerun neural network inference. It checks the five-fold leave-one-head-out membership (16 train and 4 held out in each fold), evaluates pooled segmentation metrics using actual per-image confusion counts, and checks longitudinal sample counts.

Source dataset copyright and redistribution permission must be manually reviewed, even if LICENSE.txt exists.

If source-validation CSVs lack per-image confusion counts, V4 will warn instead of inventing pooled results.

Provide terminal output and a ZIP of `submission_preparation/evidence_audit_results_v4` for review.
