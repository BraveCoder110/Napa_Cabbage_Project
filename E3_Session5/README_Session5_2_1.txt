Session 5.2.1 DIAGNOSIS (NEW FILES ONLY)
=====================================
COPY exactly one new Python file to:
  Napa_Cabbage_Project/E3_Session5/diagnose_zero_shot.py

KEEP ALL existing code, checkpoints and outputs unchanged.
This script imports your already-working E3_Session5/evaluate_zero_shot.py.

RUN FROM Napa_Cabbage_Project ROOT, in PyCharm Terminal:

python.exe E3_Session5\diagnose_zero_shot.py --images experiments\session5_1_target_annotation_v1\images --masks experiments\Session5_1_GT_Pilot20_Final_v1\masks --custom-checkpoint experiments\session3_4_baseline_v2\best_unet.pth --transfer-checkpoint experiments\session4_4_transfer_v1\best_resnet18_unet.pth --source-images E3_Session2\Bok_Choy_Dataset.zip --source-masks E3_output\binary_masks.zip --output experiments\session5_2_1_diagnostics_v1

If your PUBLIC source image ZIP is in a different folder, change only --source-images.
If your GT mask folder is named differently, change only --masks.

REQUIRED FILES:
  - 20 original target JPGs
  - 20 reviewed ground-truth PNGs
  - both preexisting best model checkpoints
  - both preexisting source ZIPs for full source validation verification

EXPECTED OUTPUTS:
  diagnosis_summary.json
  target_per_image_diagnostic.csv (60 rows: 20 x [all foreground, custom, resnet18])
  target_probability_diagnostic.csv (40 rows)
  threshold_sweep_DIAGNOSTIC_ONLY.csv (12 rows)
  source_per_image_diagnostic.csv (230 rows when full source validation requested)

If source ZIPs are unavailable, omit BOTH --source-images and --source-masks.
The source reproduction will be labeled NOT RUN; target diagnostics still run.

The script refuses to reuse the output directory. Never delete old experiments to force rerun;
use a NEW output name ending v2 if needed.

SCI/RIGOR NOTES:
- All-foreground is a zero-learning baseline.
- The 0.1-0.99 threshold sweep is a DIAGNOSTIC; selecting a threshold on the same
  evaluation data and claiming independent generalization would be leakage.
- Reproducing both source checkpoints helps distinguish a pipeline problem from
  a target-domain shift, but cannot itself prove causality.
- 20 target images from 5 heads are repeated measures, NOT 20 independent biological samples.
- Model inference here supports visual freshness assessment by segmenting the tissue,
  but Dice is NOT a freshness score.
