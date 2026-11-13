Session 5.2 — Zero-shot Cross-Domain Evaluation (NEW files ONLY)

DATE: 2026-11-02
NEW: E3_Session5/evaluate_zero_shot.py
KEEP UNCHANGED: all existing scripts, checkpoints, original photos, Session 5.1 JSON and masks.

1. Download and extract this kit into the Napa_Cabbage_Project ROOT.
   This adds E3_Session5/evaluate_zero_shot.py only.
   If this file already exists, DO NOT overwrite it; rename the NEW file before extraction.

2. The Session 5.1 GT ZIP contains a top-level folder named Session5_1_GT_Pilot20_Final_v1.
   Extract to experiments without renaming its inner folder, or adjust --masks below.

3. The pilot RGB input folder already exists at:
   experiments/session5_1_target_annotation_v1/images
   (Earlier export contains files named D0C01L01正.jpg; this script supports both that and C01L01正.jpg.)
   Alternatively use the 20 images from Session 5.1 Pilot20 Review kit.

4. Run from PROJECT ROOT in PyCharm Terminal:

python.exe E3_Session5\evaluate_zero_shot.py --images experiments\session5_1_target_annotation_v1\images --masks experiments\Session5_1_GT_Pilot20_Final_v1\masks --custom-checkpoint experiments\session3_4_baseline_v2\best_unet.pth --transfer-checkpoint experiments\session4_4_transfer_v1\best_resnet18_unet.pth --output experiments\session5_2_zero_shot_v1

If your final GT folder is experiments\session5_1_gt_pilot20_final_v1\Session5_1_GT_Pilot20_Final_v1,
change --masks to the ACTUAL inner masks directory. Do not move or overwrite existing GT.

5. Outputs (only new folder):
   per_image_metrics.csv: 40 rows (20 images x 2 models), Dice/IoU and TP FP FN TN
   group_metrics.csv: global pooled Dice/IoU for ALL, each day and each biological source head
   summary.json: measured target metrics and exact preprocessing information
   previews/: 5 contact sheets of RGB / GT / custom prediction / pretrained ResNet18 prediction

These are ZERO-SHOT results: neither model is updated. RGB is resized to 128x128
with bilinear interpolation and scaled [0,1]. Mask resized to 128x128 with NEAREST
interpolation. Threshold: sigmoid(logits)>=0.5. All images kept in EXIF-correct
portrait orientation. Metrics apply to 128x128 evaluation grid, not 1868x4000.

IMPORTANT:
- 20 photographs belong to only 5 independent source heads; do not treat as 20
  biological replicates. Pilot set covers L01 fronts only, not all 157 photos.
- The script checks all 20 pairs and refuses an existing output directory.
- A local functional test was performed using ARTIFICIAL RANDOM model weights,
  not your real trained checkpoints. Fake test results have NOT been included.
- The trained checkpoints are not present in this chat's filesystem, so no
  genuine zero-shot Dice/IoU has yet been measured here.
- Upload summary.json, per_image_metrics.csv, group_metrics.csv and five preview
  sheets (or ZIP the entire new result directory) for interpretation.
