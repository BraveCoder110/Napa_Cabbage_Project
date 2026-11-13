SESSION 5.3 — Five-head grouped few-shot adaptation (NEW FILES ONLY)
=====================================================================
OBJECTIVE
Test whether small-sample adaptation reduces wooden-background false positives
while keeping all images from a held-out biological source head unseen during
that fold's adaptation. This is a PILOT study, not freshness classification.

FILES TO COPY
NEW: E3_Session5/run_grouped_few_shot.py
KEEP: E3_Session5/evaluate_zero_shot.py
KEEP: All Session 3/4 source-trained checkpoints, old code and prior outputs.

Run from Napa_Cabbage_Project root (Windows PyCharm Terminal):

python.exe E3_Session5\run_grouped_few_shot.py --images experiments\session5_1_target_annotation_v1\images --masks experiments\Session5_1_GT_Pilot20_Final_v1\masks --custom-checkpoint experiments\session3_4_baseline_v2\best_unet.pth --transfer-checkpoint experiments\session4_4_transfer_v1\best_resnet18_unet.pth --model resnet18 --epochs 3 --lr 0.0001 --output experiments\session5_3_resnet18_loho_v1

If --masks directory differs, use the actual final GT masks directory.

PROTOCOL
Five folds, each holding out all D0/D3/D5/D7 images from one biological
source head. Every fold reinitializes from the ORIGINAL source-trained model.
Four heads (16 images) are used for training; one head (4 images) for testing.
Training-only paired geometric augmentation and mild RGB light variation.
The 0.5 threshold is pre-specified. Learning rate / epochs are fixed and not
selected by looking at held-out target outcomes.

OUTPUTS (new folder only)
oof_per_image_metrics.csv: 20 held-out image predictions, exactly once.
oof_group_metrics.csv: 5 head-level scores and overall pooled Dice/IoU.
fold_membership.csv: proof each image is train OR held-out test for each fold.
training_history.csv: train losses only, no test-based early stopping.
protocol_summary.json: reproducibility metadata and methodological caveat.

IMPORTANT LIMITATIONS
- The 20 target images come from only five independent cabbage heads.
- All 20 samples were already used in Session 5.2 zero-shot diagnostics, so
  the adaptation result is an out-of-fold pilot estimate, not a fully pristine
  independent external validation.
- All source-domain model weights stay untouched.
- Do NOT train on the 157 automatically generated unreviewed candidate masks;
  they could reinforce labeling mistakes such as wooden shadow false positives.
- This training changes ONLY the segmentation model, not a validated freshness
  classifier. Day0/3/5/7 are not ground-truth freshness classes.
- CPU training may take some time; progress prints after each epoch and fold.
- Folder already exists => script refuses overwriting; use a _v2 output name.

If the model cannot run due to missing relative import, ensure the original
E3_Session5/evaluate_zero_shot.py remains next to this NEW script.
