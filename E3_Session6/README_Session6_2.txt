E3 SESSION 6.2 — FINAL ANALYSIS MODEL + 157 CANDIDATE ROIs
NEW FILES ONLY: E3_Session6/train_final_roi_model.py and README_Session6_2.txt
KEEP all existing scripts, checkpoints and experimental outputs unchanged.

Run from Napa_Cabbage_Project root with the command given in ChatGPT.
Requires the existing E3_Session5/evaluate_zero_shot.py and
E3_Session5/run_grouped_few_shot.py, the E3_Session3 and E3_Session4 model classes,
20 pilot JPG files, 20 reviewed masks and four original day ZIP archives.

NEW output directory:
experiments/session6_2_analysis_roi_candidates_v1/
  resnet18_target_analysis_only.pth   - analysis use, NOT a CV fold or source checkpoint
  training_history.csv
  masks_128/                            - 157 images, binary 128x128
  masks_full_resolution/                - 157 masks at 1868x4000, upsampled from 128x128
  candidate_roi_manifest_157.csv
  previews_by_day/D0_all_candidates.jpg (and D3,D5,D7)
  analysis_model_protocol.json

The program refuses to run into an existing output directory. For reruns use a new
versioned output name. Inputs are never edited. Training-only augmentation remains
identical to Session 5.3. Same 20 GT images are TRAINING data for the final analysis
model; reported five-fold OOF Dice=0.9483 is the pilot evaluation reference, and
CANNOT be treated as the independent accuracy of this trained-on-all-20 model.

All 157 ROI masks are MACHINE CANDIDATES; verify the leaf/petiole vs shadow/table
boundary on all preview sheets before using them to calculate longitudinal changes.
Especially review any ROI touching borders, obviously too large/small, and all
reverse-side and L02-L04 photos that were not part of the labelled pilot.

Freshness caution: day is an observation time, not a scientifically validated
freshness or food-safety class. Original RGB must be used for color features.
