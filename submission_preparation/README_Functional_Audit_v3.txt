CS7002SR — AI Model Artifact Functional Audit v3
NEW ONLY: one new script; no existing files are replaced.

Copy submission_preparation/check_model_function_v3.py into your project's existing
submission_preparation folder. KEEP audit_model_artifact_v2.py unchanged.

From Napa_Cabbage_Project root, use ONE LINE in PyCharm Terminal:

python.exe submission_preparation\check_model_function_v3.py --output submission_preparation\functional_check_results_v3

The script imports existing E3_Session5/evaluate_zero_shot.py to obtain the exact
model classes and checkpoint-loading procedure used in Session 5.
It tests 3 checkpoints, all on CPU with a deterministic 128x128 synthetic tensor:
  - experiments/session3_4_baseline_v2/best_unet.pth
  - experiments/session4_4_transfer_v1/best_resnet18_unet.pth
  - experiments/session6_2_analysis_roi_candidates_v1/resnet18_target_analysis_only.pth
It also lists the columns and row counts of 8 existing experiment CSVs.

Important:
- The checkpoints are from your OWN trusted project. Python source is imported.
- torch.load(..., weights_only=True) is used for a preliminary safe-load check.
- It never changes checkpoints, inputs, existing output folders or code.
- Existing output directory -> refuses to run; choose a fresh --output.
- No model is retrained and no Dice/IoU is calculated by this smoke test.
- Analysis-only model results on the 20 pilot training masks remain in-sample.
- A FAIL in a checkpoint may mean metadata or loading-helper incompatibility;
  send the log and JSON for careful diagnosis. Do not replace prior models.

New results directory:
submission_preparation/functional_check_results_v3/
  model_function_summary_v3.json
  checkpoint_inventory_v3.csv
  result_csv_inventory_v3.csv

Upload a ZIP of this NEW results directory and paste the terminal output.
Do NOT include .pth files in the upload at this stage.
