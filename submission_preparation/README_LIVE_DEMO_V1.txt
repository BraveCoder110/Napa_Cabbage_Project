CS7002SR Live Cabbage Leaf Demo V1 — NEW ONLY
================================================

PURPOSE
- Take one new JPG/PNG cabbage leaf photo and display all processing stages.
- Uses your ALREADY-TRAINED Session 6.2 ResNet18UNet analysis-only checkpoint.
- Does NOT train; does NOT compute a fresh/spoiled grade, Dice or food safety.

FILES IN THIS PACKAGE
- live_leaf_demo.py: only new Python program.
- CS7002SR_Live_Photo_Demo_Step_by_Step_Manual_CN_EN_v1.docx: detailed Chinese guide + simple English speeches.
- README_LIVE_DEMO_V1.txt: this file.

COPY
Copy ONLY live_leaf_demo.py into:
  Napa_Cabbage_Project/submission_preparation/
Never overwrite older scripts, .pth files, or results.

BEFORE THE SHOW
- Python environment used for passed V3/V5 checks; requirements: torch, torchvision, numpy, opencv-python, Pillow.
- Existing model definition: E3_Session4/models/resnet18_unet.py
- Existing real checkpoint: experiments/session6_2_analysis_roi_candidates_v1/resnet18_target_analysis_only.pth
- Prepare a NEW photo .jpg/.png as submission_preparation/live_demo_inputs_v1/live_leaf_01.jpg.

RUN from Napa_Cabbage_Project root:
  python.exe submission_preparation\live_leaf_demo.py --image "submission_preparation\live_demo_inputs_v1\live_leaf_01.jpg" --output "submission_preparation\live_demo_results_live01_v1"

Or select the photo in a file-picker:
  python.exe submission_preparation\live_leaf_demo.py --output "submission_preparation\live_demo_results_picker01_v1"

Open NEW OUTPUT/demo_report.html in your browser. It is entirely local/offline.
Each run requires a NEW output folder; the program refuses to overwrite an existing one.

IMPORTANT RESEARCH BOUNDARIES
- The analysis-only checkpoint trained on all 20 target GT masks was NOT tested on independent new live photos as part of the validated 0.9483 Dice LOHO protocol.
- New photos have no ground truth so Dice cannot be calculated.
- HSV green/yellow/brown candidate ratios are heuristic visual features, not validated freshness labels.
- Background, lighting, pose and phone white balance can change outputs.
- Full-res output mask is a nearest-neighbor upsample of the 128x128 prediction.

SOFTWARE QA
- This script was tested against a SYNTHETIC checkpoint to verify execution, file creation and empty/full ROI handling.
- It MUST still be validated locally with your REAL Session6.2 checkpoint before being described as fully checked.
