AI Model Artifact — V3.1 analysis-only ResNet18 checkpoint check

Why:
V3 passes both source models but rejects the Session 6.2 analysis checkpoint
before inference. The original loader expects source-training metadata,
including pretrained_encoder. The analysis checkpoint has different metadata.
This does not by itself indicate bad or corrupted weights.

Files:
Copy check_analysis_checkpoint_v3_1.py into submission_preparation/.
Keep all existing project files unchanged.

From Napa_Cabbage_Project root in PyCharm Terminal:

python.exe submission_preparation\check_analysis_checkpoint_v3_1.py --output submission_preparation\analysis_checkpoint_results_v3_1

The script uses only your existing torch/numpy and the project's existing
E3_Session4/models/resnet18_unet.py. It never trains, edits checkpoints,
connects to the Internet or claims real-image accuracy. It:
- safely loads source and analysis weights using weights_only=True;
- compares all state_dict parameter keys and tensor shapes;
- imports your current ResNet18 U-Net definition;
- strictly loads analysis weights into that class;
- tests a deterministic synthetic 128x128 RGB tensor on CPU;
- saves JSON + Markdown diagnostic files to a NEW output directory.

Output:
submission_preparation/analysis_checkpoint_results_v3_1/
  analysis_checkpoint_check_v3_1.json
  analysis_checkpoint_check_v3_1.md

The script refuses to overwrite an existing output directory.
Send both the console output and zipped new results for review.

NOTE: A PASS does not reproduce independent Dice/IoU and the analysis-only
model was trained on all 20 pilot images. It must never be represented as a
held-out test checkpoint. Fold-specific checkpoints may not have been saved.
