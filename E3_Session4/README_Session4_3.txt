E3 Session 4.3 — NEW FILES ONLY

NEW: E3_Session4/train_transfer_smoke_test.py
KEEP: E3_Session4/models/resnet18_unet.py
KEEP: E3_Session3/train_smoke_test.py (dataset and loss utilities)
KEEP: all existing models, outputs, and checkpoints.

Run from Napa_Cabbage_Project root in PyCharm Terminal:
python.exe E3_Session4\train_transfer_smoke_test.py --images E3_Session2\Bok_Choy_Dataset.zip --masks E3_output\binary_masks.zip --output experiments\session4_3_smoke_v1 --steps 2

Uses official pretrained ResNet18 weights (already cached from Session 4.1). The script refuses to reuse an existing output path. It uses real image/mask pairs and writes a smoke-test checkpoint, log, and config. Do not use this checkpoint as final evaluation.
