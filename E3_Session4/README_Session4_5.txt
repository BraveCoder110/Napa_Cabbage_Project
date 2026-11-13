E3 Session 4.5 - Paired Model Comparison

NEW: E3_Session4/compare_models.py
KEEP: all previously existing scripts, checkpoints, results, and masks.

From Napa_Cabbage_Project root:

python.exe E3_Session4\compare_models.py --custom experiments\session3_4_evaluation_v2\per_image_metrics.csv --transfer experiments\session4_4_evaluation_v1\per_image_metrics.csv --output experiments\session4_5_comparison_v1 --top-k 10

This script:
- requires exact matching image IDs between CSVs;
- creates a new output directory and refuses existing ones;
- saves paired_metrics.csv, worst10 lists, improvement lists, summary and optional plot;
- performs no model training and overwrites nothing;
- reports per-image means, NOT global pixel-aggregated Dice/IoU.

Required CSV columns: image, dice, iou.
