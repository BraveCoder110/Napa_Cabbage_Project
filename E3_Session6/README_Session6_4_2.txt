Session 6.4.2 — D3 photo-colour / HSV sensitivity diagnosis
NEW script only: E3_Session6/diagnose_photo_color_v1.py
KEEP all old scripts, checkpoints, raw JPEGs, masks, output folders.

This script reads:
- 4 original Huang_Jing_Bai_Cai_D0/D3/D5/D7.zip files
- E3_Session5_1_Pilot20_GT_Final_v1.zip (contains 20 reviewed GT masks)
No auto-mask directories and no checkpoint are required.

Place the GT zip unchanged under experiments/ (copy it there if not already available).
Run from Napa_Cabbage_Project root, using a NEW output directory:

python.exe E3_Session6\diagnose_photo_color_v1.py --d0 datasets\Huang_Jing_Bai_Cai\Huang_Jing_Bai_Cai_D0.zip --d3 datasets\Huang_Jing_Bai_Cai\Huang_Jing_Bai_Cai_D3.zip --d5 datasets\Huang_Jing_Bai_Cai\Huang_Jing_Bai_Cai_D5.zip --d7 datasets\Huang_Jing_Bai_Cai\Huang_Jing_Bai_Cai_D7.zip --gt-zip experiments\E3_Session5_1_Pilot20_GT_Final_v1.zip --stride 4 --output experiments\session6_4_2_photo_color_diagnostics_v1

Outputs:
pilot20_leaf_wood_color.csv
paired_D0_changes_15.csv
by_day_5heads.csv
threshold_consistency_5heads.csv
protocol_and_limitations.json

The 'wood reference' is a diagnostic only: its texture, illumination and shadows vary; it is not a calibrated colour standard. If fewer than two safely sampled patches are available, that image is flagged INSUFFICIENT. Do not interpret colour proxies as validated freshness grades.
