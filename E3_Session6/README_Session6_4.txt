E3 Session 6.4 - GT vs Auto ROI Sensitivity Diagnostics
========================================================
NEW FILE ONLY: E3_Session6/compare_gt_auto_roi_features.py
KEEP: all prior E3_Session6 and E3_Session5 scripts, original data, models and outputs.

Place the NEW script beside extract_visual_freshness_features.py.
From Napa_Cabbage_Project ROOT run:

python.exe E3_Session6\compare_gt_auto_roi_features.py --d0 datasets\Huang_Jing_Bai_Cai\Huang_Jing_Bai_Cai_D0.zip --d3 datasets\Huang_Jing_Bai_Cai\Huang_Jing_Bai_Cai_D3.zip --d5 datasets\Huang_Jing_Bai_Cai\Huang_Jing_Bai_Cai_D5.zip --d7 datasets\Huang_Jing_Bai_Cai\Huang_Jing_Bai_Cai_D7.zip --gt-masks experiments\Session5_1_GT_Pilot20_Final_v1\masks --auto-masks experiments\session6_2_analysis_roi_candidates_v1\masks_full_resolution --stride 4 --output experiments\session6_4_gt_auto_sensitivity_v1

Output (new directory, refused if exists):
 per_image_GT_vs_Auto_20.csv
 paired_D0_changes_GT_vs_Auto_15.csv
 by_day_GT_vs_Auto.csv
 diagnosis_summary.json

What it does:
- Reads exactly 20 selected L01 front images (5 source heads x four days) from original day ZIPs.
- Uses EXIF transpose consistently with both GT and automatic ROI masks.
- Reuses EXACT Session6.3 HSV and texture metric definitions.
- Computes GT-vs-automatic mask Dice, IoU, FP and FN and visual feature differences.
- Computes 15 within-head, same-leaf/side D0-to-D3/D5/D7 comparisons using each mask source.
- Never overwrites images, masks, models or previous results.

Research cautions:
- Final analysis model was trained on all these 20 GT labels: this check is IN-SAMPLE, not independent segmentation validation.
- No conclusion about authentic freshness or yellowing is possible solely from HSV threshold outcomes.
- If D3 remains elevated with GT, test camera white balance/lighting or physiological effects before biological claims.
- Source heads (n=5), not the 20 date images, are the independent biological groups.
