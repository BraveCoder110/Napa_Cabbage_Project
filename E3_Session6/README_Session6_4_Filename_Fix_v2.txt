Session 6.4 filename mismatch patch v2 — NEW FILE ONLY
======================================================
Problem:
  Day-0 original JPEG and reviewed GT mask: C01L01正.jpg / C01L01正.png
  Day-0 Session 6.2 automatic mask:   D0C01L01正.png
  Day-3/5/7 filenames already match in both mask folders.

Fix:
  Place E3_Session6/compare_gt_auto_roi_features_v2.py in the existing E3_Session6 directory.
  DO NOT overwrite compare_gt_auto_roi_features.py, extract_visual_freshness_features.py,
  any checkpoints, masks, or previous experimental outputs.

Run from Napa_Cabbage_Project root:
python.exe E3_Session6\compare_gt_auto_roi_features_v2.py --d0 datasets\Huang_Jing_Bai_Cai\Huang_Jing_Bai_Cai_D0.zip --d3 datasets\Huang_Jing_Bai_Cai\Huang_Jing_Bai_Cai_D3.zip --d5 datasets\Huang_Jing_Bai_Cai\Huang_Jing_Bai_Cai_D5.zip --d7 datasets\Huang_Jing_Bai_Cai\Huang_Jing_Bai_Cai_D7.zip --gt-masks experiments\Session5_1_GT_Pilot20_Final_v1\masks --auto-masks experiments\session6_2_analysis_roi_candidates_v1\masks_full_resolution --stride 4 --output experiments\session6_4_gt_auto_sensitivity_v2

Outputs (NEW directory only):
  per_image_GT_vs_Auto_20.csv
  paired_D0_changes_GT_vs_Auto_15.csv
  by_day_GT_vs_Auto.csv
  diagnosis_summary.json

Verified against 20 actual reviewed GT masks and 20 corresponding same-content
D0-prefixed or normally named test auto-mask files: all 20 matched and the
same-mask control yielded ROI Dice=1.0000. This is a functional test, not a
measurement of the real Session 6.2 automatic ROI quality.

Note: comparison with a model trained on all 20 GT images is in-sample ROI
sensitivity only, NOT independent model evaluation or freshness validation.
