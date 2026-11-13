E3 Session 6.1 - longitudinal image inventory (NEW FILES ONLY)

NEW code file:
  E3_Session6/audit_longitudinal.py
Do NOT overwrite prior scripts or checkpoints.

Required original user ZIP archives (example suggested placement):
  datasets/Huang_Jing_Bai_Cai/Huang_Jing_Bai_Cai_D0.zip
  datasets/Huang_Jing_Bai_Cai/Huang_Jing_Bai_Cai_D3.zip
  datasets/Huang_Jing_Bai_Cai/Huang_Jing_Bai_Cai_D5.zip
  datasets/Huang_Jing_Bai_Cai/Huang_Jing_Bai_Cai_D7.zip

Run from the Napa_Cabbage_Project root using one line:
python.exe E3_Session6\audit_longitudinal.py --d0 datasets\Huang_Jing_Bai_Cai\Huang_Jing_Bai_Cai_D0.zip --d3 datasets\Huang_Jing_Bai_Cai\Huang_Jing_Bai_Cai_D3.zip --d5 datasets\Huang_Jing_Bai_Cai\Huang_Jing_Bai_Cai_D5.zip --d7 datasets\Huang_Jing_Bai_Cai\Huang_Jing_Bai_Cai_D7.zip --output experiments\session6_1_longitudinal_audit_v1

If ZIP locations differ, modify only --d0/--d3/--d5/--d7 paths.
Output directory must not exist. Script will refuse to overwrite.
No external Python libraries required, and original JPEG bytes are NOT rewritten.

Expected output:
  observation_manifest_157.csv (all 157 original observations, SHA256)
  expected_grid_160.csv (all theoretical 5*4*2*4 positions)
  missing_observations.csv (3 missing D0 reverse photos)
  longitudinal_series_40.csv (20 leaves * 2 sides)
  D0_followup_pairs_111.csv (37 complete side-series * 3 followup days)
  audit_summary.json
  README_RESEARCH_BOUNDARIES.txt

The complete audit outputs are ALSO provided as a separately downloadable ZIP,
so rerunning locally is for reproducibility, not required to obtain results.

Next: Session 6.2 ROI quality strategy and visual freshness-related indicator extraction.
Do NOT treat unreviewed automatic masks as human ground truth.
