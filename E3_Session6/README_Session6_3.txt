SESSION 6.3: VISUAL FRESHNESS-RELATED FEATURES (PROVISIONAL)
==============================================================

NEW FILE ONLY:
  E3_Session6\extract_visual_freshness_features.py
KEEP ALL existing Session3, Session4, Session5, Session6.1 and Session6.2 files/checkpoints.

Put the Python file in Napa_Cabbage_Project\E3_Session6\.

RUN IN THE PROJECT ROOT (PyCharm Terminal):

python.exe E3_Session6\extract_visual_freshness_features.py --d0 datasets\Huang_Jing_Bai_Cai\Huang_Jing_Bai_Cai_D0.zip --d3 datasets\Huang_Jing_Bai_Cai\Huang_Jing_Bai_Cai_D3.zip --d5 datasets\Huang_Jing_Bai_Cai\Huang_Jing_Bai_Cai_D5.zip --d7 datasets\Huang_Jing_Bai_Cai\Huang_Jing_Bai_Cai_D7.zip --manifest experiments\session6_2_analysis_roi_candidates_v1\candidate_roi_manifest_157.csv --masks experiments\session6_2_analysis_roi_candidates_v1\masks_full_resolution --qc experiments\session6_2_roi_qc_review_v1\ROI_QC_157_review_tracking.csv --stride 4 --output experiments\session6_3_visual_features_provisional_v1

If you have not copied or edited the ROI review CSV yet, OMIT the two options:
  --qc experiments\session6_2_roi_qc_review_v1\ROI_QC_157_review_tracking.csv
The script then marks all review statuses NOT_PROVIDED, never ACCEPTED.

OUTPUT (ALL NEW; refuses to overwrite existing directory):
  visual_features_157_PROVISIONAL.csv
  D0_followup_deltas_111_PROVISIONAL.csv
  descriptive_by_day_side_PROVISIONAL.csv
  analysis_protocol.json

FEATURE DEFINITIONS (DESCRIPTIVE, NOT CLINICALLY VALIDATED):
 - RGB mean values for unchanged original JPEG pixel colours within a candidate leaf ROI.
 - HSV means; OpenCV hue range 0-179, S/V range 0-255.
 - green heuristic: H 35-85, S>=50, V>=40.
 - yellow candidate heuristic: H 15-34, S>=45, V>=55.
 - brown/dark-warm candidate heuristic: H 5-24, S>=50, 35<=V<180.
 - ROI area fraction: foreground sampled pixels / image sampled pixels.
 - texture_neighbor_diff: average absolute neighbouring grayscale pixel difference (sampled grid).
 - stride=4 samples original unmodified RGB pixels every 4 rows/columns.
 - These color bins can overlap, especially the yellow/brown candidate ranges.
 - They cannot measure chlorophyll, true spoilage, shelf-life, or microbial safety.
 - Camera exposure, white balance, temperature/freezing, JPEG, pose and ROI quality may affect values.
 - All photo observations are correlated within just FIVE independent source heads.
 - 111 paired rows are repeated observations and NOT 111 biological replicates.

QUALITY RULE:
Use the provisional CSV to identify trends and prioritize ROI review. Do not make claims of verified
freshness trends until the relevant masks are visually reviewed/corrected. The 20 original pilot GT
may be used for spot checks, but the 157 candidate ROIs are not independently verified.

DATA INTEGRITY:
Original zipped JPEGs never modified; EXIF oriented before metrics. Full-res masks must match the
EXIF-oriented photo size. Source ZIP names and members come from the previously audited manifest.

PYTHON PACKAGES: pip install numpy pillow opencv-python (already used in prior sessions).
