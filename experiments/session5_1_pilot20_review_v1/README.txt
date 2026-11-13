Session 5.1 Pilot-20 candidate annotation review

SOURCE: User-provided D0, D3, D5, D7 images and V2 auto-generated masks.
STATUS: All masks are unverified CANDIDATES, not Ground Truth.

20 photos: 5 source heads x 4 timepoints, front side, L01 only.
Keep original photo ZIP and previous V2 package unchanged.

Subfolders:
 images/ - 20 original JPEGs (EXIF retained, files not edited)
 candidate_masks/ - 20 V2 auto-generated binary masks (1868x4000)
 candidate_json/ - 20 editable Labelme JSONs
 Pilot20_C01_review.jpg ... Pilot20_C05_review.jpg - 5 full overview images with GREEN candidate areas
 review_tracking.csv - 20 image-specific QA records, all PENDING

To inspect/edit in Labelme:
1) COPY (not MOVE) candidate_json/*.json into images/ while preserving matching base filenames.
2) Open image folder with Labelme; existing candidate polygons should load.
3) Check shadow is excluded; visible green and white tissue included; smooth boundary.
4) Correct polygons manually where needed, save JSON, mark review status.
5) NEVER replace independent Ground Truth using test-model predictions.
6) Only reviewer-accepted masks should be used for quantitative model evaluation.

Scientific limit: 20 images are repeated measures of FIVE leaves from FIVE heads.
No inference of statistical independence across timepoints.
