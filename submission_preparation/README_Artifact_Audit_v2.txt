CS7002SR / AS1 — AI Model Artifact structural audit, v2 (NEW ONLY)

WHAT THIS IS
A read-only, standard-library Python script to inspect source code, checkpoints,
original dataset archives, manually reviewed masks, target prediction artefacts,
and some expected experiment outputs. It creates a NEW audit folder and refuses
to overwrite an existing folder.

WHAT THIS IS NOT
It does NOT train or evaluate a model, load checkpoint tensors, prove that a
checkpoint matches the report, check dataset reuse permission, or certify that
157 masks are correct. No script can establish those facts from filenames alone.

SETUP
Copy submission_preparation/audit_model_artifact_v2.py into the same relative
location inside your existing Napa_Cabbage_Project. KEEP all original files.

RUN (Windows, from Napa_Cabbage_Project root)
python.exe submission_preparation\audit_model_artifact_v2.py --output submission_preparation\audit_results_v2

OPTIONAL SHA256 / ZIP CRC CHECKS (can read large files; slower)
python.exe submission_preparation\audit_model_artifact_v2.py --output submission_preparation\audit_results_with_hashes_v2 --hash-files --check-zip-crc

OUTPUTS
artifact_inventory_v2.csv
artifact_audit_summary_v2.json
artifact_audit_summary_v2.md

WHAT TO SEND BACK
Zip only the newly created audit_results_v2 folder and upload it here.
Do not upload large pretrained checkpoints until we establish the correct
final submission bundle structure.

INTERPRETING RESULTS
OK = present with expected shape/count where checked; NOT scientifically validated.
CHECK = present but unexpected count or other manual verification needed.
MISSING = not found at recorded relative path (could be located elsewhere).
INVALID = unreadable or invalid path/file.

All output files are newly written. DO NOT put output on top of existing folders.

PLANNED FINAL SUBMISSION COMPONENTS
1. Source code: Custom U-Net/ResNet18 models; source validation; zero-shot;
   LOHO few-shot; ROI/feature extraction; diagnostic scripts.
2. Checkpoints: source Custom & ResNet18; analysis-only ResNet18 and, if
   retained, each LOHO fold's outputs/checkpoints.
3. Datasets: source dataset access and licence; original 157 image ZIPs;
   reviewed pilot20 masks; public source masks or reproducible mask generation.
4. Evidence: fold-level and overall metrics, logs, selected figures,
   protocol/seed, missing-image manifest, limitations.
5. Reproduction README and dependency/environment versions.
6. Separate approved Technical Report and LSEP Essay (not edited here).

DO NOT claim freshness classification or Fresh/Rotten labels. Dates are time points.
