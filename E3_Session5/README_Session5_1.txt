E3 Session 5.1 - Prepare an independently labeled target-domain pilot

FILE ACTIONS:
NEW: E3_Session5/prepare_target_pilot.py
KEEP: ALL older scripts, datasets, masks, model checkpoints and results.
No other files need to be copied into E3_Session5.

From Napa_Cabbage_Project ROOT, run:
python.exe E3_Session5\prepare_target_pilot.py --d0 datasets\Huang_Jing_Bai_Cai\Huang_Jing_Bai_Cai_D0.zip --d3 datasets\Huang_Jing_Bai_Cai\Huang_Jing_Bai_Cai_D3.zip --d5 datasets\Huang_Jing_Bai_Cai\Huang_Jing_Bai_Cai_D5.zip --d7 datasets\Huang_Jing_Bai_Cai\Huang_Jing_Bai_Cai_D7.zip --output experiments\session5_1_target_annotation_v1

Change the ZIP paths to their *real* local paths if stored elsewhere. This command
assumes copies exist in datasets\Huang_Jing_Bai_Cai, and does not relocate them.

This script exports 20 unchanged JPEGs: C01..C05, L01, front, days 0/3/5/7.
This is a pilot benchmark for subsequent zero-shot testing (not a final
statistically independent test set). Source filenames, ZIP members, and image
hashes are captured in annotation_manifest.csv.

NEXT MANUAL ACTION:
1. Install a polygon annotation application (e.g. labelme) if necessary.
2. Open each image, draw one or more polygons covering the visible numbered
   detached leaf (excluding tabletop/background), label class "leaf".
3. Save JSON annotations under a NEW directory (do not overwrite the JPEGs).
4. Review uncertain boundaries and log reviewer decisions.
5. Do not run pretrained U-Net or SAM predictions as unquestioned GT.

IMPORTANT: This phase is only dataset preparation. Dice/IoU cannot be reported
on the target domain until independent masks and their review are complete.
