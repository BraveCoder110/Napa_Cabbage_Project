E3 SESSION 2 - ANYLABELING POLYGON TO BINARY MASK

1) Place Bok_Choy_Dataset.zip in the same folder as convert_polygons.py. download dataset Bok_Choy_Dataset.zip for this directory
2) Install dependencies:
   python -m pip install numpy pillow opencv-python
3) Run:
   python convert_polygons.py --zip Bok_Choy_Dataset.zip --output E3_output
4) Review:
   E3_output/mask_previews.jpg
   E3_output/mask_quality_report.csv
   E3_output/issues.txt
   E3_output/binary_masks.zip

Masks: 0 = background, 255 = any labeled bok choy plant.
Dataset original fixed split: hik_0001..0460 = train; hik_0461..0575 = val.
WARNING: Binary masks are segmentation targets, not freshness labels.
WARNING: This source dataset describes Shanghai Bok Choy, not Huang Jing Bai Cai.
WARNING: A mask is a union of instances and does not retain individual instance IDs.
WARNING: Inspect overlays before training. No scientific claims follow from conversion alone.
