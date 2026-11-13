Session 5.1 Pilot20 GT v1

Input: user-reviewed Labelme polygons (20 JSON) paired with original RGB photos.
Masks: binary 8-bit PNG, 255=leaf, 0=background, size 1868x4000 (EXIF orientation corrected).
Conversion: PIL ImageDraw.polygon fill for Labelme-style polygon semantics. Self-crossing polygons are rasterized without modifying user JSON.
Five contact sheets show the actual mask overlay with correctly oriented images.
Important: 20 files were manually reviewed by the user; machine QC alone does not independently prove every pixel is correct.
No image-derived freshness label. All photos relate to 5 source heads; samples are not biologically independent.
Keep all prior code, checkpoint and V2 auto-generated candidates unchanged.
