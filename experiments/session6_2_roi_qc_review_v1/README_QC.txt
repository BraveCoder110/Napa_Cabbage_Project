Session 6.2 ROI QC REVIEW (not GT validation)

Review all four contact sheets. Prioritize HIGH cases, then MEDIUM, then STANDARD.
Flag true segmentation errors in review_tracking.csv: human_review_status = ACCEPTED / NEEDS_FIX / UNCERTAIN.
Keep shadow_background_FP and leaf_petiole_FN as YES/NO/UNCERTAIN.
Original masks, checkpoints and previous experiment results remain unchanged.
Automatic border contact is not necessarily an error.
Check leaf petiole tips, shadow patches, edge leaks, especially reverse sides and leaves L02-L04.
To correct an ROI, use the original EXIF-transposed RGB and the full-resolution mask; visual contact sheets alone do not support exact corrections.
For colour measurements, use original RGB (no colour augmentation), not tinted preview images.
The full-res masks are upsampled from 128x128 and have limited boundary precision.
Do not report 157 masks as manually verified or independently validated.
