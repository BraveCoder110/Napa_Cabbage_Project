# Evidence tiers and report controls

STRONGEST PILOT EVIDENCE
- 115-image source validation reproduced.
- 20-image target zero-shot baseline and results are direct frozen-model tests.
- 20 out-of-fold predictions in five head-held-out folds; no train/test overlap of source heads within a fold.

TRAINING-SET SENSITIVITY, NOT INDEPENDENT TEST
- Session 6.2 analysis model used all 20 manual GT examples.
- 20-image GT-vs-auto mean Dice 0.9576 is in-sample.

EXPLORATORY PHOTOGRAPHIC OBSERVATION
- D3 yellow candidate peak persists across three HSV settings and the GT-vs-auto ROI choice, but wood reference also changes.
- The ROI occupies a smaller share of the frame on D7 than D0; cannot call this physical area loss.
- 157-image ROI colour features are provisional: 157 QC statuses NOT_PROVIDED.

NO CLAIMS
- No validated fresh/rotten labels, no food safety prediction, no independent external test, no 157 independent biological replicates, no transformer experiment completed.
- No single model deployability validation on the 137 other images; masks are low-resolution 128x128 estimates resized to native dimensions.
- No causal isolation of augmentation or pretraining.
