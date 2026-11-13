# Technical Report — Results and Discussion (Simple English Draft)

## Research aim
This pilot project investigates deep learning methods that can support image-based visual freshness assessment of cabbage during refrigerated storage. Leaf segmentation is used to isolate visible cabbage tissue before analysing image-based colour and shape indicators. Storage day is an observation time, not a validated freshness label.

## Dataset and protocol
We collected 157 photographs across D0 (37), D3 (40), D5 (40), and D7 (40) from 20 detached leaves belonging to five source cabbage heads. Twenty L01 front-side photographs (five heads by four times) received manually reviewed binary foreground masks. The independent biological source count is five, not 157.

## Segmentation findings (Figure 1; Table 1)
Two models learned source-domain segmentation using 575 public Shanghai Bok Choy images (460 train; 115 validation). Source validation global Dice was 0.9576 for Custom U-Net and 0.9484 for ResNet18 U-Net. However, zero-shot target Dice fell to 0.5709 and 0.5868. Models falsely predicted much of the wooden table as foreground. A no-learning all-foreground predictor scored 0.5692 Dice. The original source scores were reproduced, reducing concern about a gross checkpoint-loading problem.

For few-shot adaptation, each of five folds trained on 16 images from four cabbage heads and evaluated four images from the held-out fifth head. Both models were reinitialized from their source checkpoints for each fold and trained for three epochs with training-only augmentation. Out-of-fold global Dice improved to 0.6645 for Custom U-Net and 0.9483 for pretrained ResNet18 U-Net. This suggests that limited target labels and transfer learning substantially improve segmentation in the present pilot protocol. Architecture, parameter count, and initialization differ, so the gain cannot be attributed solely to ImageNet pretraining.

## Visual freshness-related image indicators (Figures 2–5; Tables 2–5)
The pilot's manual masks yielded mean candidate yellow ratios of 2.18%, 7.04%, 1.56%, and 1.89% on D0, D3, D5, and D7. In the same 20 photographs, automatic ROI masks produced 2.48%, 8.12%, 3.03%, and 3.44%. Thus the D3 peak persisted under both ROI definitions, but later small changes depended on the mask. All three examined HSV threshold sets also showed an elevated D3 group mean.

However, the photographic wooden-background reference changed in RGB composition on D3, including a lower blue-channel share. This suggests possible differences in illumination or camera white balance. The wood is not a standard reference colour chart, and these data cannot separate photographic colour changes from real biological yellowing.

On the 20 manually reviewed images, the fraction of the photograph occupied by cabbage foreground decreased from 45.43% on D0 to 34.93% on D7. This is a two-dimensional image fraction, not calibrated physical leaf area or a measured shrinkage percentage. Camera geometry, leaf folding and ROI differences are plausible contributors.

## Limitations and conclusion
The pilot has only five independent source heads, limited manually reviewed labels (20 images), and no independent sensory, chemical, or microbiological freshness ground truth. The final analysis model was fitted using all 20 pilot masks: its agreement with those same masks (mean Dice 0.9576) is in-sample, NOT external validation; the held-out-head segmentation evidence comes from the earlier five-fold procedure. All 157 automated ROI results remain provisional without full manual QC.

The strongest finding is that source-trained models can fail severely on a new imaging background, while small-sample head-grouped adaptation markedly improves segmentation. The colour and projected-area observations are promising exploratory visual indicators, but do not establish true freshness classes or food safety.
