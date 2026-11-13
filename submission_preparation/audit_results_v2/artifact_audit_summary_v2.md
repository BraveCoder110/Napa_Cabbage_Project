# CS7002SR AI Model Artifact: structural audit

Run from project root; no inputs were modified.

| Priority | Item | Status | Details |
|---|---|---|---|
| CRITICAL | Custom U-Net architecture | OK | Exists; content/implementation not functionally tested |
| CRITICAL | Custom U-Net source training | OK | Exists; content/implementation not functionally tested |
| CRITICAL | Custom U-Net source evaluation | OK | Exists; content/implementation not functionally tested |
| CRITICAL | Custom U-Net saved checkpoint | OK | Exists; content/implementation not functionally tested |
| CRITICAL | ResNet18 U-Net architecture | OK | Exists; content/implementation not functionally tested |
| CRITICAL | ResNet18 U-Net source training | OK | Exists; content/implementation not functionally tested |
| CRITICAL | ResNet18 U-Net source evaluation | OK | Exists; content/implementation not functionally tested |
| CRITICAL | ResNet18 source checkpoint | OK | Exists; content/implementation not functionally tested |
| CRITICAL | Zero-shot target evaluation | OK | Exists; content/implementation not functionally tested |
| CRITICAL | Leave-one-head-out few-shot script | OK | Exists; content/implementation not functionally tested |
| CRITICAL | 20 reviewed target masks | OK | 20 immediate files, 0 subdirectories |
| CRITICAL | 20 pilot target RGB | OK | 28 immediate files, 0 subdirectories |
| CRITICAL | Target D0 archive | OK | JPG count 37; CRC not tested |
| CRITICAL | Target D3 archive | OK | JPG count 40; CRC not tested |
| CRITICAL | Target D5 archive | OK | JPG count 40; CRC not tested |
| CRITICAL | Target D7 archive | OK | JPG count 40; CRC not tested |
| IMPORTANT | Longitudinal data audit | OK | Exists; content/implementation not functionally tested |
| IMPORTANT | Analysis-only model training | OK | Exists; content/implementation not functionally tested |
| IMPORTANT | 157 candidate mask manifest | OK | 157 data rows; expected 157 |
| IMPORTANT | 157 full resolution candidate masks | OK | 157 immediate files, 0 subdirectories |
| IMPORTANT | Visual feature extraction | OK | Exists; content/implementation not functionally tested |
| IMPORTANT | 157 image features | OK | 157 data rows; expected 157 |
| IMPORTANT | 111 D0 follow-up pairs | OK | 111 data rows; expected 111 |
| IMPORTANT | GT-versus-auto comparison v2 | OK | Exists; content/implementation not functionally tested |
| IMPORTANT | 20 GT/auto comparisons | OK | 20 data rows; expected 20 |
| IMPORTANT | Photo colour diagnostic | OK | Exists; content/implementation not functionally tested |
| SUPPORTING | 20 GT photo-colour diagnostic output | OK | 20 data rows; expected 20 |
| SUPPORTING | Report-ready figures | OK | 13 immediate files, 0 subdirectories |

## Interpretation
This is an inventory, not a reproducibility or scientific validity test. Status OK confirms path/content structure only. Source dataset 575-image binary masks, full source training results, LOHO fold results, and all 157 ROI QA are not fully verified by this manifest. The 20-image analysis mask Dice 0.9576 is in-sample and NOT the five-fold out-of-fold Dice 0.9483.

## User verification tasks
- Official public Bok Choy dataset URL, author, usage licence and redistribution permission
- Required source dataset/mask availability and clear access instructions
- Functional rerun evidence for source val, target zero-shot, and five LOHO folds
- Checkpoint safety/compatibility load verified in the user's own trusted environment
- Whether model checkpoints are retained in final submission and within platform upload limit
- Academic integrity / AI tool declaration and final Canvas packaging rules
