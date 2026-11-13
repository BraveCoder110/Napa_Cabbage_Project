# V4 Experiment Evidence Audit

PASS: 30 | WARN: 3 | FAIL: 0

This audit recomputes pooled Dice/IoU from stored TP/FP/FN. It does not rerun model prediction.
Source dataset license and full independent run still need separate review.

| Category | Check | Status | Details |
|---|---|---|---|
| zero_shot | custom: 20 unique images / 5 heads / 4 days | PASS | 20 rows |
| zero_shot | custom: global Dice/IoU recalculated | PASS | Dice=0.57088603; IoU=0.39946852 |
| zero_shot | custom: Technical Report reported Dice | PASS | computed 0.57088603 |
| zero_shot | resnet18: 20 unique images / 5 heads / 4 days | PASS | 20 rows |
| zero_shot | resnet18: global Dice/IoU recalculated | PASS | Dice=0.58675959; IoU=0.41518739 |
| zero_shot | resnet18: Technical Report reported Dice | PASS | computed 0.58675959 |
| loho_custom | protocol JSON readable | PASS | epochs=3; lr=0.0001 |
| loho_custom | 20 unique held-out image results | PASS | 20 rows |
| loho_custom | per-image held-out head equals image head | PASS |  |
| loho_custom | pooled TP/FP/FN reproduce ALL metrics | PASS | Dice=0.66453142; IoU=0.49760169 |
| loho_custom | Technical Report Dice consistency | PASS | computed 0.66453142 |
| loho_custom | five folds / 16 train + 4 held-out per fold / no head leakage | PASS |  |
| loho_resnet18 | protocol JSON readable | PASS | epochs=3; lr=0.0001 |
| loho_resnet18 | 20 unique held-out image results | PASS | 20 rows |
| loho_resnet18 | per-image held-out head equals image head | PASS |  |
| loho_resnet18 | pooled TP/FP/FN reproduce ALL metrics | PASS | Dice=0.94829793; IoU=0.90167924 |
| loho_resnet18 | Technical Report Dice consistency | PASS | computed 0.94829793 |
| loho_resnet18 | five folds / 16 train + 4 held-out per fold / no head leakage | PASS |  |
| longitudinal | 157 images / five biological heads | PASS | rows=157 |
| longitudinal | day counts D0=37, D3/5/7=40 | PASS | {'D0': 37, 'D3': 40, 'D5': 40, 'D7': 40} |
| longitudinal | 40 leaf-side series; 37 complete | PASS | rows=40, complete=37 |
| longitudinal | 111 D0 follow-up pairs | PASS | rows=111 |
| longitudinal | three specific D0 reverse photos absent | PASS | {('C04', 'L03', 'reverse', 'D0'), ('C05', 'L03', 'reverse', 'D0'), ('C05', 'L04', 'reverse', 'D0')} |
| roi_sensitivity | 20 in-sample GT/auto comparisons | PASS | rows=20 |
| roi_sensitivity | mean in-sample mask Dice approximately 0.9576 | PASS | mean=0.957622956 |
| roi_sensitivity | diagnosis explicitly not freshness validation | PASS | ROI_SENSITIVITY_DIAGNOSTIC_NOT_FRESHNESS_VALIDATION |
| visual_features | 157 provisional features | PASS | rows=157 |
| colour_diagnostic | 20 samples in colour diagnostic | PASS | rows=20 |
| source_validation | custom validation evaluation file has 115 images | PASS | rows=115 |
| source_validation | custom source metric reproduction | WARN | No TP/FP/FN in per-image table; needs original evaluation run for full verification |
| source_validation | resnet18 validation evaluation file has 115 images | PASS | rows=115 |
| source_validation | resnet18 source metric reproduction | WARN | No TP/FP/FN in per-image table; needs original evaluation run for full verification |
| dataset_licence | source license review | WARN | Manual review of attribution and redistribution is required |
