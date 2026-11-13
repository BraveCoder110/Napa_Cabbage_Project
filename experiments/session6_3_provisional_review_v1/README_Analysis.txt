SESSION 6.3 — PROVISIONAL REAL-DATA ANALYSIS (2026-11-03)
Source: user-run session6_3_visual_features_provisional_v1.zip
Observed: 157 images; 111 day0-to-followup pairs; 37 complete day series; five source heads.
All 157 ROI QC statuses = NOT_PROVIDED. No color calibration. Neither yellow nor brown candidate is a validated freshness label.

Paired means across 37 comparable image-side series per follow-up day (D0 baseline):
                              D3          D5          D7
Green ratio Δ (pp)         -2.31       +0.56       +0.20
Yellow candidate Δ (pp)    +9.47       -1.13       -0.03
Brown candidate Δ (pp)     +1.98       -0.23       +1.58
ROI image fraction Δ (pp)  -0.91       -5.78       -8.07

D3 yellow-candidate spike is NOT monotone. In 34/37 paired series, D3 yellow increased relative to D0. Mean increase averaged by source head was positive for all five source heads. A substantial reversion by D5 means do NOT interpret as proven biological yellowing or freshness recovery. Review D3 lighting, white balance, camera effects, possible freezing and threshold sensitivity.

ROI image fraction decline may reflect positioning, leaf deformation, crop boundaries and mask errors, NOT calibrated physical leaf area. ROI masks are automatic candidates, not verified GT for 157 photos. Stride=4 pixel sampling was used, so color features are approximate sample-based estimates.

Independent biological sample count = 5 source heads. These are descriptive preliminary values, not inferential evidence or predictive freshness accuracy. 37 side-series are repeated observations from only 20 detached leaves.

Next: check ROI QC and D3 color consistency. Prioritize five heads' D0/D3/D5 matched front-side L01 scenes plus a small reverse/other-leaf sample; record ACCEPTED/NEEDS_FIX/UNCERTAIN and reasons. Re-run with --qc and a NEW output directory once reviewed; do not overwrite previous outputs.
