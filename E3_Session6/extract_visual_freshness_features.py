"""Session 6.3: provisional image-based visual freshness indicators.

Uses original RGB JPEG pixels (EXIF aligned) under candidate/resolved ROI masks.
No freshness class is inferred; no automatic ROI is considered verified ground truth.
"""
import argparse
import csv
import io
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

DAYS = (0, 3, 5, 7)
MISSING = {'C04L03反.jpg', 'C05L03反.jpg', 'C05L04反.jpg'}

def read_rows(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write_rows(path, rows, fields):
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

def image_from_zip(z, member):
    with Image.open(io.BytesIO(z.read(member))) as obj:
        im = ImageOps.exif_transpose(obj).convert('RGB')
        return np.asarray(im)

def load_mask(mask_path, rgb_size):
    if not mask_path.exists():
        raise FileNotFoundError(f'Missing candidate ROI: {mask_path}')
    with Image.open(mask_path) as m:
        if m.size != rgb_size:
            raise ValueError(f'Mask size {m.size} != EXIF-oriented RGB size {rgb_size}: {mask_path}')
        return np.asarray(m.convert('L')) > 127

def metric_values(rgb, mask, stride):
    # Pixel-stride sampling uses unchanged RGB values; no photometric augmentation.
    rgb = rgb[::stride, ::stride]
    mask = mask[::stride, ::stride]
    n = int(mask.sum())
    if n < 300:
        return None
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    selected_rgb = rgb[mask].astype(np.float32)
    selected_hsv = hsv[mask]
    h, s, v = [selected_hsv[:, i].astype(np.int16) for i in range(3)]
    # These bins are explicitly heuristic appearance proxies, NOT verified
    # chlorophyll, yellowing, necrosis, disease, or biological freshness labels.
    green = (h >= 35) & (h <= 85) & (s >= 50) & (v >= 40)
    yellow = (h >= 15) & (h < 35) & (s >= 45) & (v >= 55)
    brown = (h >= 5) & (h < 25) & (s >= 50) & (v >= 35) & (v < 180)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # Neighbor difference texture is an appearance feature, not calibrated texture.
    h_valid = mask[:, :-1] & mask[:, 1:]
    v_valid = mask[:-1, :] & mask[1:, :]
    parts = []
    if h_valid.any():
        parts.append(np.abs(gray[:, :-1] - gray[:, 1:])[h_valid])
    if v_valid.any():
        parts.append(np.abs(gray[:-1, :] - gray[1:, :])[v_valid])
    contrast = float(np.concatenate(parts).mean()) if parts else float('nan')
    return dict(roi_pixel_count_sampled=n,
                roi_fraction=float(mask.mean()),
                mean_r=float(selected_rgb[:, 0].mean()),
                mean_g=float(selected_rgb[:, 1].mean()),
                mean_b=float(selected_rgb[:, 2].mean()),
                mean_h=float(h.mean()),
                mean_s=float(s.mean()),
                mean_v=float(v.mean()),
                green_ratio=float(green.mean()),
                yellow_candidate_ratio=float(yellow.mean()),
                brown_candidate_ratio=float(brown.mean()),
                texture_neighbor_diff=contrast)

def main():
    ap = argparse.ArgumentParser()
    for d in DAYS:
        ap.add_argument(f'--d{d}', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--masks', type=Path, required=True, help='Session 6.2 masks_full_resolution folder')
    ap.add_argument('--qc', type=Path, help='Optional completed ROI_QC_157_review_tracking.csv')
    ap.add_argument('--stride', type=int, default=4, help='Fixed native-pixel sampling step, default 4')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if not 1 <= args.stride <= 16:
        ap.error('--stride must be between 1 and 16')
    if args.output.exists():
        ap.error(f'Output already exists; refusing overwrite: {args.output}')
    manifest = read_rows(args.manifest)
    if len(manifest) != 157 or len({r['image'] for r in manifest}) != 157:
        ap.error('Expected exactly 157 unique observations in Session 6.2 manifest')
    qc = {r['image']: r for r in read_rows(args.qc)} if args.qc else {}
    paths = {d: getattr(args, f'd{d}') for d in DAYS}
    args.output.mkdir(parents=True)
    features = []
    try:
        archives = {d: zipfile.ZipFile(paths[d]) for d in DAYS}
        for i, r in enumerate(sorted(manifest, key=lambda x: (int(x['day']), x['image'])), 1):
            day = int(r['day'])
            z = archives[day]
            image = image_from_zip(z, r['source_member'])
            fullmask = load_mask(args.masks / Path(r['image']).with_suffix('.png'),
                                 (image.shape[1], image.shape[0]))
            metrics = metric_values(image, fullmask, args.stride)
            if metrics is None:
                raise ValueError(f'ROI too small to compute features: {r["image"]}')
            row = {k: r[k] for k in ['image', 'day', 'source_head', 'leaf_id', 'side']}
            row['series_id'] = r['source_head'] + r['leaf_id'] + r['side']
            row['mask_provenance'] = 'SESSION6_2_FINAL_MODEL_AUTO_CANDIDATE'
            row['human_review_status'] = qc.get(r['image'], {}).get('human_review_status', 'NOT_PROVIDED').strip().upper() or 'PENDING'
            row.update(metrics)
            features.append(row)
            if i % 40 == 0 or i == 157:
                print(f'Extracted {i}/157 feature rows')
    finally:
        for z in locals().get('archives', {}).values():
            z.close()
    numerical = ['roi_fraction','mean_r','mean_g','mean_b','mean_h','mean_s','mean_v',
                 'green_ratio','yellow_candidate_ratio','brown_candidate_ratio','texture_neighbor_diff']
    fields = ['image','day','source_head','leaf_id','side','series_id','mask_provenance',
              'human_review_status','roi_pixel_count_sampled'] + numerical
    write_rows(args.output/'visual_features_157_PROVISIONAL.csv', features, fields)
    lookup = {(r['series_id'], int(r['day'])): r for r in features}
    pairs = []
    for r in features:
        day = int(r['day'])
        if day == 0:
            continue
        baseline = lookup.get((r['series_id'], 0))
        if not baseline:
            continue
        p = {k:r[k] for k in ['series_id','source_head','leaf_id','side']}
        p.update(day=day, baseline_image=baseline['image'], followup_image=r['image'],
                 baseline_roi_qc=baseline['human_review_status'],followup_roi_qc=r['human_review_status'])
        p['both_roi_accepted'] = int(baseline['human_review_status']=='ACCEPTED' and r['human_review_status']=='ACCEPTED')
        for metric in numerical:
            p[f'delta_{metric}'] = float(r[metric] - baseline[metric])
        pairs.append(p)
    if len(pairs) != 111:
        raise ValueError(f'Expected 111 D0/follow-up matched pairs; got {len(pairs)}')
    pair_fields = ['series_id','source_head','leaf_id','side','day','baseline_image','followup_image',
                   'baseline_roi_qc','followup_roi_qc','both_roi_accepted'] + [f'delta_{k}' for k in numerical]
    write_rows(args.output/'D0_followup_deltas_111_PROVISIONAL.csv', pairs, pair_fields)
    # Summaries are descriptive only; repeated images and heads are not independent.
    grouped = defaultdict(list)
    for r in features:
        grouped[(int(r['day']), r['side'])].append(r)
    desc = []
    for (day, side), rows in sorted(grouped.items()):
        rec = dict(day=day, side=side, image_count=len(rows),
                   accepted_count=sum(r['human_review_status']=='ACCEPTED' for r in rows))
        for metric in ['green_ratio','yellow_candidate_ratio','brown_candidate_ratio','texture_neighbor_diff','roi_fraction']:
            rec[f'mean_{metric}'] = float(np.mean([r[metric] for r in rows]))
        desc.append(rec)
    write_rows(args.output/'descriptive_by_day_side_PROVISIONAL.csv',desc,
               ['day','side','image_count','accepted_count']+[f'mean_{m}' for m in ['green_ratio','yellow_candidate_ratio','brown_candidate_ratio','texture_neighbor_diff','roi_fraction']])
    # Repeated leaf-side comparison is descriptive; no p-values on pseudoreplicates.
    qc_counts = dict(Counter(r['human_review_status'] for r in features))
    summary = dict(status='PROVISIONAL_UNTIL_ROI_QC',observations=157,matched_pairs=111,
                   complete_four_day_series=sum(all((sid,d) in lookup for d in DAYS) for sid in set(r['series_id'] for r in features)),
                   independent_source_heads=5, sampling_stride=args.stride,
                   roi_review_status_counts=qc_counts,
                   note='Heuristic visible colour bins, not validated freshness classes. No statistical independence of repeated observations. Not calibrated for colour or projected physical area.',
                   baseline_missing=list(sorted(MISSING)))
    with open(args.output/'analysis_protocol.json','w',encoding='utf-8') as f:
        json.dump(summary,f,indent=2,ensure_ascii=False)
    print('PASS: 157 RGB/ROI feature rows; 111 matched D0 comparisons; 37 complete series')
    print('QC status counts:',qc_counts)
    print('Output:',args.output)
    print('IMPORTANT: Results are PROVISIONAL; visually audited ROI and colour calibration are needed for strong freshness conclusions.')

if __name__=='__main__':
    main()
