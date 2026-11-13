"""Session 6.4 v2: ROI sensitivity using 20 reviewed GTs and automatic ROI candidates.
Run in the project root. This is a QC comparison, NOT an independent model test.
Uses the exact feature definitions in Session 6.3; does not alter input files.
"""
import argparse
import csv
import json
import io
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from extract_visual_freshness_features import metric_values, load_mask, write_rows

DAYS = (0, 3, 5, 7)
METRICS = ('green_ratio', 'yellow_candidate_ratio', 'brown_candidate_ratio',
           'mean_r', 'mean_g', 'mean_b', 'mean_s', 'mean_v',
           'texture_neighbor_diff', 'roi_fraction')

def discover_member(z, name):
    # ZIP archives may store Chinese member names with variant encodings;
    # choose by last path component after the ZIP decoder resolves its name.
    matches = [p for p in z.namelist() if p.replace('\\', '/').split('/')[-1] == name]
    if len(matches) != 1:
        raise ValueError(f'Expected one JPEG named {name}, found {len(matches)}')
    return matches[0]

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for day in DAYS:
        ap.add_argument(f'--d{day}', type=Path, required=True)
    ap.add_argument('--gt-masks', type=Path, required=True)
    ap.add_argument('--auto-masks', type=Path, required=True)
    ap.add_argument('--stride', type=int, default=4)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if not 1 <= args.stride <= 16:
        ap.error('stride must be 1..16')
    if args.output.exists():
        ap.error(f'Output exists; refusing to overwrite: {args.output}')
    for d in DAYS:
        if not getattr(args, f'd{d}').is_file():
            ap.error(f'Missing input ZIP D{d}')
    if not args.gt_masks.is_dir() or not args.auto_masks.is_dir():
        ap.error('Both GT and automatic mask folders must exist')

    # Avoid creating output until input file paths are validated.
    names = [(d, f'{"" if d == 0 else "D" + str(d)}C{h:02d}L01正.jpg')
             for d in DAYS for h in range(1, 6)]
    # GT filenames preserve original D0 naming (e.g. C01L01正.png).
    # Session 6.2 automatic ROI filenames ALWAYS have a day prefix,
    # including D0 (e.g. D0C01L01正.png). Never rename source files.
    def mask_paths(day, original_jpg):
        gt_path = args.gt_masks / Path(original_jpg).with_suffix('.png')
        auto_stem = original_jpg if day != 0 else f'D0{original_jpg}'
        auto_path = args.auto_masks / Path(auto_stem).with_suffix('.png')
        return gt_path, auto_path

    for day, name in names:
        gt_path, auto_path = mask_paths(day, name)
        for path in (gt_path, auto_path):
            if not path.is_file():
                ap.error(f'Missing mask: {path}')
    args.output.mkdir(parents=True)
    rows = []
    archives = {}
    try:
        archives = {d: zipfile.ZipFile(getattr(args, f'd{d}')) for d in DAYS}
        for d, name in names:
            z = archives[d]
            with Image.open(io.BytesIO(z.read(discover_member(z, name)))) as im:
                rgb = np.asarray(ImageOps.exif_transpose(im).convert('RGB'))
            wh = (rgb.shape[1], rgb.shape[0])
            gt_path, auto_path = mask_paths(d, name)
            gt = load_mask(gt_path, wh)
            auto = load_mask(auto_path, wh)
            a = int((gt & auto).sum()); b = int((auto & ~gt).sum()); c = int((gt & ~auto).sum())
            g_metrics = metric_values(rgb, gt, args.stride)
            p_metrics = metric_values(rgb, auto, args.stride)
            if g_metrics is None or p_metrics is None:
                raise ValueError('Too few sampled ROI pixels: ' + name)
            row = dict(image=name, day=d, source_head=f'C{int(name.split("C")[1][:2]):02d}',
                       gt_pixels=int(gt.sum()), auto_pixels=int(auto.sum()),
                       intersection_pixels=a, false_positive_pixels=b, false_negative_pixels=c,
                       mask_dice=2*a/max(1,2*a+b+c), mask_iou=a/max(1,a+b+c))
            for m in METRICS:
                row[f'gt_{m}'] = float(g_metrics[m])
                row[f'auto_{m}'] = float(p_metrics[m])
                row[f'auto_minus_gt_{m}'] = float(p_metrics[m] - g_metrics[m])
            rows.append(row)
            print(f'Checked {len(rows)}/20: {name}; ROI Dice={row["mask_dice"]:.4f}')
    finally:
        for z in archives.values():
            z.close()

    cols = list(rows[0].keys())
    write_rows(args.output/'per_image_GT_vs_Auto_20.csv', rows, cols)
    lookup = {(r['source_head'], r['day']): r for r in rows}
    pairs=[]
    for h in [f'C{i:02d}' for i in range(1, 6)]:
        base = lookup[(h, 0)]
        for day in (3, 5, 7):
            follow = lookup[(h, day)]
            out = dict(source_head=h, day=day, baseline=base['image'], followup=follow['image'])
            for m in METRICS:
                out[f'GT_delta_{m}']=follow[f'gt_{m}']-base[f'gt_{m}']
                out[f'AUTO_delta_{m}']=follow[f'auto_{m}']-base[f'auto_{m}']
                out[f'delta_sensitivity_{m}']=out[f'AUTO_delta_{m}']-out[f'GT_delta_{m}']
            pairs.append(out)
    write_rows(args.output/'paired_D0_changes_GT_vs_Auto_15.csv',pairs,list(pairs[0]))
    group=[]
    for d in DAYS:
        records=[r for r in rows if r['day']==d]
        out=dict(day=d, n_heads=5,
                 mean_mask_dice=float(np.mean([r['mask_dice'] for r in records])),
                 mean_mask_iou=float(np.mean([r['mask_iou'] for r in records])))
        for m in ('green_ratio','yellow_candidate_ratio','brown_candidate_ratio','roi_fraction'):
            out[f'GT_mean_{m}']=float(np.mean([r[f'gt_{m}'] for r in records]))
            out[f'AUTO_mean_{m}']=float(np.mean([r[f'auto_{m}'] for r in records]))
            out[f'mean_AUTO_minus_GT_{m}']=out[f'AUTO_mean_{m}']-out[f'GT_mean_{m}']
        group.append(out)
    write_rows(args.output/'by_day_GT_vs_Auto.csv',group,list(group[0]))
    summary=dict(status='ROI_SENSITIVITY_DIAGNOSTIC_NOT_FRESHNESS_VALIDATION',
        images=20,heads=5,pairs=15,days=list(DAYS),mask_source='user-reviewed GT versus analysis-model automatic candidate',
        sampling_stride=args.stride,
        mean_mask_dice=float(np.mean([r['mask_dice'] for r in rows])),
        min_mask_dice=float(min(r['mask_dice'] for r in rows)),
        remarks=[
            'The analysis model was trained on all 20 GT images. This GT-versus-auto mask comparison is IN-SAMPLE and cannot validate segmentation generalization.',
            'Compare GT-based D3 colour change with auto-mask-based change. If both show spikes, ROI mismatch alone does not explain them.',
            'Shared changes can still reflect lighting, white balance, storage effects, or heuristic HSV thresholds. No calibrated freshness conclusions.',
            'Only L01 front of 5 heads is evaluated; other 137 images remain candidate ROI unless individually reviewed.'
        ])
    (args.output/'diagnosis_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print('PASS: 20 GT/auto mask comparisons and 15 within-head D0 changes')
    print(f'Mean GT-vs-auto ROI Dice (IN-SAMPLE, NOT external validation): {summary["mean_mask_dice"]:.4f}')
    print('Output:',args.output)

if __name__ == '__main__':
    main()
