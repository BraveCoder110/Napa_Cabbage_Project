"""Session 6.4.2: leaf vs same-photo wood-reference colour diagnostics.

Uses 20 manually reviewed L01-front GT masks. Wood is NOT a colour card.
No colour correction and no verified freshness labels are produced.
NEW outputs only; refuses to overwrite an existing folder.
"""
import argparse
import csv
import io
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

DAYS = [0, 3, 5, 7]
HSV_BINS = {
    'yellow_base': (15, 35, 45, 55),
    'yellow_narrow': (18, 32, 65, 75),
    'yellow_broad': (12, 38, 35, 45),
}


def write_csv(path, rows, keys):
    with open(path, 'w', encoding='utf-8-sig', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def image_data(z, member):
    with Image.open(io.BytesIO(z.read(member))) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert('RGB'))


def load_gt(z, stem, expected_size):
    names = [n for n in z.namelist() if n.endswith('/masks/' + stem + '.png')]
    if len(names) != 1:
        raise ValueError(f'Expected exactly one GT PNG for {stem}, found {len(names)}')
    with Image.open(io.BytesIO(z.read(names[0]))) as im:
        if im.size != expected_size:
            raise ValueError(f'GT orientation/size mismatch for {stem}: {im.size} vs {expected_size}')
        return np.asarray(im.convert('L')) > 127


def mean_or_nan(a):
    return float(np.mean(a)) if a.size else float('nan')


def stats(rgb_pixels, key):
    pixels = rgb_pixels.astype(np.float32)
    means = pixels.mean(axis=0)
    color_sum = pixels.sum(axis=1).clip(min=1)
    fractions = (pixels / color_sum[:, None]).mean(axis=0)
    lab = cv2.cvtColor(rgb_pixels.reshape(1, -1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    lab_mean = lab.astype(np.float32).mean(axis=0)
    return {f'{key}_mean_{c}': float(means[i]) for i, c in enumerate('rgb')} | {
        f'{key}_chroma_{c}': float(fractions[i]) for i, c in enumerate('rgb')
    } | {f'{key}_lab_{c}': float(lab_mean[i]) for i, c in enumerate('Lab')}


def extract(rgb, gt, stride):
    # Sampling after EXIF alignment retains original RGB pixel values.
    rgb = rgb[::stride, ::stride]
    fg = gt[::stride, ::stride]
    h, w = fg.shape
    # Corner rectangles are fixed in IMAGE coordinates, not chosen to create a trend.
    # Reject a corner if it overlaps the GT after 3% image-height/width expansion.
    pad = max(3, round(min(h, w) * 0.03))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*pad+1, 2*pad+1))
    avoid = cv2.dilate(fg.astype(np.uint8), kernel) > 0
    patch_h = max(4, round(h * 0.12))
    patch_w = max(4, round(w * 0.12))
    margins_h, margins_w = round(h*.02), round(w*.02)
    patches = {
        'TL': (margins_h, margins_h+patch_h, margins_w, margins_w+patch_w),
        'TR': (margins_h, margins_h+patch_h, w-margins_w-patch_w, w-margins_w),
        'BL': (h-margins_h-patch_h, h-margins_h, margins_w, margins_w+patch_w),
        'BR': (h-margins_h-patch_h, h-margins_h, w-margins_w-patch_w, w-margins_w),
    }
    wood_pieces = []
    accepted = []
    rejected = []
    for name, (y0, y1, x0, x1) in patches.items():
        valid = ~avoid[y0:y1, x0:x1]
        fraction_valid = float(valid.mean())
        if fraction_valid < .95:
            rejected.append(name)
            continue
        wood_pieces.append(rgb[y0:y1, x0:x1][valid])
        accepted.append(name)
    if int(fg.sum()) < 1000:
        raise ValueError('GT foreground too small after downsampling')
    leaf_rgb = rgb[fg]
    out = {'leaf_pixel_count_sampled': int(fg.sum()), 'wood_patches_accepted': '|'.join(accepted),
           'wood_patches_rejected': '|'.join(rejected), 'wood_patch_count': len(accepted),
           'wood_reference_status': 'AVAILABLE' if len(accepted) >= 2 else 'INSUFFICIENT'}
    out.update(stats(leaf_rgb, 'leaf'))
    hsv = cv2.cvtColor(leaf_rgb.reshape(1,-1,3), cv2.COLOR_RGB2HSV).reshape(-1,3)
    H, S, V = hsv[:,0], hsv[:,1], hsv[:,2]
    for label, (lo,hi,smin,vmin) in HSV_BINS.items():
        out[label + '_ratio'] = float(((H>=lo)&(H<hi)&(S>=smin)&(V>=vmin)).mean())
    if wood_pieces:
        wood_rgb = np.concatenate(wood_pieces)
        out.update(stats(wood_rgb, 'wood'))
        out['wood_pixel_count_sampled'] = len(wood_rgb)
    else:
        out['wood_pixel_count_sampled'] = 0
        for key in ('mean_r','mean_g','mean_b','chroma_r','chroma_g','chroma_b','lab_L','lab_a','lab_b'):
            out['wood_'+key] = float('nan')
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for d in DAYS:
        p.add_argument(f'--d{d}', type=Path, required=True)
    p.add_argument('--gt-zip', type=Path, required=True, help='E3_Session5_1_Pilot20_GT_Final_v1.zip; NOT the 157 auto masks')
    p.add_argument('--stride', type=int, default=4)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.stride < 1:
        p.error('--stride must be >=1')
    if args.output.exists():
        p.error('Output already exists: choose a NEW directory, never overwrite.')
    args.output.mkdir(parents=True)
    rows = []
    with zipfile.ZipFile(args.gt_zip) as zg:
        for d in DAYS:
            with zipfile.ZipFile(getattr(args, f'd{d}')) as zi:
                for head in range(1,6):
                    head_id = f'C{head:02d}'
                    stem = ('' if d == 0 else f'D{d}') + head_id + 'L01正'
                    items = [n for n in zi.namelist() if n.endswith('/'+stem+'.jpg')]
                    if len(items) != 1:
                        raise ValueError(f'Expected one RGB image {stem}.jpg, found {len(items)}')
                    rgb = image_data(zi, items[0])
                    gt = load_gt(zg, stem, (rgb.shape[1],rgb.shape[0]))
                    measures = extract(rgb,gt,args.stride)
                    rows.append({'head':head_id, 'day':d, 'stem':stem, 'image':stem+'.jpg', **measures})
                    print(f'{stem}: wood patches={measures["wood_patch_count"]}, yellow(base)={measures["yellow_base_ratio"]:.4f}')
    assert len(rows) == 20
    metric_keys = ['yellow_base_ratio','yellow_narrow_ratio','yellow_broad_ratio',
                   'leaf_mean_r','leaf_mean_g','leaf_mean_b','leaf_chroma_r','leaf_chroma_g','leaf_chroma_b',
                   'wood_mean_r','wood_mean_g','wood_mean_b','wood_chroma_r','wood_chroma_g','wood_chroma_b',
                   'wood_lab_L','wood_lab_a','wood_lab_b']
    baseline = {r['head']:r for r in rows if r['day']==0}
    pairs=[]
    for r in rows:
        if r['day']==0: continue
        b=baseline[r['head']]
        entry={'head':r['head'],'day':r['day'],'wood_reference_status':r['wood_reference_status'],
               'D0_wood_reference_status':b['wood_reference_status']}
        for k in metric_keys:
            entry['delta_'+k] = r[k]-b[k]
        pairs.append(entry)
    assert len(pairs)==15
    summary=[]
    for d in DAYS:
        ss=[r for r in rows if r['day']==d]
        obj={'day':d,'image_count':len(ss),'wood_reference_available':sum(r['wood_reference_status']=='AVAILABLE' for r in ss)}
        for k in metric_keys:
            vals=np.array([r[k] for r in ss],dtype=np.float64)
            obj['mean_'+k]=float(np.nanmean(vals)) if np.isfinite(vals).any() else float('nan')
        summary.append(obj)
    # Five-head paired directional check, not a significance test.
    consistency=[]
    for d in (3,5,7):
        ss=[r for r in pairs if r['day']==d]
        consistency.append({'day':d,'n_heads':len(ss),**{
            'heads_positive_delta_'+k:sum(r['delta_'+k]>0 for r in ss)
            for k in ('yellow_base_ratio','yellow_narrow_ratio','yellow_broad_ratio', 'wood_chroma_b')
        }})
    write_csv(args.output/'pilot20_leaf_wood_color.csv', rows, list(rows[0]))
    write_csv(args.output/'paired_D0_changes_15.csv', pairs, list(pairs[0]))
    write_csv(args.output/'by_day_5heads.csv', summary, list(summary[0]))
    write_csv(args.output/'threshold_consistency_5heads.csv', consistency,list(consistency[0]))
    protocol={'n_images':20,'n_heads':5,'pairs':15,'stride':args.stride,'main_roi':'manually reviewed GT on L01 front only',
              'bins_HSV_OpenCV':HSV_BINS,'reference':'Fixed four corner patches, 12% each dimension, 2% inset; leaf-dilated exclusions; require >=2 patches for usable reference',
              'limitations':['Wood is NOT a standard colour card','Patch selection does not guarantee absence of shadow','HSV cutoffs are heuristic, not biological freshness labels','Camera exposure and white balance cannot be resolved from JPEG alone','Only five source heads and L01 front','Any colour correction must be separately justified and validated'],
              'interpretation':'Diagnostic, not calibrated colour or verified freshness'}
    (args.output/'protocol_and_limitations.json').write_text(json.dumps(protocol,indent=2,ensure_ascii=False),encoding='utf-8')
    print('PASS: 20 GT-based leaf/wood diagnoses, 15 D0 pairs; five heads and three thresholds')
    print('Wood reference availability:', sum(r['wood_reference_status']=='AVAILABLE' for r in rows),'/20')
    print('Output:',args.output)

if __name__=='__main__':
    main()
