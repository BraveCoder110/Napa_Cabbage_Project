"""Session 5.2.1: diagnose frozen-source to target-domain segmentation transfer.

This script never modifies model checkpoints, original images, masks or old outputs.
All thresholds other than 0.5 are diagnostic only, NOT independently tuned test results.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch


def metric(tp, fp, fn):
    return (2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.0,
            tp/(tp+fp+fn) if tp+fp+fn else 1.0)


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def get_models(project, custom_ckpt, transfer_ckpt):
    # Reuse the exact architecture and checkpoint loading from Session 5.2.
    sys.path.insert(0, str(project / 'E3_Session5'))
    from evaluate_zero_shot import get_models as model_classes, load_model
    custom_cls, transfer_cls = model_classes(project)
    return {
        'custom': load_model('custom', custom_ckpt, custom_cls, transfer_cls),
        'resnet18': load_model('resnet18', transfer_ckpt, custom_cls, transfer_cls)
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images', required=True, type=Path, help='Pilot 20 target JPG directory')
    p.add_argument('--masks', required=True, type=Path, help='Pilot 20 reviewed binary GT directory')
    p.add_argument('--custom-checkpoint', required=True, type=Path)
    p.add_argument('--transfer-checkpoint', required=True, type=Path)
    p.add_argument('--source-images', type=Path, help='Optional original source image ZIP')
    p.add_argument('--source-masks', type=Path, help='Optional original source masks ZIP')
    p.add_argument('--source-max-images', type=int, default=0, help='0 means all 115 val samples; >0 is a nonrepresentative smoke subset')
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    if args.output.exists():
        p.error('Output directory already exists; choose a NEW versioned output path.')
    for path in (args.images, args.masks):
        if not path.is_dir():
            p.error(f'Missing folder: {path}')
    for path in (args.custom_checkpoint, args.transfer_checkpoint):
        if not path.is_file():
            p.error(f'Missing checkpoint: {path}')
    if (args.source_images is None) != (args.source_masks is None):
        p.error('Provide both --source-images and --source-masks, or neither.')
    if args.source_max_images < 0:
        p.error('Invalid --source-max-images')
    if args.source_images:
        for path in (args.source_images, args.source_masks):
            if not path.is_file():
                p.error(f'Missing source ZIP: {path}')
    torch.set_num_threads(min(4, torch.get_num_threads()))
    project = Path(__file__).resolve().parent.parent
    from evaluate_zero_shot import find_pairs
    pairs = find_pairs(args.images, args.masks)
    models = get_models(project, args.custom_checkpoint, args.transfer_checkpoint)

    # No directory is created until all prerequisite paths/checkpoints are checked.
    args.output.mkdir(parents=True, exist_ok=False)
    scores = []
    probability_rows = []
    threshold_rows = []
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    count_by_threshold = defaultdict(lambda: [0,0,0])
    gt_count = 0
    n_pixels = 0
    with torch.inference_mode():
        for stem, img_path, mask_path, day, head in pairs:
            with Image.open(img_path) as f:
                im = ImageOps.exif_transpose(f).convert('RGB')
                full_size = im.size
                small = im.resize((128,128), Image.Resampling.BILINEAR)
                image = np.asarray(small, dtype=np.float32).transpose(2,0,1).copy()/255.0
                x = torch.from_numpy(image).unsqueeze(0)
            with Image.open(mask_path) as f:
                if f.size != full_size:
                    raise ValueError(f'RGB/GT orientation mismatch for {stem}: {full_size} vs {f.size}')
                native = np.asarray(f.convert('L'))
                if not np.isin(native, [0,255]).all():
                    raise ValueError(f'Non-binary mask: {stem}')
                truth = np.asarray(f.convert('L').resize((128,128), Image.Resampling.NEAREST)) > 127
            fg = int(truth.sum()); n = truth.size
            gt_count += fg; n_pixels += n
            # A no-learning all-foreground baseline must be reported for context.
            d,i = metric(fg,n-fg,0)
            scores.append(dict(image=stem, day=day, head=head, model='all_foreground',
                               dice=d, iou=i, tp=fg, fp=n-fg, fn=0, pred_fraction=1.0))
            for model_name, model in models.items():
                prob = torch.sigmoid(model(x))[0,0].detach().cpu().numpy()
                pred = prob >= 0.5
                tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum()); fn = int((~pred & truth).sum())
                d,i = metric(tp,fp,fn)
                scores.append(dict(image=stem, day=day, head=head, model=model_name,
                                   dice=d, iou=i, tp=tp, fp=fp, fn=fn,
                                   pred_fraction=float(pred.mean())))
                # Explicit background probability is diagnostic evidence for FP.
                probability_rows.append(dict(image=stem, day=day, head=head, model=model_name,
                    fg_mean=float(prob[truth].mean()) if fg else None,
                    bg_mean=float(prob[~truth].mean()) if fg < n else None,
                    fg_p10=float(np.quantile(prob[truth], 0.1)) if fg else None,
                    bg_p50=float(np.median(prob[~truth])) if fg < n else None,
                    bg_p90=float(np.quantile(prob[~truth], 0.9)) if fg < n else None,
                    bg_above_05_fraction=float((prob[~truth] >= 0.5).mean()) if fg < n else None))
                for threshold in thresholds:
                    pp = prob >= threshold
                    t = int((pp & truth).sum()); f = int((pp & ~truth).sum()); miss = int((~pp & truth).sum())
                    counts = count_by_threshold[(model_name,threshold)]
                    counts[0] += t; counts[1] += f; counts[2] += miss
    # Reproduce original zero-shot per-image counts as a consistency assertion.
    old = defaultdict(lambda:[0,0,0])
    for r in scores:
        c = old[r['model']]; c[0] += r['tp']; c[1] += r['fp']; c[2] += r['fn']
    baseline_summary = {}
    for name,c in sorted(old.items()):
        d,i = metric(*c)
        baseline_summary[name]={'global_dice':d,'global_iou':i,'tp':c[0],'fp':c[1],'fn':c[2],
                                'foreground_fraction':(c[0]+c[1])/n_pixels}
    for (name,threshold),c in sorted(count_by_threshold.items()):
        d,i = metric(*c)
        threshold_rows.append(dict(model=name, threshold=threshold, dice_diagnostic=d,
                                   iou_diagnostic=i, tp=c[0], fp=c[1], fn=c[2],
                                   predicted_foreground_fraction=(c[0]+c[1])/n_pixels))
    source_summary = {'status':'not_run','reason':'Source ZIPs not provided'}
    if args.source_images:
        sys.path.insert(0,str(project/'E3_Session3'))
        from train_smoke_test import PairedZipDataset
        ds = PairedZipDataset(str(args.source_images),str(args.source_masks),'val',128)
        if len(ds) != 115:
            raise ValueError(f'Expected 115 source validation images; got {len(ds)}')
        max_n = min(len(ds),args.source_max_images) if args.source_max_images else len(ds)
        aggregate=defaultdict(lambda:[0,0,0]); subset=[]
        with torch.inference_mode():
            for k in range(max_n):
                rgb, gt, stem = ds[k]
                x = rgb.unsqueeze(0)
                truth = gt[0].numpy() >= 0.5
                for name,model in models.items():
                    prob = torch.sigmoid(model(x))[0,0].cpu().numpy()
                    pred = prob >= 0.5
                    t = int((pred & truth).sum()); f=int((pred & ~truth).sum()); miss=int((~pred & truth).sum())
                    c=aggregate[name];c[0]+=t;c[1]+=f;c[2]+=miss
                    d,i=metric(t,f,miss)
                    subset.append(dict(image=stem,model=name,dice=d,iou=i,tp=t,fp=f,fn=miss))
        write_csv(args.output/'source_per_image_diagnostic.csv',subset)
        source_summary={'status':'full_validation' if max_n==115 else 'subset_only',
                        'images_evaluated':max_n,'expected_full_validation':115,
                        'results':{name:dict(zip(['global_dice','global_iou'],metric(*c))) for name,c in aggregate.items()}}

    write_csv(args.output/'target_per_image_diagnostic.csv',scores)
    write_csv(args.output/'target_probability_diagnostic.csv',probability_rows)
    write_csv(args.output/'threshold_sweep_DIAGNOSTIC_ONLY.csv',threshold_rows)
    report={
      'warning':'Threshold sweep is post-hoc on the target pilot and must not be reported as independent test improvement.',
      'reference_unchanged_target_global_dice':{'custom':0.57088603,'resnet18':0.58675959},
      'preprocessing':'EXIF-aligned RGB, resize bilinear 128x128, RGB [0,1], GT nearest 128x128, threshold 0.5',
      'n_target_images':len(pairs),'gt_foreground_fraction':gt_count/n_pixels,
      'target':baseline_summary,'source_reproduction':source_summary}
    for name, expected in report['reference_unchanged_target_global_dice'].items():
        if abs(report['target'][name]['global_dice']-expected)>0.0001:
            raise AssertionError(f'{name} differs from Session 5.2 target result; investigate preprocessing or files')
    (args.output/'diagnosis_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PASS: target diagnostics matched Session 5.2 frozen checkpoint results')
    for name,row in baseline_summary.items():
        print(f"{name}: target Dice={row['global_dice']:.4f} IoU={row['global_iou']:.4f}")
    if args.source_images:
        print(f"Source reproduction: {source_summary['status']} on {source_summary['images_evaluated']} validation images")
        for name,row in source_summary['results'].items():
            print(f"  {name}: source Dice={row['global_dice']:.4f} IoU={row['global_iou']:.4f}")
    print('Threshold sweep is DIAGNOSTIC ONLY; do not claim it as independent test performance.')
    print('Output:', args.output)


if __name__=='__main__':
    main()
