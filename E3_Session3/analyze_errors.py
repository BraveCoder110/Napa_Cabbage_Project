"""Export worst-case source-domain segmentation errors without modifying existing artifacts."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from models.unet import LightweightUNet
from train_smoke_test import PairedZipDataset


def score(tp, fp, fn):
    return (2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7), (tp + 1e-7) / (tp + fp + fn + 1e-7)


def safe_directory(path, inputs):
    out = path.resolve()
    for p in inputs:
        inp = p.resolve()
        if out == inp or out in inp.parents or inp in out.parents:
            raise ValueError(f'Output directory conflicts with an input path: {inp}')
    if path.exists():
        raise FileExistsError(f'Output directory already exists: {path}. Choose a NEW output directory.')
    path.mkdir(parents=True, exist_ok=False)


def to_rgb(image):
    return np.clip(image.permute(1, 2, 0).cpu().numpy() * 255, 0, 255).astype(np.uint8)


def make_panel(rgb, truth, pred, name, dice, iou):
    h, w = truth.shape
    fp = pred & ~truth
    fn = truth & ~pred
    gt = (truth.astype(np.uint8) * 255)
    predicted = (pred.astype(np.uint8) * 255)
    overlay = rgb.copy()
    overlay[fp] = [255, 40, 40]   # False positive: red
    overlay[fn] = [35, 185, 255]  # False negative: cyan
    error = np.zeros_like(rgb)
    error[fp] = [255, 40, 40]
    error[fn] = [35, 185, 255]
    arrays = [rgb, np.repeat(gt[..., None], 3, axis=2), np.repeat(predicted[..., None], 3, axis=2), overlay, error]
    header = 43
    panel = Image.new('RGB', (w * len(arrays), h + header), 'white')
    for j, arr in enumerate(arrays):
        panel.paste(Image.fromarray(arr), (j * w, header))
    draw = ImageDraw.Draw(panel)
    draw.text((5, 4), f'{name} | Dice={dice:.4f} IoU={iou:.4f}', fill='black')
    for j, label in enumerate(['RGB', 'GT', 'Pred', 'Error overlay', 'Error only']):
        draw.text((j * w + 5, 23), label, fill='black')
    return panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--masks', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--metrics', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if not 0 < args.threshold < 1 or args.top_k < 1:
        parser.error('Require 0 < threshold < 1 and top-k >= 1')
    for p in [args.images, args.masks, args.checkpoint, args.metrics]:
        if not p.is_file():
            parser.error(f'Input file does not exist: {p}')
    if args.output.exists():
        parser.error(f'Output exists, refusing overwrite: {args.output}. Choose a NEW path.')
    torch.set_num_threads(max(1, args.threads))
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    model = LightweightUNet(base_channels=config['base_channels'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    dataset = PairedZipDataset(str(args.images), str(args.masks), 'val', config['size'])

    with args.metrics.open(newline='', encoding='utf-8-sig') as handle:
        old = list(csv.DictReader(handle))
    if not old or any(not {'image', 'dice', 'iou'} <= set(row) for row in old):
        parser.error('Metrics CSV must contain image, dice, iou columns')
    old_by_id = {r['image']: r for r in old}
    if len(old_by_id) != len(old):
        parser.error('Duplicate image names in metrics CSV')
    results = []
    samples = {}
    tp_total = fp_total = fn_total = 0
    with torch.inference_mode():
        for i in range(len(dataset)):
            rgb_tensor, gt_tensor, name = dataset[i]
            pred_tensor = torch.sigmoid(model(rgb_tensor.unsqueeze(0)))[0, 0] >= args.threshold
            truth = (gt_tensor[0] >= 0.5).cpu().numpy()
            pred = pred_tensor.cpu().numpy()
            tp = int(np.count_nonzero(pred & truth))
            fp = int(np.count_nonzero(pred & ~truth))
            fn = int(np.count_nonzero(~pred & truth))
            tn = int(np.count_nonzero(~pred & ~truth))
            dice, iou = score(tp, fp, fn)
            results.append(dict(image=name, dice=dice, iou=iou, tp=tp, fp=fp, fn=fn, tn=tn,
                                gt_pixels=tp+fn, pred_pixels=tp+fp, foreground_error_fraction=(fp+fn)/max(1, tp+fp+fn)))
            samples[name] = (to_rgb(rgb_tensor), truth, pred)
            tp_total += tp; fp_total += fp; fn_total += fn
    if set(old_by_id) != {r['image'] for r in results}:
        raise RuntimeError('The image identifiers in metrics CSV do not match checkpoint evaluation')
    if args.threshold == 0.5:
        for row in results:
            previous = old_by_id[row['image']]
            if abs(float(previous['dice']) - row['dice']) > 1e-5 or abs(float(previous['iou']) - row['iou']) > 1e-5:
                raise RuntimeError(f'Metric mismatch at {row["image"]}. Verify checkpoint and input data before continuing.')
    worst = sorted(results, key=lambda r: (r['dice'], r['image']))[:args.top_k]
    safe_directory(args.output, [args.images, args.masks, args.checkpoint, args.metrics])
    detail = args.output / 'worst_cases'
    detail.mkdir()
    panels = []
    for rank, r in enumerate(worst, 1):
        rgb, gt, pred = samples[r['image']]
        panel = make_panel(rgb, gt, pred, r['image'], r['dice'], r['iou'])
        panel.save(detail / f'{rank:02d}_{r["image"]}.png')
        panels.append(panel)
    if panels:
        overview = Image.new('RGB', (panels[0].width, sum(p.height for p in panels)), 'white')
        y = 0
        for panel in panels:
            overview.paste(panel, (0, y))
            y += panel.height
        overview.save(args.output / 'worst10_overview.png')
    fields = list(results[0])
    for name, rows in [('all_val_detailed_metrics.csv', results), ('worst10_metrics.csv', worst)]:
        with (args.output / name).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    global_dice, global_iou = score(tp_total, fp_total, fn_total)
    summary = dict(images=len(results), threshold=args.threshold, global_dice=global_dice, global_iou=global_iou,
                   mean_image_dice=float(np.mean([r['dice'] for r in results])),
                   mean_image_iou=float(np.mean([r['iou'] for r in results])),
                   images_below_dice_090=sum(r['dice'] < 0.9 for r in results),
                   worst_image=worst[0]['image'], worst_dice=worst[0]['dice'])
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'Analyzed: {len(results)} validation images | global Dice={global_dice:.4f} | global IoU={global_iou:.4f}')
    print(f'Mean per-image Dice={summary["mean_image_dice"]:.4f} | IoU={summary["mean_image_iou"]:.4f}')
    print(f'Worst image: {worst[0]["image"]} | Dice={worst[0]["dice"]:.4f}; below 0.90: {summary["images_below_dice_090"]}')
    print(f'PASS: Read-only error analysis saved to {args.output}')


if __name__ == '__main__':
    main()
