"""Evaluate a saved baseline checkpoint and export prediction previews.

This file is NEW and does not change earlier source files or checkpoints.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader

from models.unet import LightweightUNet
from train_smoke_test import PairedZipDataset


def safe_checkpoint(path):
    """Load only locally generated trusted weights and metadata."""
    return torch.load(path, map_location='cpu', weights_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', required=True)
    parser.add_argument('--masks', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', default='experiments/session3_4_evaluation')
    parser.add_argument('--split', choices=['val', 'train'], default='val')
    parser.add_argument('--preview-count', type=int, default=6)
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists() and any(destination.iterdir()):
        parser.error(f'Output folder already contains files: {destination}. Choose a NEW --output folder.')
    destination.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    ckpt = safe_checkpoint(args.checkpoint)
    settings = ckpt['config']
    model = LightweightUNet(base_channels=settings['base_channels'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    dataset = PairedZipDataset(args.images, args.masks, args.split, settings['size'])
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    records, examples = [], []
    tp_sum = fp_sum = fn_sum = 0
    with torch.no_grad():
        for images, masks, names in loader:
            logits = model(images)
            prediction = torch.sigmoid(logits) >= 0.5
            truth = masks >= 0.5
            tp = int((prediction & truth).sum().item())
            fp = int((prediction & ~truth).sum().item())
            fn = int((~prediction & truth).sum().item())
            tp_sum += tp; fp_sum += fp; fn_sum += fn
            dice = (2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7)
            iou = (tp + 1e-7) / (tp + fp + fn + 1e-7)
            records.append({'image': names[0], 'dice': dice, 'iou': iou})
            if len(examples) < args.preview_count:
                rgb = (images[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                gt = (truth[0, 0].numpy().astype(np.uint8) * 255)
                pred = (prediction[0, 0].numpy().astype(np.uint8) * 255)
                examples.append((names[0], rgb, gt, pred))
    with (destination / 'per_image_metrics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['image', 'dice', 'iou'])
        writer.writeheader(); writer.writerows(records)
    global_dice = (2 * tp_sum + 1e-7) / (2 * tp_sum + fp_sum + fn_sum + 1e-7)
    global_iou = (tp_sum + 1e-7) / (tp_sum + fp_sum + fn_sum + 1e-7)
    (destination / 'summary.txt').write_text(
        f'Split: {args.split}\nImages: {len(records)}\nGlobal Dice: {global_dice:.6f}\nGlobal IoU: {global_iou:.6f}\n'
        'Threshold: 0.5\nNote: Source-domain evaluation only; not Huang Jing Bai Cai validation.\n',
        encoding='utf-8')
    if examples:
        size = examples[0][1].shape[0]
        sheet = Image.new('RGB', (size * 3, (size + 24) * len(examples)), '#ffffff')
        draw = ImageDraw.Draw(sheet)
        for idx, (name, rgb, gt, pred) in enumerate(examples):
            y = idx * (size + 24)
            for j, arr in enumerate([rgb, gt, pred]):
                sheet.paste(Image.fromarray(arr).convert('RGB'), (j * size, y + 24))
            draw.text((4, y + 4), name + ' | RGB / GT / Pred', fill='#111111')
        sheet.save(destination / 'predictions_preview.png')
    print(f'{args.split}: {len(records)} images | global Dice={global_dice:.4f} | IoU={global_iou:.4f}')
    print(f'PASS: Evaluation saved to {destination}')


if __name__ == '__main__':
    main()
