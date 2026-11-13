"""Train a reproducible CPU-friendly U-Net segmentation baseline.

This file is NEW. It imports existing, previously verified code unchanged.
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.unet import LightweightUNet
from train_smoke_test import PairedZipDataset, combined_loss


def scores_from_counts(tp, fp, fn, eps=1e-7):
    """Compute foreground Dice and IoU from pooled confusion counts."""
    return (2 * tp + eps) / (2 * tp + fp + fn + eps), (tp + eps) / (tp + fp + fn + eps)


def run_epoch(model, loader, optimizer=None):
    """Train or evaluate one epoch; aggregate segmentation metrics globally."""
    training = optimizer is not None
    model.train(training)
    total_loss, total_samples = 0.0, 0
    tp = fp = fn = 0
    with torch.set_grad_enabled(training):
        for images, masks, _ in loader:
            logits = model(images)
            loss = combined_loss(logits, masks)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss encountered')
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            batch_size = images.size(0)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            prediction = torch.sigmoid(logits) >= 0.5
            truth = masks >= 0.5
            tp += int((prediction & truth).sum().item())
            fp += int((prediction & ~truth).sum().item())
            fn += int((~prediction & truth).sum().item())
    dice, iou = scores_from_counts(tp, fp, fn)
    return total_loss / total_samples, dice, iou


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', required=True)
    parser.add_argument('--masks', required=True)
    parser.add_argument('--output', default='experiments/session3_4')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--base-channels', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--max-train-batches', type=int, default=0, help='Diagnostic only: 0 means full dataset')
    parser.add_argument('--max-val-batches', type=int, default=0, help='Diagnostic only: 0 means full dataset')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.size < 16 or args.size % 8 or args.lr <= 0:
        parser.error('Invalid training arguments')
    destination = Path(args.output)
    # Refuse to overwrite any experiment artifact from an earlier run.
    if destination.exists() and any(destination.iterdir()):
        parser.error(f'Output folder already contains files: {destination}. Choose a NEW --output folder.')
    destination.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    train = PairedZipDataset(args.images, args.masks, 'train', args.size)
    val = PairedZipDataset(args.images, args.masks, 'val', args.size)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=generator)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    if args.max_train_batches:
        from itertools import islice
        train_loader = list(islice(train_loader, args.max_train_batches))
    if args.max_val_batches:
        from itertools import islice
        val_loader = list(islice(val_loader, args.max_val_batches))
    model = LightweightUNet(base_channels=args.base_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f'Device: CPU | Train pairs: {len(train)} | Validation pairs: {len(val)}', flush=True)
    if args.max_train_batches or args.max_val_batches:
        print('WARNING: Diagnostic subset mode; these metrics are NOT a full validation.', flush=True)
    (destination / 'config.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    rows = []
    best_dice = -1.0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice, train_iou = run_epoch(model, train_loader, optimizer)
        val_loss, val_dice, val_iou = run_epoch(model, val_loader)
        row = dict(epoch=epoch, train_loss=train_loss, train_dice=train_dice,
                   train_iou=train_iou, val_loss=val_loss, val_dice=val_dice, val_iou=val_iou)
        rows.append(row)
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({'model_state_dict': model.state_dict(), 'config': vars(args),
                        'epoch': epoch, 'best_val_dice': val_dice}, destination / 'best_unet.pth')
        print(f'Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} '
              f'| val_dice={val_dice:.4f} | val_iou={val_iou:.4f}', flush=True)
        with (destination / 'history.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
    (destination / 'runtime.txt').write_text(f'elapsed_seconds={time.perf_counter() - started:.2f}\n'
                                             f'pytorch={torch.__version__}\n', encoding='utf-8')
    print(f'PASS: Saved best checkpoint and history to {destination}', flush=True)


if __name__ == '__main__':
    main()
