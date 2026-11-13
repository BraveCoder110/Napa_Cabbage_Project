"""Run a small real-data training smoke test on CPU.

This script tests optimizer updates and file I/O. It does not establish
model accuracy, convergence, or out-of-domain generalization.
"""
import argparse
import csv
import io
import random
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from models.unet import LightweightUNet


class PairedZipDataset(Dataset):
    """Load corresponding RGB JPEG and binary PNG masks directly from ZIPs."""

    def __init__(self, image_zip: str, mask_zip: str, split: str, size: int = 128):
        self.image_zip = image_zip
        self.mask_zip = mask_zip
        self.split = split
        self.size = size
        with zipfile.ZipFile(mask_zip) as archive:
            mask_paths = sorted(n for n in archive.namelist()
                                if n.startswith(f"{split}/masks/") and n.endswith(".png"))
        with zipfile.ZipFile(image_zip) as archive:
            all_images = {Path(n).stem: n for n in archive.namelist()
                          if n.lower().endswith((".jpg", ".jpeg", ".png"))}
        self.items = []
        for mask_path in mask_paths:
            stem = Path(mask_path).stem
            if stem not in all_images:
                raise FileNotFoundError(f"Missing input image for {mask_path}")
            self.items.append((all_images[stem], mask_path, stem))
        if not self.items:
            raise ValueError(f"No image-mask pairs found for split={split}")
        self._images = None
        self._masks = None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        # Create ZIP readers lazily to avoid sharing open handles across workers.
        if self._images is None:
            self._images = zipfile.ZipFile(self.image_zip)
            self._masks = zipfile.ZipFile(self.mask_zip)
        image_path, mask_path, stem = self.items[index]
        with Image.open(io.BytesIO(self._images.read(image_path))) as im:
            image = im.convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR)
            image_array = np.asarray(image, dtype=np.float32).copy() / 255.0
        with Image.open(io.BytesIO(self._masks.read(mask_path))) as im:
            mask = im.convert("L").resize((self.size, self.size), Image.Resampling.NEAREST)
            mask_array = (np.asarray(mask, dtype=np.uint8) > 127).astype(np.float32)
        x = torch.from_numpy(image_array.transpose(2, 0, 1))
        y = torch.from_numpy(mask_array[None, ...].copy())
        return x, y, stem


def soft_dice_loss(logits, targets, epsilon=1.0):
    """Compute differentiable Dice loss on foreground probabilities."""
    probs = torch.sigmoid(logits)
    axes = (1, 2, 3)
    overlap = (probs * targets).sum(dim=axes)
    denominator = probs.sum(dim=axes) + targets.sum(dim=axes)
    return (1.0 - (2.0 * overlap + epsilon) / (denominator + epsilon)).mean()


def combined_loss(logits, targets):
    """Combine pixel-wise BCE with region-overlap Dice loss."""
    return nn.functional.binary_cross_entropy_with_logits(logits, targets) + soft_dice_loss(logits, targets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--output", default="experiments/session3_3")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.size < 16 or args.size % 8 != 0:
        parser.error("steps and batch-size must be positive; size must be >=16 and divisible by 8")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    dataset = PairedZipDataset(args.images, args.masks, "train", args.size)
    print(f"Training pairs: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = LightweightUNet(base_channels=args.base_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    history = []
    iterator = iter(loader)
    for step in range(1, args.steps + 1):
        try:
            images, masks, names = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, masks, names = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = combined_loss(logits, masks)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"Non-finite gradient norm at step {step}")
        before = model.head.weight.detach().clone()
        optimizer.step()
        update = (model.head.weight.detach() - before).abs().max().item()
        if update <= 0:
            raise RuntimeError("Model parameters did not update")
        history.append({"step": step, "loss": round(loss.item(), 6), "max_head_update": update})
        print(f"Step {step}/{args.steps} | loss={loss.item():.4f} | max head update={update:.7f} | {list(names)}", flush=True)

    # The checkpoint is a smoke-test artifact, not a final trained model.
    checkpoint = output / "unet_smoke_test.pth"
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "config": vars(args), "steps_completed": args.steps, "purpose": "smoke_test_only"}, checkpoint)
    with (output / "training_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "loss", "max_head_update"])
        writer.writeheader()
        writer.writerows(history)
    print(f"Saved checkpoint: {checkpoint}")
    print(f"Saved log: {output / 'training_log.csv'}")
    print("PASS: Real-image training steps, gradients, optimizer updates and checkpoint saving.")
    print("NOTE: Four steps are not enough to claim convergence or segmentation accuracy.")


if __name__ == "__main__":
    main()
