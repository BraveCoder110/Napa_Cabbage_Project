"""E3 Session 3.1: validate paired RGB images and binary segmentation masks."""

'''
E3 SESSION 3.1 - CPU DATALOADER CHECK

Place check_dataloader.py, Bok_Choy_Dataset.zip, and the E3_output folder in the same folder.
E3_output/binary_masks.zip must exist from Session 2.

Run:
  python -m pip install torch numpy pillow
  python check_dataloader.py --images Bok_Choy_Dataset.zip --masks E3_output/binary_masks.zip

The script verifies 460 train and 115 val pairs, tensor shapes,
mask labels and alignment in session3_1_pair_preview.png.
No model training happens in this session.
'''
import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

IMAGE_ROOT = "Bok_Choy_Dataset/images/"


class BokChoySegmentationDataset(Dataset):
    """Read the original JPEG and generated PNG mask without extracting either ZIP."""

    def __init__(self, image_zip, mask_zip, split="train", size=128, limit=None):
        if split not in ("train", "val"):
            raise ValueError("Split must be train or val")
        self.image_zip = str(image_zip)
        self.mask_zip = str(mask_zip)
        self.size = size
        indices = range(1, 461) if split == "train" else range(461, 576)
        self.ids = [f"hik_{idx:04d}" for idx in indices]
        if limit is not None:
            self.ids = self.ids[:limit]
        self.split = split
        # Verify all image-mask pairs up front, without decoding full image files.
        with zipfile.ZipFile(self.image_zip) as iz, zipfile.ZipFile(self.mask_zip) as mz:
            image_names, mask_names = set(iz.namelist()), set(mz.namelist())
            missing = [(sid, "image" if IMAGE_ROOT + sid + ".jpg" not in image_names else "mask")
                       for sid in self.ids if (IMAGE_ROOT + sid + ".jpg" not in image_names or
                                              f"{split}/masks/{sid}.png" not in mask_names)]
            if missing:
                raise FileNotFoundError(f"Missing pairs: {missing[:5]}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sid = self.ids[index]
        # Open ZIP archives in each call; this is safe with DataLoader workers=0.
        with zipfile.ZipFile(self.image_zip) as archive:
            image = Image.open(io.BytesIO(archive.read(IMAGE_ROOT + sid + ".jpg"))).convert("RGB")
        with zipfile.ZipFile(self.mask_zip) as archive:
            mask = Image.open(io.BytesIO(archive.read(f"{self.split}/masks/{sid}.png"))).convert("L")
        if image.size != mask.size:
            raise ValueError(f"Size mismatch for {sid}: {image.size} versus {mask.size}")
        # Bilinear interpolation is for RGB images; nearest-neighbor preserves mask labels.
        image = image.resize((self.size, self.size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.size, self.size), Image.Resampling.NEAREST)
        x = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
        y = torch.from_numpy((np.asarray(mask, dtype=np.uint8) > 127).astype(np.float32).copy()).unsqueeze(0)
        return x, y, sid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, default=Path("Bok_Choy_Dataset.zip"))
    parser.add_argument("--masks", type=Path, default=Path("E3_output/binary_masks.zip"))
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    torch.manual_seed(42)
    for split, count in (("train", 460), ("val", 115)):
        dataset = BokChoySegmentationDataset(args.images, args.masks, split, args.size)
        print(f"{split}: {len(dataset)}/{count} image-mask pairs found")
        if len(dataset) != count:
            raise RuntimeError("Unexpected dataset size")
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        x, y, names = next(iter(loader))
        assert x.shape == (min(args.batch_size, count), 3, args.size, args.size)
        assert y.shape == (min(args.batch_size, count), 1, args.size, args.size)
        assert torch.isfinite(x).all() and torch.isfinite(y).all()
        assert set(torch.unique(y).tolist()).issubset({0.0, 1.0})
        print(f"  first batch: {list(names)}; RGB {tuple(x.shape)}; mask {tuple(y.shape)}")
        print(f"  RGB range: [{x.min():.3f}, {x.max():.3f}]; mask values: {torch.unique(y).tolist()}")
        if split == "train":
            original = (x[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            mask = (y[0, 0].numpy() * 255).astype(np.uint8)
            overlay = original.copy()
            overlay[mask > 0] = (0.55 * overlay[mask > 0] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
            panel = np.concatenate((original, np.repeat(mask[..., None], 3, axis=2), overlay), axis=1)
            Image.fromarray(panel).save("session3_1_pair_preview.png")
            print("  Preview saved: session3_1_pair_preview.png")
    print("PASS: data loading and tensor validation completed.")


if __name__ == "__main__":
    main()
