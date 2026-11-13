"""Convert AnyLabeling polygon annotations into binary semantic segmentation masks.

The original dataset archive is read-only and is never modified.
"""
import argparse
import csv
import io
import json
from collections import Counter
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image


ROOT = 'Bok_Choy_Dataset/'


def parse_args():
    parser = argparse.ArgumentParser(description='Create binary masks from polygon annotations.')
    parser.add_argument('--zip', required=True, type=Path, help='Path to original dataset ZIP')
    parser.add_argument('--output', default=Path('E3_output'), type=Path)
    parser.add_argument('--preview-count', default=12, type=int)
    return parser.parse_args()


def split_for_id(identifier):
    """Keep the dataset publisher's original fixed train/validation split."""
    index = int(identifier.split('_')[-1])
    if 1 <= index <= 460:
        return 'train'
    if 461 <= index <= 575:
        return 'val'
    raise ValueError(f'Unexpected image identifier: {identifier}')


def create_mask(annotation, width, height):
    """Union only polygon shapes; ignore boxes and direction lines."""
    mask = np.zeros((height, width), dtype=np.uint8)
    polygon_count = 0
    invalid_count = 0
    for shape in annotation.get('shapes', []):
        if shape.get('shape_type') != 'polygon':
            continue
        points = np.asarray(shape.get('points', []), dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
            invalid_count += 1
            continue
        # AnyLabeling files in this ZIP contain pixel coordinates, not [0,1] coordinates.
        if (points[:, 0].min() < -1 or points[:, 0].max() > width + 1 or
                points[:, 1].min() < -1 or points[:, 1].max() > height + 1):
            invalid_count += 1
            continue
        vertices = np.rint(points).astype(np.int32)
        vertices[:, 0] = np.clip(vertices[:, 0], 0, width - 1)
        vertices[:, 1] = np.clip(vertices[:, 1], 0, height - 1)
        if cv2.contourArea(vertices) < 1:
            invalid_count += 1
            continue
        cv2.fillPoly(mask, [vertices], 255)
        polygon_count += 1
    return mask, polygon_count, invalid_count


def png_bytes(array):
    output = io.BytesIO()
    Image.fromarray(array).save(output, format='PNG')
    return output.getvalue()


def make_preview(rgb, mask):
    """Create input, ground-truth mask, and semi-transparent overlay panels."""
    target = (320, 240)
    original = cv2.resize(rgb, target, interpolation=cv2.INTER_AREA)
    binary = cv2.resize(mask, target, interpolation=cv2.INTER_NEAREST)
    color = np.zeros_like(original)
    color[:, :, 1] = 255
    overlay = original.copy()
    foreground = binary > 0
    overlay[foreground] = (0.55 * original[foreground] + 0.45 * color[foreground]).astype(np.uint8)
    return np.concatenate([original, np.repeat(binary[..., None], 3, axis=2), overlay], axis=1)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    previews = []
    issues = []
    # Store only masks and reports in the output archive; users keep the original images ZIP.
    mask_zip_path = args.output / 'binary_masks.zip'
    with zipfile.ZipFile(args.zip) as source, zipfile.ZipFile(mask_zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as masks:
        for number in range(1, 576):
            identifier = f'hik_{number:04d}'
            image_name = f'{ROOT}images/{identifier}.jpg'
            label_name = f'{ROOT}labels/{identifier}.json'
            try:
                rgb = np.asarray(Image.open(io.BytesIO(source.read(image_name))).convert('RGB'))
                annotation = json.loads(source.read(label_name))
                height, width = rgb.shape[:2]
                mask, count, invalid = create_mask(annotation, width, height)
                if count == 0 or np.count_nonzero(mask) == 0:
                    issues.append(f'{identifier}: no valid foreground polygon')
                if invalid:
                    issues.append(f'{identifier}: {invalid} invalid polygons')
                split = split_for_id(identifier)
                masks.writestr(f'{split}/masks/{identifier}.png', png_bytes(mask))
                rows.append([identifier, split, width, height, count, invalid,
                             round(float(np.count_nonzero(mask) / mask.size), 6)])
                if len(previews) < args.preview_count:
                    previews.append(make_preview(rgb, mask))
            except (KeyError, ValueError, OSError, json.JSONDecodeError) as error:
                issues.append(f'{identifier}: {type(error).__name__}: {error}')
    with (args.output / 'mask_quality_report.csv').open('w', newline='', encoding='utf-8') as output:
        writer = csv.writer(output)
        writer.writerow(['image_id', 'split', 'width', 'height', 'valid_polygons',
                         'invalid_polygons', 'foreground_ratio'])
        writer.writerows(rows)
    (args.output / 'issues.txt').write_text('\n'.join(issues) if issues else 'No issues detected.\n', encoding='utf-8')
    if previews:
        # Reduce memory usage by stacking only a small number of downscaled examples.
        Image.fromarray(np.concatenate(previews, axis=0)).save(args.output / 'mask_previews.jpg', quality=88)
    print(f'Converted {len(rows)} images. Split counts: {dict(Counter(row[1] for row in rows))}')
    print(f'Total valid polygons: {sum(row[4] for row in rows)}')
    print(f'Images with reported issues: {len(issues)}')
    print(f'Outputs saved in: {args.output.resolve()}')


if __name__ == '__main__':
    main()
