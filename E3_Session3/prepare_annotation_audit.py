"""Create a read-only, prediction-blind audit package for the worst segmentation cases."""
import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw

FIELDS = ["image", "original_dice", "original_iou", "reviewer", "review_date", "visible_unannotated_object", "gt_includes_non_object", "ambiguous_boundary", "audit_decision", "region_description", "evidence_note"]
ALLOWED_EXT = {'.jpg', '.jpeg', '.png'}


def zip_map(archive, extensions):
    result = {}
    for name in archive.namelist():
        path = Path(name)
        if path.suffix.lower() not in extensions:
            continue
        key = path.stem.lower()
        if key in result:
            raise ValueError(f"Duplicate image stem in ZIP: {key}")
        result[key] = name
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', required=True, type=Path)
    parser.add_argument('--masks', required=True, type=Path)
    parser.add_argument('--metrics', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--top-k', type=int, default=10)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error('--top-k must be positive')
    if args.output.exists():
        parser.error(f'Output already exists; refusing to overwrite: {args.output}')
    with args.metrics.open('r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows or not {'image','dice','iou'}.issubset(rows[0]):
        parser.error('Metrics CSV must have image,dice,iou columns')
    rows = sorted(rows, key=lambda r: float(r['dice']))[:args.top_k]
    with zipfile.ZipFile(args.images) as images_zip, zipfile.ZipFile(args.masks) as masks_zip:
        images = zip_map(images_zip, {'.jpg', '.jpeg', '.png'})
        masks = zip_map(masks_zip, {'.png'})
        for r in rows:
            id_ = r['image'].lower()
            if id_ not in images or id_ not in masks:
                raise FileNotFoundError(f'Missing RGB or mask for {id_}')
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / 'cases').mkdir()
        audit_rows, provenance = [], []
        thumb_w, thumb_h = 480, 300
        overview = Image.new('RGB', (thumb_w * 2, thumb_h * len(rows)), 'white')
        draw = ImageDraw.Draw(overview)
        for i, r in enumerate(rows):
            id_ = r['image'].lower()
            rgb_bytes = images_zip.read(images[id_]); mask_bytes = masks_zip.read(masks[id_])
            rgb = Image.open(io.BytesIO(rgb_bytes)).convert('RGB')
            mask = Image.open(io.BytesIO(mask_bytes)).convert('L')
            folder = args.output / 'cases' / id_
            folder.mkdir()
            # Preserve original RGB bytes, including original resolution and metadata.
            (folder / ('original_rgb' + Path(images[id_]).suffix.lower())).write_bytes(rgb_bytes)
            (folder / 'source_gt_mask.png').write_bytes(mask_bytes)
            # The audit preview may resize, but never changes the source assets.
            for j, (name, im) in enumerate((('RGB', rgb), ('GT mask', mask))):
                panel = ImageOps.contain(im, (thumb_w - 20, thumb_h - 40))
                x = j * thumb_w + (thumb_w - panel.width)//2
                y = i * thumb_h + 30 + (thumb_h - 40 - panel.height)//2
                overview.paste(panel, (x, y))
                draw.text((j*thumb_w + 8, i*thumb_h + 8), f'{id_} | {name}', fill='black')
            audit_rows.append({'image': id_, 'original_dice': r['dice'], 'original_iou': r['iou'],
                               'reviewer': '', 'review_date': '', 'visible_unannotated_object': '',
                               'gt_includes_non_object': '', 'ambiguous_boundary': '',
                               'audit_decision': '', 'region_description': '', 'evidence_note': ''})
            provenance.append({'image': id_, 'rgb_archive_path': images[id_], 'gt_archive_path': masks[id_],
                               'rgb_sha256': hashlib.sha256(rgb_bytes).hexdigest(),
                               'mask_sha256': hashlib.sha256(mask_bytes).hexdigest(),
                               'original_rgb_size': list(rgb.size), 'source_mask_size': list(mask.size)})
        with (args.output / 'annotation_audit.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader(); writer.writerows(audit_rows)
        (args.output / 'source_provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
        overview.save(args.output / 'blind_rgb_gt_overview.png')
    print(f'PASS: Exported {len(rows)} source RGB images and original GT masks')
    print(f'Blind audit: {args.output / "blind_rgb_gt_overview.png"}')
    print(f'Review sheet: {args.output / "annotation_audit.csv"}')
    print('Original ZIP files and model checkpoints were not modified.')

if __name__ == '__main__':
    main()
