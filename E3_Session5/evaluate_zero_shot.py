"""Session 5.2: frozen-model cross-domain segmentation evaluation.

Only reads existing checkpoints, target RGB and human-reviewed GT.
Never writes into existing outputs. All comments are in English.
"""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch


def get_models(project):
    sys.path.insert(0, str(project / 'E3_Session3'))
    from models.unet import LightweightUNet
    sys.path.insert(0, str(project / 'E3_Session4' / 'models'))
    from resnet18_unet import ResNet18UNet
    return LightweightUNet, ResNet18UNet


def load_model(name, path, lightweight_cls, transfer_cls):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    if 'model_state_dict' not in ck or 'config' not in ck:
        raise ValueError(f'Checkpoint format not recognized: {path}')
    size = int(ck['config']['size'])
    if size != 128:
        raise ValueError(f'Expected original 128px training size, found {size} in {path}')
    if name == 'custom':
        model = lightweight_cls(base_channels=int(ck['config']['base_channels']))
    else:
        if ck.get('architecture') != 'ResNet18UNet' or ck.get('pretrained_encoder') is not True:
            raise ValueError('Transfer checkpoint does not identify the trained pretrained ResNet18 U-Net.')
        model = transfer_cls(pretrained=False)
    model.load_state_dict(ck['model_state_dict'], strict=True)
    model.eval()
    return model


def find_pairs(image_dir, mask_dir):
    imgs = {p.stem: p for p in image_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg')}
    masks = {p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() == '.png'}
    if len(imgs) != 20 or len(masks) != 20:
        raise ValueError(f'Expected exactly 20 input JPG and 20 GT PNG; found {len(imgs)} and {len(masks)}')
    pairs = []
    for stem, mask in sorted(masks.items()):
        # Pilot exporters may name D0 photos either C01L01... or D0C01L01...
        img = imgs.get(stem) or imgs.get('D0' + stem)
        if img is None:
            raise ValueError(f'No matching JPEG for GT {mask.name}')
        m = re.fullmatch(r'(?:D([357]))?(C0[1-5])L01正', stem)
        if not m:
            raise ValueError(f'Unexpected image identity: {stem}')
        pairs.append((stem, img, mask, int(m.group(1) or 0), m.group(2)))
    if sorted([day for _,_,_,day,_ in pairs]) != [0]*5+[3]*5+[5]*5+[7]*5:
        raise ValueError('Pilot sampling does not have five images at each of D0/D3/D5/D7')
    return pairs


def metrics(tp, fp, fn):
    d = 2 * tp + fp + fn
    u = tp + fp + fn
    return ((2 * tp / d) if d else 1.0, (tp / u) if u else 1.0)


def draw_sheet(rows, path, size=384):
    # Each row contains: displayed RGB, GT, custom prediction, transfer prediction.
    w = size * 4
    h = (size + 32) * len(rows)
    sheet = Image.new('RGB', (w, h), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, (name, rgb, gt, pred1, pred2) in enumerate(rows):
        y = i * (size + 32)
        draw.text((6,y+8), f'{name}   RGB | reviewed GT | Custom | ResNet18', fill='black')
        for j, arr in enumerate((rgb, gt, pred1, pred2)):
            im = Image.fromarray(arr)
            if im.mode != 'RGB':
                im = im.convert('L').convert('RGB')
            im.thumbnail((size, size), Image.Resampling.BILINEAR if j == 0 else Image.Resampling.NEAREST)
            sheet.paste(im.resize((size,size), Image.Resampling.BILINEAR if j == 0 else Image.Resampling.NEAREST), (j*size,y+32))
    sheet.save(path, quality=88)


def save_csv(path, fields, records):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', required=True, type=Path)
    parser.add_argument('--masks', required=True, type=Path)
    parser.add_argument('--custom-checkpoint', required=True, type=Path)
    parser.add_argument('--transfer-checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    # Refuse overwrites of existing experiment outputs.
    if args.output.exists():
        parser.error(f'Output already exists: {args.output}. Choose a fresh versioned folder.')
    for p in (args.images, args.masks):
        if not p.is_dir():
            parser.error(f'Missing directory: {p}')
    for p in (args.custom_checkpoint, args.transfer_checkpoint):
        if not p.is_file():
            parser.error(f'Missing checkpoint: {p}')
    torch.set_num_threads(min(4, torch.get_num_threads()))
    pairs = find_pairs(args.images, args.masks)
    project = Path(__file__).resolve().parent.parent
    LightweightUNet, ResNet18UNet = get_models(project)
    models = {
        'custom': load_model('custom', args.custom_checkpoint, LightweightUNet, ResNet18UNet),
        'resnet18': load_model('resnet18', args.transfer_checkpoint, LightweightUNet, ResNet18UNet)
    }
    args.output.mkdir(parents=True, exist_ok=False)
    preview_dir = args.output / 'previews'
    preview_dir.mkdir()
    counts = defaultdict(lambda: [0,0,0])
    records = []
    sheets_by_head = defaultdict(list)
    with torch.inference_mode():
        for stem, image_path, mask_path, day, head in pairs:
            with Image.open(image_path) as source:
                rgb = ImageOps.exif_transpose(source).convert('RGB')
                native_size = rgb.size
                small = rgb.resize((128,128), Image.Resampling.BILINEAR)
                x = torch.from_numpy(np.asarray(small, dtype=np.float32).transpose(2,0,1).copy()/255.0).unsqueeze(0)
            with Image.open(mask_path) as source:
                if source.size != native_size:
                    raise ValueError(f'EXIF-aligned dimensions mismatch for {stem}: RGB {native_size}, GT {source.size}')
                native_labels = np.asarray(source.convert('L'))
                if not np.isin(native_labels, [0,255]).all():
                    raise ValueError(f'GT is not strictly binary: {stem}')
                gt_small = source.convert('L').resize((128,128), Image.Resampling.NEAREST)
                truth = np.asarray(gt_small) > 127
            preds = {}
            for model_name, model in models.items():
                p = (torch.sigmoid(model(x))[0,0].numpy() >= 0.5)
                preds[model_name] = p
                tp = int(np.logical_and(p, truth).sum())
                fp = int(np.logical_and(p, ~truth).sum())
                fn = int(np.logical_and(~p, truth).sum())
                tn = int(np.logical_and(~p, ~truth).sum())
                dice, iou = metrics(tp,fp,fn)
                records.append(dict(image=stem, model=model_name, day=day, head=head,
                                    dice=round(dice,8), iou=round(iou,8), tp=tp,fp=fp,fn=fn,tn=tn,
                                    foreground_gt=int(truth.sum()), foreground_pred=int(p.sum())))
                for key in ('ALL',f'D{day}',head):
                    c = counts[(model_name,key)]
                    c[0] += tp; c[1] += fp; c[2] += fn
            # Preview uses full input aspect ratio compressed only for contact sheets.
            sheets_by_head[head].append((stem, np.asarray(small), (truth.astype('uint8')*255),
                                         (preds['custom'].astype('uint8')*255),
                                         (preds['resnet18'].astype('uint8')*255)))
    save_csv(args.output/'per_image_metrics.csv',list(records[0]),records)
    aggs=[]
    for (model,key),(tp,fp,fn) in sorted(counts.items()):
        dice,iou=metrics(tp,fp,fn)
        aggs.append(dict(model=model,group=key,n_images=(20 if key=='ALL' else 4 if key.startswith('C') else 5),
                         global_dice=round(dice,8),global_iou=round(iou,8),tp=tp,fp=fp,fn=fn))
    save_csv(args.output/'group_metrics.csv',list(aggs[0]),aggs)
    for head,rows in sorted(sheets_by_head.items()):
        draw_sheet(sorted(rows,key=lambda z: int(re.match(r'D([357])',z[0]).group(1)) if z[0].startswith('D') else 0),
                   preview_dir/f'{head}_RGB_GT_two_models.jpg')
    summary = {
        'design':'Zero-shot frozen models; 20 manually reviewed GT; 5 biological source heads; D0,D3,D5,D7',
        'evaluation_resolution':'128 x 128 for both RGB and nearest-neighbour GT masks',
        'normalization':'RGB 0..1; no ImageNet mean/std normalization, consistent with source training',
        'threshold':0.5,
        'source_domain_reference_not_target_results':{'custom_val_dice':0.9576,'resnet18_val_dice':0.9484},
        'results':{row['model']:{'target_global_dice':row['global_dice'],'target_global_iou':row['global_iou']}
                   for row in aggs if row['group']=='ALL'},
        'notes':'Metrics are on resized 128x128 masks; do not claim independent sample N=20 or original-resolution performance.'
    }
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print('PASS: Evaluated 20 target images with both frozen checkpoints')
    for name,data in summary['results'].items():
        print(f"{name}: target Global Dice={data['target_global_dice']:.4f} IoU={data['target_global_iou']:.4f}")
    print(f'Output: {args.output}')

if __name__=='__main__':
    main()
