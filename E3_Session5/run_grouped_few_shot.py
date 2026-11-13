"""Session 5.3: leakage-controlled five-head leave-one-group-out adaptation.

Uses source-trained checkpoints as fixed initializers for each fold.  All source
heads in a held-out fold remain unseen during target-domain adaptation.
No existing file or output folder is overwritten.
"""
import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F


def metric_counts(pred, truth):
    return (int(np.logical_and(pred, truth).sum()),
            int(np.logical_and(pred, ~truth).sum()),
            int(np.logical_and(~pred, truth).sum()))


def scores(c):
    tp, fp, fn = c
    return (2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
            tp / (tp + fp + fn) if tp + fp + fn else 1.0)


def read_data(image_dir, mask_dir):
    # Reuse audited pairing and EXIF-aware preprocessing from Session 5.2.
    from evaluate_zero_shot import find_pairs
    pairs = find_pairs(image_dir, mask_dir)
    data = []
    for stem, image_path, mask_path, day, head in pairs:
        with Image.open(image_path) as f:
            rgb = ImageOps.exif_transpose(f).convert('RGB')
            true_size = rgb.size
            a = np.asarray(rgb.resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float32).copy() / 255.0
        with Image.open(mask_path) as f:
            if f.size != true_size:
                raise ValueError(f'Image/mask direction mismatch: {stem}')
            raw = np.asarray(f.convert('L'))
            if not np.isin(raw, [0, 255]).all():
                raise ValueError(f'Nonbinary GT: {stem}')
            m = np.asarray(f.convert('L').resize((128,128), Image.Resampling.NEAREST), dtype=np.uint8).copy() > 127
        data.append(dict(stem=stem, day=day, head=head,
                         x=torch.from_numpy(a.transpose(2, 0, 1).copy()),
                         y=torch.from_numpy(m[None].astype(np.float32).copy())))
    if len(data) != 20 or sorted(set(d['head'] for d in data)) != ['C01','C02','C03','C04','C05']:
        raise ValueError('Expect five source heads and 20 labeled images.')
    return data


def augment(x, y, rng):
    # Training-only paired geometry; independent mild photometric change to RGB.
    if rng.random() < 0.5:
        x, y = x.flip(-1), y.flip(-1)
    if rng.random() < 0.5:
        x, y = x.flip(-2), y.flip(-2)
    k = rng.randrange(4)
    x, y = torch.rot90(x, k, (-2,-1)), torch.rot90(y, k, (-2,-1))
    brightness = rng.uniform(0.9, 1.1)
    contrast = rng.uniform(0.9, 1.1)
    x = ((x - 0.5) * contrast + 0.5) * brightness
    return x.clamp(0,1), y


def loss_fn(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = logits.sigmoid()
    dice = (2 * (p * target).sum(dim=(1,2,3)) + 1) / (p.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + 1)
    return bce + (1 - dice).mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--images', required=True, type=Path)
    ap.add_argument('--masks', required=True, type=Path)
    ap.add_argument('--custom-checkpoint', required=True, type=Path)
    ap.add_argument('--transfer-checkpoint', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--model', choices=['resnet18','custom'], default='resnet18')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--lr', type=float, default=0.0001)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args()
    if args.output.exists():
        ap.error(f'Output exists (refusing overwrite): {args.output}')
    if args.epochs < 1 or args.epochs > 30 or not (0 < args.lr < 0.1):
        ap.error('Check epochs (1–30) and learning rate (0 < lr < 0.1)')
    for p in [args.images,args.masks]:
        if not p.is_dir(): ap.error(f'Missing directory: {p}')
    checkpoint = args.transfer_checkpoint if args.model == 'resnet18' else args.custom_checkpoint
    if not checkpoint.is_file(): ap.error(f'Missing model checkpoint: {checkpoint}')
    torch.set_num_threads(max(1,min(args.threads,torch.get_num_threads())))
    # Relative import resolves because this script and evaluate_zero_shot.py are siblings.
    from evaluate_zero_shot import get_models, load_model
    project = Path(__file__).resolve().parent.parent
    custom_cls, transfer_cls = get_models(project)
    data = read_data(args.images,args.masks)
    heads = sorted(set(d['head'] for d in data))
    args.output.mkdir(parents=True,exist_ok=False)
    records, hist, manifest = [], [], []
    agg = defaultdict(lambda: [0,0,0])
    for fold_index, holdout in enumerate(heads):
        rng = random.Random(args.seed + fold_index)
        torch.manual_seed(args.seed + fold_index)
        np.random.seed(args.seed + fold_index)
        train = [d for d in data if d['head'] != holdout]
        test = [d for d in data if d['head'] == holdout]
        assert len(train)==16 and len(test)==4
        for d in data:
            manifest.append(dict(fold=holdout, image=d['stem'], head=d['head'], day=d['day'],
                                 role='held_out_test' if d['head']==holdout else 'train'))
        model = load_model(args.model,checkpoint,custom_cls,transfer_cls)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        print(f'Fold {fold_index+1}/5: held out {holdout}; train=16 images/4 heads; test=4 images',flush=True)
        for ep in range(args.epochs):
            model.train()
            order=list(range(16))
            rng.shuffle(order)
            losses=[]
            for offset in range(0,16,2):
                batch=[augment(train[i]['x'],train[i]['y'],rng) for i in order[offset:offset+2]]
                xb=torch.stack([a for a,b in batch]); yb=torch.stack([b for a,b in batch])
                optimizer.zero_grad(set_to_none=True)
                z=model(xb)
                loss=loss_fn(z,yb)
                if not torch.isfinite(loss).item(): raise RuntimeError(f'Non-finite training loss: {holdout} epoch{ep+1}')
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
            avg=sum(losses)/len(losses)
            hist.append(dict(held_out_head=holdout,epoch=ep+1,train_loss=round(avg,7),n_train=16))
            print(f'  epoch {ep+1}/{args.epochs}, train_loss={avg:.5f}',flush=True)
        model.eval()
        with torch.inference_mode():
            for d in test:
                probs=model(d['x'][None]).sigmoid()[0,0].cpu().numpy()
                pred=probs>=0.5
                gt=d['y'][0].numpy()>0.5
                c=metric_counts(pred,gt)
                dice,iou=scores(c)
                records.append(dict(image=d['stem'],head=d['head'],day=d['day'],model=args.model,
                                    held_out_head=holdout,dice=round(dice,8),iou=round(iou,8),
                                    tp=c[0],fp=c[1],fn=c[2],pred_fraction=round(float(pred.mean()),8)))
                for i,v in enumerate(c): agg['ALL'][i]+=v;agg[holdout][i]+=v
        print('  held-out Dice='+'%.4f'%scores(agg[holdout])[0],flush=True)
    def dump(filename,rows):
        with (args.output/filename).open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    dump('oof_per_image_metrics.csv',records)
    dump('training_history.csv',hist)
    dump('fold_membership.csv',manifest)
    groups=[]
    for k,c in sorted(agg.items()):
        a,b=scores(c)
        groups.append(dict(group=k,images=20 if k=='ALL' else 4,global_dice=round(a,8),global_iou=round(b,8),tp=c[0],fp=c[1],fn=c[2]))
    dump('oof_group_metrics.csv',groups)
    summary={'protocol':'Pre-specified 5-fold leave-one-source-head-out; each model reset to source checkpoint at fold start',
             'model':args.model,'epochs':args.epochs,'lr':args.lr,'seed':args.seed,
             'target_images':20,'source_heads':5,'training_images_per_fold':16,
             'test_images_per_fold':4,'test_image_reuse_across_folds':False,
             'threshold':0.5,'img_size':128,
             'important':'Out-of-fold adaptation estimate; same 20 manually reviewed samples were used for earlier zero-shot diagnostics. Pilot, not untouched external test.',
             'overall':groups[0] if groups[0]['group']=='ALL' else next(v for v in groups if v['group']=='ALL')}
    (args.output/'protocol_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PASS: 5 head-held-out folds; 20 out-of-fold predictions, no head leakage')
    print(f"{args.model} adapted OOF Global Dice={summary['overall']['global_dice']:.4f} IoU={summary['overall']['global_iou']:.4f}")
    print(f'Output: {args.output}')

if __name__=='__main__':
    main()
