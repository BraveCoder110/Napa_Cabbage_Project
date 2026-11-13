"""Session 6.2: train an analysis-only ResNet18 U-Net, predict 157 candidate ROIs.

Do not treat predictions as reviewed GT or use training-image scores as independent evaluation.
This script creates a NEW output directory and does not modify previous checkpoints.
"""
import argparse
import csv
import io
import json
import random
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch


def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def preview_sheet(entries, path):
    # Small side-by-side thumbnails; no change to source JPEG or final masks.
    thumb_w, thumb_h = 144, 310
    cell_w, cell_h = 305, 355
    cols = 5
    rows = (len(entries) + cols - 1) // cols
    canvas = Image.new('RGB', (cols * cell_w, rows * cell_h), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (name, photo, mask, status) in enumerate(entries):
        x0, y0 = (i % cols) * cell_w, (i // cols) * cell_h
        photo = photo.resize((thumb_w, thumb_h), Image.Resampling.BILINEAR)
        mask = mask.resize((thumb_w, thumb_h), Image.Resampling.NEAREST)
        overlay = np.array(photo).copy()
        fg = np.asarray(mask) > 127
        green = np.array([10, 220, 60], dtype=np.float32)
        overlay[fg] = (overlay[fg].astype(np.float32) * 0.53 + green * 0.47).astype(np.uint8)
        canvas.paste(photo, (x0, y0 + 24))
        canvas.paste(Image.fromarray(overlay), (x0 + thumb_w + 6, y0 + 24))
        draw.text((x0 + 3, y0 + 4), name, fill='black')
        draw.text((x0 + 3, y0 + 336), status, fill='black')
    canvas.save(path, quality=85)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--images', type=Path, required=True, help='20 reviewed pilot RGB files')
    ap.add_argument('--masks', type=Path, required=True, help='20 reviewed pilot binary masks')
    ap.add_argument('--transfer-checkpoint', type=Path, required=True)
    for d in (0, 3, 5, 7):
        ap.add_argument(f'--d{d}', type=Path, required=True, help=f'Original D{d} JPG ZIP')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--lr', type=float, default=0.0001)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args()
    if args.output.exists():
        ap.error(f'Output already exists; refusing overwrite: {args.output}')
    if not 1 <= args.epochs <= 30 or not 0 < args.lr < 0.1:
        ap.error('Invalid epochs or learning rate')
    for p in [args.images, args.masks]:
        if not p.is_dir():
            ap.error(f'Missing directory: {p}')
    for p in [args.transfer_checkpoint, args.d0, args.d3, args.d5, args.d7]:
        if not p.is_file():
            ap.error(f'Missing input file: {p}')

    # Import previously validated project modules without modifying them.
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root / 'E3_Session5'))
    from evaluate_zero_shot import get_models, load_model
    from run_grouped_few_shot import read_data, augment, loss_fn
    torch.set_num_threads(max(1, min(args.threads, 16)))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = read_data(args.images, args.masks)
    custom_cls, transfer_cls = get_models(project_root)
    model = load_model('resnet18', args.transfer_checkpoint, custom_cls, transfer_cls)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Pre-flight archive audit BEFORE creating any output.
    archive_entries = []
    expected_count = {0: 37, 3: 40, 5: 40, 7: 40}
    seen = set()
    for day in (0, 3, 5, 7):
        archive = getattr(args, f'd{day}')
        with zipfile.ZipFile(archive) as z:
            members = [m for m in z.infolist() if not m.is_dir() and m.filename.lower().endswith(('.jpg', '.jpeg'))]
            if len(members) != expected_count[day]:
                ap.error(f'D{day}: expected {expected_count[day]} JPG, found {len(members)}')
            for info in members:
                stem = Path(info.filename).stem
                pat = r'(?:D[357])?C0[1-5]L0[1-4][正反]'
                if re.fullmatch(pat, stem) is None:
                    ap.error(f'Unexpected image name: {info.filename}')
                name = stem if day != 0 else (stem if stem.startswith('D0') else 'D0'+stem)
                if name in seen:
                    ap.error(f'Duplicate name across ZIPs: {name}')
                seen.add(name)
                archive_entries.append((day, archive, info.filename, name))
    if len(seen) != 157:
        ap.error(f'Expected 157 unique photographs, got {len(seen)}')

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'masks_full_resolution').mkdir()
    (args.output/'masks_128').mkdir()
    (args.output/'previews_by_day').mkdir()
    hist = []
    rng = random.Random(args.seed)
    print('Training ONE analysis-only model on all 20 manually reviewed GT images.', flush=True)
    for ep in range(args.epochs):
        model.train()
        order = list(range(len(data)))
        rng.shuffle(order)
        losses = []
        for start in range(0,len(data),2):
            batch = [augment(data[j]['x'],data[j]['y'],rng) for j in order[start:start+2]]
            x = torch.stack([p[0] for p in batch])
            y = torch.stack([p[1] for p in batch])
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            if not torch.isfinite(loss).item():
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        mean_loss = float(np.mean(losses))
        hist.append(dict(epoch=ep+1, mean_train_loss=round(mean_loss,8), n_training_images=20))
        print(f'  epoch {ep+1}/{args.epochs}: train loss={mean_loss:.5f}', flush=True)
    save_csv(args.output/'training_history.csv', hist)

    # Save the deployment checkpoint, separate from earlier evaluated source checkpoints.
    checkpoint_path = args.output/'resnet18_target_analysis_only.pth'
    torch.save({'architecture':'ResNet18UNet','model_state_dict':model.state_dict(),
                'source_checkpoint_name':args.transfer_checkpoint.name,
                'config':{'size':128,'epochs':args.epochs,'lr':args.lr,'seed':args.seed},
                'training_image_count':20, 'training_heads':5,
                'purpose':'Candidate ROI generation on the same image collection; not independent performance evaluation'}, checkpoint_path)

    model.eval()
    manifest = []
    day_previews = defaultdict(list)
    counts = Counter()
    for index,(day,archive,member,stem) in enumerate(sorted(archive_entries, key=lambda e:(e[0],e[3]))):
        with zipfile.ZipFile(archive) as z:
            raw = z.read(member)
        with Image.open(io.BytesIO(raw)) as source:
            rgb = ImageOps.exif_transpose(source).convert('RGB')
            original_size = rgb.size
            small = rgb.resize((128,128), Image.Resampling.BILINEAR)
        if original_size != (1868,4000):
            raise ValueError(f'Unexpected EXIF-aligned dimensions: {stem}: {original_size}')
        x = torch.from_numpy(np.asarray(small,dtype=np.float32).transpose(2,0,1).copy()/255.0).unsqueeze(0)
        with torch.inference_mode():
            prob = model(x).sigmoid()[0,0].cpu().numpy()
        seg = (prob>=0.5).astype(np.uint8)*255
        # Nearest neighbour resize keeps masks binary. The full-res mask is an UPSCALED
        # 128x128 prediction, not evidence of independent high-resolution accuracy.
        mini = Image.fromarray(seg,mode='L')
        full = mini.resize(original_size,Image.Resampling.NEAREST)
        mini.save(args.output/'masks_128'/f'{stem}.png')
        full.save(args.output/'masks_full_resolution'/f'{stem}.png',optimize=True)
        frac = float((seg>0).mean())
        flags=[]
        if frac<0.05: flags.append('VERY_SMALL_ROI')
        if frac>0.90: flags.append('VERY_LARGE_ROI')
        if np.any(seg[0,:]) or np.any(seg[-1,:]) or np.any(seg[:,0]) or np.any(seg[:,-1]):
            flags.append('TOUCHES_IMAGE_BORDER')
        status = ';'.join(flags) if flags else 'AUTO_CANDIDATE_REVIEW_REQUIRED'
        manifest.append(dict(image=stem+'.jpg',day=day,source_head=stem[-8:-5] if False else re.search(r'C0[1-5]',stem).group(),
                             leaf_id=re.search(r'L0[1-4]',stem).group(),side=stem[-1],
                             source_zip=archive.name, source_member=member,
                             roi_fraction_128=round(frac,6),mean_probability=round(float(prob.mean()),6),
                             status_flags=status,annotation_type='MODEL_PREDICTION_NOT_GROUND_TRUTH'))
        # Preview only uses lightweight copies, not the 4000px images.
        day_previews[day].append((stem, rgb.resize((144,310),Image.Resampling.BILINEAR),mini,status[:29]))
        counts[day]+=1
        if (index+1)%20==0 or index+1==157:
            print(f'Predicted {index+1}/157 candidate ROIs',flush=True)
    save_csv(args.output/'candidate_roi_manifest_157.csv',manifest)
    for day,entries in day_previews.items():
        preview_sheet(entries,args.output/'previews_by_day'/f'D{day}_all_candidates.jpg')
    info={'status':'COMPLETED_CANDIDATE_GENERATION_NOT_HUMAN_GT',
          'total_predictions':len(manifest),'counts_by_day':dict(counts),
          'model':'ResNet18UNet initialized from source-domain checkpoint; trained on all 20 pilot GT images',
          'preprocessing':'EXIF transpose; RGB 0..1; bilinear 128x128; threshold 0.5; binary nearest-neighbor upsample',
          'training':{'n_images':20,'source_heads':5,'epochs':args.epochs,'lr':args.lr,'seed':args.seed},
          'warning':'All 157 candidate masks require ROI quality review. Pilot 20 overlap with training; do not use them as independent validation. Full resolution masks are upsampled 128x128 outputs. All images have not been assigned freshness labels.',
          'recommendation':'For final longitudinal color metrics, use original EXIF-oriented RGB pixels within reviewed/corrected masks; report ROI QC and sensitivity.'}
    (args.output/'analysis_model_protocol.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PASS: Trained final analysis-only model; saved checkpoint and 157 candidate ROIs')
    print(f'Output: {args.output}')
    print('IMPORTANT: masks are AUTOMATIC CANDIDATES, not independently reviewed GT.')

if __name__ == '__main__':
    main()
