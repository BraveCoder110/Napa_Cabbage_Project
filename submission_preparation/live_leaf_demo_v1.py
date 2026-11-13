#!/usr/bin/env python3
"""CS7002SR: one-photo research demo, not a calibrated freshness grader.

Uses the existing, analysis-only ResNet18UNet checkpoint (no training).
Read-only inputs. Output directory must not exist. Python 3.10+.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import html
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch

DEFAULT_CHECKPOINT = Path('experiments/session6_2_analysis_roi_candidates_v1/resnet18_target_analysis_only.pth')
VALID_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}


def select_image_file():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        p = filedialog.askopenfilename(title='Select your new cabbage leaf photo', filetypes=[('Image files','*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff'),('All files','*.*')])
        root.destroy()
        return Path(p) if p else None
    except Exception as e:
        raise RuntimeError('Graphical file selection is unavailable. Use --image "path/to/your/photo.jpg".') from e


def validate_figure(x):
    if not math.isfinite(float(x)):
        raise ValueError('Non-finite numerical feature.')
    return float(x)


def save_small(img: Image.Image, out: Path, max_size=(1000, 1400)):
    im = img.copy()
    im.thumbnail(max_size, Image.Resampling.LANCZOS)
    im.save(out, format='PNG')


def make_colour_map(rgb: np.ndarray, roi: np.ndarray, green: np.ndarray, yellow: np.ndarray, brown: np.ndarray) -> Image.Image:
    # Brown and yellow may overlap because the original project definitions overlap.
    # Display precedence: brown > yellow > green. Numeric ratios remain independent.
    base = rgb.astype(np.float32) * .28
    base[roi] = rgb[roi].astype(np.float32) * .55
    base[green] = [18, 183, 72]
    base[yellow] = [255, 217, 42]
    base[brown] = [173, 83, 40]
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def feature_values(rgb: np.ndarray, mask: np.ndarray, stride: int):
    # Source logic: E3_Session6/extract_visual_freshness_features.py:metric_values.
    sm_rgb = rgb[::stride, ::stride]
    sm_roi = mask[::stride, ::stride]
    if int(sm_roi.sum()) < 300:
        return {'roi_pixel_count_sampled': int(sm_roi.sum()), 'roi_fraction':float(sm_roi.mean()), 'mean_r':None,'mean_g':None,'mean_b':None,'mean_h':None,'mean_s':None,'mean_v':None,'green_ratio':None,'yellow_candidate_ratio':None,'brown_candidate_ratio':None,'texture_neighbor_diff':None}, None
    hsv = cv2.cvtColor(sm_rgb, cv2.COLOR_RGB2HSV)
    pix_rgb = sm_rgb[sm_roi].astype(np.float32)
    pix_hsv = hsv[sm_roi]
    h,s,v = [pix_hsv[:,i].astype(np.int16) for i in range(3)]
    green = (h>=35)&(h<=85)&(s>=50)&(v>=40)
    yellow = (h>=15)&(h<35)&(s>=45)&(v>=55)
    brown = (h>=5)&(h<25)&(s>=50)&(v>=35)&(v<180)
    gray = cv2.cvtColor(sm_rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)
    horiz = sm_roi[:,:-1] & sm_roi[:,1:]
    vert = sm_roi[:-1,:] & sm_roi[1:,:]
    differences = []
    if horiz.any(): differences.append(np.abs(gray[:,:-1]-gray[:,1:])[horiz])
    if vert.any(): differences.append(np.abs(gray[:-1,:]-gray[1:,:])[vert])
    texture = float(np.concatenate(differences).mean()) if differences else None
    result = {
        'roi_pixel_count_sampled':int(sm_roi.sum()),
        'roi_fraction':validate_figure(sm_roi.mean()),
        'mean_r':validate_figure(pix_rgb[:,0].mean()),
        'mean_g':validate_figure(pix_rgb[:,1].mean()),
        'mean_b':validate_figure(pix_rgb[:,2].mean()),
        'mean_h':validate_figure(h.mean()),
        'mean_s':validate_figure(s.mean()),
        'mean_v':validate_figure(v.mean()),
        'green_ratio':validate_figure(green.mean()),
        'yellow_candidate_ratio':validate_figure(yellow.mean()),
        'brown_candidate_ratio':validate_figure(brown.mean()),
        'texture_neighbor_diff':validate_figure(texture) if texture is not None else None,
    }
    return result, (h,s,v)


def draw_bars(features, dest):
    W,H=980,420
    image=Image.new('RGB',(W,H),'white'); d=ImageDraw.Draw(image)
    entries=[('Green candidate',features['green_ratio'],(25,154,80)),('Yellow candidate',features['yellow_candidate_ratio'],(230,179,31)),('Brown candidate',features['brown_candidate_ratio'],(142,87,53))]
    d.text((35,25),'HSV candidate-colour ratios inside predicted ROI',fill='#172533')
    d.text((35,50),'Heuristic, not verified freshness categories. Bins can overlap.',fill='#56606b')
    for j,(label,p,col) in enumerate(entries):
        y=110+j*95
        d.text((35,y+12),label,fill='#172533')
        d.rectangle((240,y+7,800,y+38),fill='#e8eaee')
        if p is not None: d.rectangle((240,y+7,240+int(p*560),y+38),fill=col)
        d.text((820,y+12),f'{100*p:.2f}%' if p is not None else 'N/A',fill='#172533')
    image.save(dest)


def make_html(out, src_name, model_name, feat, warnings):
    cards=[('01','Original upright photo','01_input_upright.png','EXIF rotation is corrected; image data is not colour-calibrated.'),
           ('02','Model input (128 x 128)','02_model_input_128.png','Resized RGB / 255. This is the tensor supplied to the network.'),
           ('03','Predicted probability map','03_probability_heatmap.png','Sigmoid of model logits; NOT a calibrated probability of being fresh.'),
           ('04','Binary leaf mask','04_binary_mask.png','Pixel is foreground if predicted value is >= 0.5.'),
           ('05','Overlay on original image','05_segmentation_overlay.png','A visual check for leakage onto the table and missing leaf regions.'),
           ('06','Leaf isolated from background','06_foreground_only.png','Uses the same predicted mask. Not a new model prediction.'),
           ('07','HSV colour candidate highlights','07_colour_candidates.png','Green, yellow and brown candidate pixels are illustrative. Colour groups can overlap.'),
           ('08','Candidate colour ratio chart','08_candidate_ratio_chart.png','Fractions of sampled pixels INSIDE the model-selected ROI.')]
    style='''body{font:16px/1.55 Arial,sans-serif;background:#f4f6f7;color:#172533;margin:0}main{max-width:1050px;margin:auto;padding:30px}h1{margin:0}.sub{color:#596573}.alert{background:#fff1dc;padding:15px;border-left:5px solid #a87012;margin:22px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:18px}.card{background:#fff;border-radius:12px;padding:17px;box-shadow:0 2px 9px #d7dee4}img{width:100%;max-height:390px;object-fit:contain;background:#f3f3f3}table{border-collapse:collapse;background:white;width:100%}td,th{border-bottom:1px solid #ddd;padding:9px;text-align:left}footer{color:#596573;margin:30px 0}'''
    chunks=[f'<!doctype html><html lang="en"><meta charset="utf-8"><title>Research leaf demo</title><style>{style}</style><main>',
        '<h1>Live Leaf Image Analysis — Research Demonstration</h1>',f'<p class="sub">Input: {html.escape(src_name)} · Model: {html.escape(model_name)} · One image only</p>',
        '<div class="alert"><strong>Research boundary:</strong> This is a predicted leaf mask and exploratory visual features. No verified fresh/moderate/spoiled label, food safety assessment, or ground-truth Dice for this newly captured photograph.</div>',
        '<h2>Processing stages</h2><div class="grid">']
    for n,title,pict,desc in cards:
        chunks.append(f'<article class="card"><h3>{n}. {html.escape(title)}</h3><img src="{pict}" alt="{html.escape(title)}"><p>{html.escape(desc)}</p></article>')
    chunks+=['</div><h2>Numerical output</h2><table><tr><th>Measurement</th><th>Value</th><th>Interpretation</th></tr>']
    for key,description in [('roi_fraction','Fraction of photograph covered by predicted leaf'),('green_ratio','Candidate green pixels / ROI pixels'),('yellow_candidate_ratio','Candidate yellow pixels / ROI pixels'),('brown_candidate_ratio','Candidate brown pixels / ROI pixels'),('texture_neighbor_diff','Average neighboring grayscale difference (arbitrary units)')]:
        v=feat[key]
        value= 'N/A' if v is None else (f'{v*100:.2f}%' if key!='texture_neighbor_diff' else f'{v:.3f}')
        chunks.append(f'<tr><td>{html.escape(key)}</td><td>{html.escape(value)}</td><td>{html.escape(description)}</td></tr>')
    chunks+=['</table><h2>Quality alerts</h2><ul>']
    for w in warnings: chunks.append('<li>'+html.escape(w)+'</li>')
    if not warnings:chunks.append('<li>None triggered by simple automated checks. Visual inspection is still required.</li>')
    chunks+=['</ul><footer>All results are descriptive. Appearance is sensitive to camera, illumination, background, and segmentation errors.</footer></main></html>']
    (out/'demo_report.html').write_text('\n'.join(chunks),encoding='utf-8')


def run(args):
    root=Path(args.project_root).expanduser().resolve()
    if not (root/'E3_Session4/models/resnet18_unet.py').is_file():
        raise FileNotFoundError(f'Model definition missing under project root: {root}')
    photo = Path(args.image).expanduser().resolve() if args.image else select_image_file()
    if photo is None: raise RuntimeError('No image selected. Nothing was written.')
    if not photo.is_file() or photo.suffix.lower() not in VALID_EXTS:
        raise ValueError(f'Please select an existing supported photograph, got: {photo}')
    ckpt=(root/args.checkpoint).resolve() if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint).resolve()
    if not ckpt.is_file():raise FileNotFoundError(f'Missing analysis checkpoint: {ckpt}')
    out=Path(args.output).expanduser().resolve()
    if out.exists():raise FileExistsError(f'Output already exists; use a NEW output directory: {out}')
    if out==root or out==photo.parent or out==ckpt.parent:raise ValueError('Choose a dedicated NEW output folder.')
    if args.stride<1 or args.stride>16:raise ValueError('stride must be 1..16; standard=4')
    print('STEP 1/8 Read photo and apply EXIF rotation')
    with Image.open(photo) as source:
        rgb_image=ImageOps.exif_transpose(source).convert('RGB')
    w,h=rgb_image.size
    if min(w,h)<128:raise ValueError('Image too small: minimum side must be >=128 pixels.')
    print(f'  Upright image: {w} x {h}')
    print('STEP 2/8 Prepare 128 x 128 model input')
    mini=rgb_image.resize((128,128),Image.Resampling.BILINEAR)
    x=torch.from_numpy(np.asarray(mini,dtype=np.float32).transpose(2,0,1).copy()/255.0).unsqueeze(0)
    print('STEP 3/8 Safely load the original analysis-only checkpoint')
    sys.path.insert(0,str(root/'E3_Session4/models'))
    from resnet18_unet import ResNet18UNet
    ck=torch.load(str(ckpt),map_location='cpu',weights_only=True)
    if not isinstance(ck,dict) or ck.get('architecture')!='ResNet18UNet' or 'model_state_dict' not in ck:
        raise ValueError('This is not the expected analysis-only ResNet18UNet checkpoint.')
    if 'analysis' not in str(ck.get('purpose','')).lower() and 'candidate' not in str(ck.get('purpose','')).lower():
        raise ValueError('Purpose field does not identify the target analysis ROI model. Use the correct default checkpoint.')
    if int(ck.get('config',{}).get('size',0))!=128:
        raise ValueError('Checkpoint input size is not the expected 128.')
    model=ResNet18UNet(pretrained=False).cpu()
    model.load_state_dict(ck['model_state_dict'],strict=True)
    model.eval()
    print('STEP 4/8 Run CPU inference and sigmoid')
    with torch.inference_mode():
        logits=model(x)
        assert tuple(logits.shape)==(1,1,128,128),tuple(logits.shape)
        prob=torch.sigmoid(logits)[0,0].cpu().numpy()
    if not np.isfinite(prob).all():raise RuntimeError('Nonfinite model output')
    seg=(prob>=0.5).astype(np.uint8)*255
    if seg.min()==seg.max():print('  WARNING: mask is all background or all foreground')
    print('STEP 5/8 Make binary mask, full-size mask, overlay')
    mask_full=Image.fromarray(seg,mode='L').resize((w,h),Image.Resampling.NEAREST)
    roi=np.asarray(mask_full)>127
    rgb=np.asarray(rgb_image)
    overlay=np.asarray(rgb_image,dtype=np.float32).copy()
    overlay[roi]=overlay[roi]*.60+np.array([12,220,98])*.40
    isolated=np.full_like(rgb,255)
    isolated[roi]=rgb[roi]
    print('STEP 6/8 Calculate ORIGINAL-project HSV candidate thresholds on sampled upright RGB')
    features,_=feature_values(rgb,roi,args.stride)
    small=rgb[::args.stride,::args.stride]
    hh=cv2.cvtColor(small,cv2.COLOR_RGB2HSV)
    H,S,V=[hh[:,:,i] for i in range(3)]
    green=(H>=35)&(H<=85)&(S>=50)&(V>=40)
    yellow=(H>=15)&(H<35)&(S>=45)&(V>=55)
    brown=(H>=5)&(H<25)&(S>=50)&(V>=35)&(V<180)
    small_roi=roi[::args.stride,::args.stride]
    colour=make_colour_map(small,small_roi,green&small_roi,yellow&small_roi,brown&small_roi)
    warnings=[]
    if features['roi_pixel_count_sampled']<300:warnings.append('Too few foreground pixels for reliable colour calculations; colour outputs shown as N/A.')
    if features['roi_fraction']<.05:warnings.append('Very small segmented region; inspect the overlay.')
    if features['roi_fraction']>.90:warnings.append('Very large segmented region; possible background leakage.')
    if min(w,h)>3_000:warnings.append('High-resolution photo: candidate colour ratios use stride sampling of original pixels.')
    warnings.append('New images have no GT: Dice/IoU cannot be calculated from this photograph.')
    warnings.append('Model not validated on uncontrolled live photos; camera lighting and background may bias features.')
    print('STEP 7/8 Save stage outputs to a NEW folder')
    out.mkdir(parents=True,exist_ok=False)
    save_small(rgb_image,out/'01_input_upright.png')
    mini.save(out/'02_model_input_128.png')
    prob_heat=cv2.applyColorMap((prob*255).clip(0,255).astype(np.uint8),cv2.COLORMAP_TURBO)
    Image.fromarray(cv2.cvtColor(prob_heat,cv2.COLOR_BGR2RGB)).resize((512,512),Image.Resampling.NEAREST).save(out/'03_probability_heatmap.png')
    Image.fromarray(seg).resize((512,512),Image.Resampling.NEAREST).save(out/'04_binary_mask.png')
    mask_full.save(out/'04_binary_mask_full_size.png')
    save_small(Image.fromarray(np.clip(overlay,0,255).astype(np.uint8)),out/'05_segmentation_overlay.png')
    save_small(Image.fromarray(isolated),out/'06_foreground_only.png')
    save_small(colour,out/'07_colour_candidates.png')
    draw_bars(features,out/'08_candidate_ratio_chart.png')
    with (out/'feature_summary.csv').open('w',newline='',encoding='utf-8-sig') as fh:
        writer=csv.writer(fh);writer.writerow(['feature','value','unit_or_note'])
        for k,v in features.items():writer.writerow([k,v,'fraction' if k.endswith('ratio') or k=='roi_fraction' else 'original analysis unit'])
    rec={'demo_version':'live_leaf_demo_v1','source_photo':str(photo),'source_sha256':hashlib.sha256(photo.read_bytes()).hexdigest(),
         'model_checkpoint':str(ckpt),'checkpoint_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest(),
         'model_architecture':'ResNet18UNet','model_type':'Session6 analysis-only (all 20 reviewed target masks)','image_upright_size':[w,h],
         'image_model_size':[128,128], 'colour_sample_stride':args.stride,'mask_threshold':0.5,
         'mask_128_foreground_fraction':float((seg>0).mean()), 'feature_metrics':features,'warnings':warnings,
         'research_boundary':'No validated freshness grade, no ground truth for this photo, no per-photo Dice.'}
    (out/'demo_metadata.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding='utf-8')
    make_html(out,photo.name,'ResNet18UNet (Session 6.2 analysis-only)',features,warnings)
    print('STEP 8/8 Summary and interpretation')
    for k in ('roi_fraction','green_ratio','yellow_candidate_ratio','brown_candidate_ratio'):
        print(f'  {k}: {features[k]*100:.2f}%' if features[k] is not None else f'  {k}: N/A (ROI too small)')
    print('  IMPORTANT: NO FRESHNESS GRADE OR FOOD-SAFETY RESULT.')
    print(f'  Open: {out / "demo_report.html"}')
    return 0


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',default='.',help='Existing Napa_Cabbage_Project root; default current directory')
    ap.add_argument('--image',help='One new photo; if omitted, launches a file-picker window')
    ap.add_argument('--checkpoint',default=str(DEFAULT_CHECKPOINT),help='Session6 analysis-only checkpoint')
    ap.add_argument('--output',required=True,help='New, nonexistent result folder')
    ap.add_argument('--stride',type=int,default=4,help='Feature sampling stride (matches Session6 default)')
    args=ap.parse_args()
    try:return run(args)
    except Exception as e:
        print('DEMO FAILED: '+type(e).__name__+': '+str(e),file=sys.stderr)
        return 1
if __name__=='__main__':raise SystemExit(main())
