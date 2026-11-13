#!/usr/bin/env python3
"""Build a conservative, auditable CS7002SR artifact folder. Read-only sources.

No training; no destructive operations; dataset licence review is NOT automated.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, shutil, sys, zipfile
from pathlib import Path

EXPERIMENT_DIRS = [
    'session3_4_baseline_v2','session3_4_evaluation_v2',
    'session4_4_transfer_v1','session4_4_evaluation_v1','session4_5_comparison_v1',
    'session5_2_zero_shot_v1','session5_3_custom_loho_v1',
    'session5_3_resnet18_loho_v1','session5_3_model_comparison_v1',
    'session6_1_longitudinal_audit_v1',
    'session6_3_visual_features_provisional_v1',
    'session6_4_gt_auto_sensitivity_v2',
    'session6_4_2_photo_color_reproduction_v1',
    'session6_5_report_results_v1',
    'v5_1_source_custom_recheck_new','v5_1_source_resnet18_recheck_new',
]
ROI_FILES = [
    'experiments/session6_2_analysis_roi_candidates_v1/resnet18_target_analysis_only.pth',
    'experiments/session6_2_analysis_roi_candidates_v1/analysis_model_protocol.json',
    'experiments/session6_2_analysis_roi_candidates_v1/candidate_roi_manifest_157.csv',
    'experiments/session6_2_analysis_roi_candidates_v1/training_history.csv',
]
OPTIONAL_PREFIXES = ['submission_preparation/audit_results_v2',
    'submission_preparation/functional_check_results_v3',
    'submission_preparation/analysis_checkpoint_results_v3_1',
    'submission_preparation/evidence_audit_results_v4',
    'submission_preparation/environment_license_results_v5']


def sha256(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while True:
            b=f.read(1024*1024)
            if not b:break
            h.update(b)
    return h.hexdigest()


def all_files(dirpath):
    return sorted(p for p in dirpath.rglob('*') if p.is_file() and not p.is_symlink())


def source_map(root, include_target):
    """Explicit allowlist; no speculative copying of public dataset originals."""
    items={}
    def add(rel, required=False, label='evidence'):
        p=root/rel
        if not p.is_file():
            if required:raise FileNotFoundError('Required file missing: '+str(rel))
            return
        items[rel]=(p,label)
    for folder in ['E3_Session2','E3_Session3','E3_Session4','E3_Session5','E3_Session6']:
        d=root/folder
        if not d.is_dir():raise FileNotFoundError('Required code folder missing: '+str(d))
        for p in all_files(d):
            if p.suffix.lower() in {'.py','.txt','.md'}:
                add(p.relative_to(root),'required' if p.suffix.lower()=='.py' else 'documentation')
    for rel in ['experiments/session3_4_baseline_v2/best_unet.pth',
                'experiments/session4_4_transfer_v1/best_resnet18_unet.pth',
                *ROI_FILES]:add(Path(rel),required=True,label='model_or_protocol')
    for folder in EXPERIMENT_DIRS:
        d=root/'experiments'/folder
        if not d.is_dir():raise FileNotFoundError('Required evidence folder missing: '+str(d))
        for p in all_files(d):
            if p.suffix.lower() in {'.csv','.json','.txt','.md','.png','.jpg','.jpeg'}:
                add(p.relative_to(root),label='experimental_evidence')
    # 20 reviewed GT masks + 20 corresponding RGB pilot images.
    masks=root/'experiments/Session5_1_GT_Pilot20_Final_v1/masks'
    if not masks.is_dir():raise FileNotFoundError(str(masks))
    mask_paths=sorted(masks.glob('*.png'))
    if len(mask_paths)!=20:raise ValueError(f'Expected 20 reviewed GT masks, found {len(mask_paths)}')
    imgs=root/'experiments/session5_1_pilot20_review_v1/images'
    for p in mask_paths:
        add(p.relative_to(root),required=True,label='reviewed_ground_truth')
        add((imgs/(p.stem+'.jpg')).relative_to(root),required=True,label='pilot_rgb_image')
    for rel in ['experiments/Session5_1_GT_Pilot20_Final_v1/gt_manifest.csv',
                'experiments/Session5_1_GT_Pilot20_Final_v1/README.txt']:
        add(Path(rel),label='reviewed_ground_truth')
    for rel in ['submission_preparation/live_leaf_demo_v1.py',
                'submission_preparation/audit_model_artifact_v2.py',
                'submission_preparation/check_model_function_v3.py',
                'submission_preparation/check_analysis_checkpoint_v3_1.py',
                'submission_preparation/audit_experiment_evidence_v4.py',
                'submission_preparation/audit_environment_and_licence_v5.py',
                'demo.py','Readme.md','PROJECT_STRUCTURE.md']:
        add(Path(rel),label='supporting_code_or_doc')
    # A known camera image for offline demo (the 20 pilot images already suffice).
    for d in OPTIONAL_PREFIXES:
        p=root/d
        if p.is_dir():
            for f in all_files(p):
                if f.suffix.lower() in {'.csv','.json','.txt','.md'}:
                    add(f.relative_to(root),label='audit_evidence')
    if include_target:
        t=root/'datasets/Huang_Jing_Bai_Cai'
        for day in [0,3,5,7]:
            folder=t/f'Huang_Jing_Bai_Cai_D{day}'
            if not folder.is_dir():raise FileNotFoundError(str(folder))
            photos=sorted(p for p in folder.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'})
            want=37 if day==0 else 40
            if len(photos)!=want:raise ValueError(f'D{day} expected {want} images, found {len(photos)}')
            for p in photos:add(p.relative_to(root),label='self_collected_target_photo')
    return items


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',default='.',help='Napa_Cabbage_Project root')
    p.add_argument('--output',required=True,help='NEW artifact directory; MUST NOT exist')
    p.add_argument('--include-target-157',action='store_true',help='Include 157 self-collected RGB photos; increases size')
    p.add_argument('--zip',action='store_true',help='Also create a ZIP beside the output folder')
    p.add_argument('--dry-run',action='store_true',help='Verify inputs and show size; do not write')
    a=p.parse_args();root=Path(a.project_root).resolve();out=Path(a.output).resolve()
    if not (root/'E3_Session3/train_baseline.py').is_file():p.error('Project root not detected. Run from Napa_Cabbage_Project.')
    if out.exists():p.error('Refusing to overwrite existing output: '+str(out))
    if out==root or root in out.parents:p.error('Output MUST be outside original project, e.g. ../CS7002SR_Artifact_V6_NEW')
    if root==out or out in root.parents:p.error('Output must not be a parent of the project')
    zip_path=out.with_suffix('.zip')
    if a.zip and zip_path.exists():p.error('Refusing to overwrite ZIP: '+str(zip_path))
    items=source_map(root,a.include_target_157)
    bytes_total=sum(src.stat().st_size for src,_ in items.values())
    print(f'Files selected: {len(items)}; bytes: {bytes_total:,} (~{bytes_total/1048576:.1f} MiB)')
    print('PUBLIC SOURCE DATASET IMAGES / LABELS / ZIP: EXCLUDED pending external licence/provenance confirmation')
    if a.dry_run:return 0
    # Create only AFTER every required path/count is checked.
    out.mkdir(parents=True,exist_ok=False)
    records=[]
    try:
        for rel,(src,role) in sorted(items.items(),key=lambda x:x[0].as_posix()):
            dst=out/rel
            dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(src,dst)
            records.append({'relative_path':rel.as_posix(),'role':role,'bytes':dst.stat().st_size,'sha256':sha256(dst)})
        with (out/'FILE_MANIFEST_V6.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=['relative_path','role','bytes','sha256']);w.writeheader();w.writerows(records)
        summary={'status':'STAGED_NOT_INDEPENDENTLY_VERIFIED', 'files':len(records),
                 'payload_bytes':bytes_total, 'include_target_157':a.include_target_157,
                 'source_dataset_redistribution':'EXCLUDED_PENDING_MANUAL_REVIEW',
                 'validation':'needs V7 clean-folder/clean-environment testing',
                 'source_root_is_not_bundled':True}
        (out/'ARTIFACT_BUILD_SUMMARY_V6.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        (out/'README_START_HERE.md').write_text('''# CS7002SR — Cabbage Leaf Segmentation & Exploratory Visual Indicators

## Scope / scientific boundary

This deliverable provides trained Custom U-Net and ResNet18-U-Net checkpoints, code, evaluated segmentation evidence, a target analysis-only ResNet18 U-Net checkpoint, 20 reviewed target GT masks and matching photos, and a one-photo demo when its script is present. This is **not** a calibrated freshness classifier and makes **no food-safety assessment**. D0, D3, D5 and D7 denote storage days, NOT freshness labels.

## Core evidence

- Source 115-image validation, independently rerun: Custom Dice 0.957581 / IoU 0.918614; ResNet18 Dice 0.948380 / IoU 0.901827.
- Target 20-image zero-shot and five-fold leave-one-head-out (LOHO) evidence in experiments/. LOHO ResNet18 pooled out-of-fold Dice 0.9483 / IoU 0.9017. Analysis-only model was fitted to all 20 GT and must NOT be treated as independent LOHO validation.
- `FILE_MANIFEST_V6.csv` provides SHA-256 for every copied file. Do not interpret this manifest as a proof of model correctness.

## Source public dataset: excluded intentionally

Original public Bok Choy RGB images, annotations, and source-derived binary-mask ZIP are intentionally **NOT copied** here pending final verified source, attribution, licence and third-party rights assessment. To rerun the 115-image source-domain validation, obtain the original dataset via a verified authorized source, generate aligned binary masks with `E3_Session2/convert_polygons.py` using the original documented protocol, then use the original evaluation commands in the project. Source provenance and licence must be documented before redistributing any public-data-derived material. This packaging step neither validates nor grants permission.

## Target-data scope

20 reviewed target GT PNGs and matching RGB photographs are included to examine annotations. Full 157-photo longitudinal set is included ONLY if build used `--include-target-157`. If absent, results for 157 photos are saved analysis evidence, not fully reproducible without the full target images.

## Live demo — from artifact folder (when script present)

Open a Terminal inside this folder (its root). Ensure your Python environment has torch, torchvision, numpy, opencv-python, Pillow. Then run:

```powershell
python.exe submission_preparation\\live_leaf_demo_v1.py --image "experiments\\session5_1_pilot20_review_v1\\images\\D3C01L01正.jpg" --output "demo_check_v7_new"
```

Use a NEW output name on each attempt. The script reads the analysis-only checkpoint and produces a report (`demo_report.html`). It predicts a mask and HSV candidate-colour values; the demo does not calculate Dice for that image unless an evaluation process loads GT.

## Clean-folder test / V7 pending

This folder has been STAGED, not yet independently run. Run a clean-folder load and demo test before claiming a portable submission. `FILE_MANIFEST_V6.csv` files are copied read-only from the original project.

## Submission integrity

The verified Technical Report and 3,000-word LSEP essay are separate deliverables. Cite the dataset and literature in Harvard author-date style and confirm the institute's Canvas file-size and required submission format. Do not claim the package is verified merely because the inventory exists.
''',encoding='utf-8')
        (out/'SOURCE_DATASET_ACCESS_AND_LICENCE_REVIEW_REQUIRED.md').write_text('''# Source dataset access and attribution — pending final review

The project's source public Bok Choy dataset is NOT bundled here. Existing project `datasets/Bok_Choy_Dataset/LICENSE.txt`, `README.md` and `dataset_info.json` suggest licence information, but publication page, attribution, upstream scope and third-party rights require manual verification. Do not assert redistribution is authorized solely on a local LICENSE file. Before release, record original dataset landing-page URL, creators, year, version, licence URL, DOI and changes/derived masks, with an access date. Data availability should distinguish source public data from 157 self-collected target photos.
''',encoding='utf-8')
        if a.zip:
            with zipfile.ZipFile(zip_path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=4,allowZip64=True) as z:
                for f in all_files(out):z.write(f,arcname=out.name+'/'+f.relative_to(out).as_posix())
            print('ZIP: '+str(zip_path))
        print('STAGED: '+str(out)); print('NEXT: run V7 independent execution before submission.')
        return 0
    except Exception:
        print('PARTIAL OUTPUT MAY EXIST at '+str(out)+'; script does not delete it.',file=sys.stderr)
        raise

if __name__=='__main__':raise SystemExit(main())
