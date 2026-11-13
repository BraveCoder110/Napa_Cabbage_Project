#!/usr/bin/env python3
"""Read-only environment, source-license and submission-dependency audit (V5)."""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import re
import sys

PACKAGE_MAP = {
    'torch': 'torch', 'torchvision': 'torchvision', 'numpy': 'numpy',
    'PIL': 'Pillow', 'cv2': 'opencv-python', 'pandas': 'pandas',
    'matplotlib': 'matplotlib', 'sklearn': 'scikit-learn',
    'scipy': 'scipy', 'tqdm': 'tqdm', 'labelme': 'labelme',
    'albumentations': 'albumentations', 'segmentation_models_pytorch': 'segmentation-models-pytorch',
    'yaml': 'PyYAML', 'skimage': 'scikit-image', 'seaborn': 'seaborn',
    'requests': 'requests', 'timm': 'timm',
}
CODE_DIRS = ('E3_Session2', 'E3_Session3', 'E3_Session4', 'E3_Session5', 'E3_Session6')
REQUIRED = [
 ('Custom U-Net model', 'E3_Session3/models/unet.py'),
 ('ResNet18 U-Net model', 'E3_Session4/models/resnet18_unet.py'),
 ('Custom source checkpoint', 'experiments/session3_4_baseline_v2/best_unet.pth'),
 ('ResNet18 source checkpoint', 'experiments/session4_4_transfer_v1/best_resnet18_unet.pth'),
 ('Analysis-only checkpoint', 'experiments/session6_2_analysis_roi_candidates_v1/resnet18_target_analysis_only.pth'),
 ('Source license', 'datasets/Bok_Choy_Dataset/LICENSE.txt'),
 ('Source provenance', 'datasets/Bok_Choy_Dataset/dataset_info.json'),
 ('Source README', 'datasets/Bok_Choy_Dataset/README.md'),
 ('Source hashes', 'datasets/Bok_Choy_Dataset/checksums.sha256'),
 ('20 reviewed masks', 'experiments/Session5_1_GT_Pilot20_Final_v1/masks'),
 ('LOHO ResNet evidence', 'experiments/session5_3_resnet18_loho_v1/oof_per_image_metrics.csv'),
 ('V4 audit evidence', 'submission_preparation/evidence_audit_results_v4/evidence_summary_v4.json'),
]

def sha256(path: Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def safe_text(path, limit=24000):
    if not path.is_file(): return None
    s=path.read_text(encoding='utf-8-sig',errors='replace')
    return s[:limit] + ('\n[TRUNCATED: inspect original file]' if len(s)>limit else '')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',default='.',help='Napa_Cabbage_Project root (default: cwd)')
    ap.add_argument('--output',required=True,help='NEW output directory; must not exist')
    args=ap.parse_args()
    root=Path(args.project_root).resolve()
    out=Path(args.output).resolve()
    if not root.is_dir(): ap.error('Project root not found.')
    if out.exists(): ap.error('Output already exists; choose a NEW directory to avoid overwrite.')
    if root == out: ap.error('Output cannot be project root.')
    if not (root/'E3_Session5').is_dir(): ap.error('Expected E3_Session5 not found in root.')
    out.mkdir(parents=True,exist_ok=False)

    checked=[]
    for label,rel in REQUIRED:
        p=root/rel
        checked.append({'item':label,'relative_path':rel,'exists':p.exists(),'type':'directory' if p.is_dir() else ('file' if p.is_file() else 'missing'), 'bytes':p.stat().st_size if p.is_file() else None})

    imports={}
    parse_errors=[]
    for d in CODE_DIRS:
        directory=root/d
        if not directory.is_dir():continue
        for p in directory.rglob('*.py'):
            rel=p.relative_to(root).as_posix()
            try: t=ast.parse(p.read_text(encoding='utf-8-sig',errors='replace'),filename=rel)
            except (SyntaxError,UnicodeError) as e:
                parse_errors.append({'file':rel,'error':str(e)});continue
            for node in ast.walk(t):
                if isinstance(node,ast.Import): names=[a.name.split('.')[0] for a in node.names]
                elif isinstance(node,ast.ImportFrom) and node.level==0: names=[(node.module or '').split('.')[0]]
                else:continue
                for name in names:
                    if name in PACKAGE_MAP: imports.setdefault(name,set()).add(rel)
    packages=[]
    for imp,paths in sorted(imports.items()):
        dist=PACKAGE_MAP[imp]
        try: version=md.version(dist); status='INSTALLED'
        except md.PackageNotFoundError: version='';status='NOT_FOUND'
        packages.append({'import_name':imp,'distribution':dist,'installed_version':version,'status':status,'example_script':sorted(paths)[0],'script_count':len(paths)})
    with (out/'dependency_inventory_v5.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=['import_name','distribution','installed_version','status','example_script','script_count']);w.writeheader();w.writerows(packages)
    with (out/'required_file_inventory_v5.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=['item','relative_path','exists','type','bytes']);w.writeheader();w.writerows(checked)
    (out/'requirements_environment_snapshot_v5.txt').write_text(
        '# Environment snapshot, not a tested clean-install requirements lock\n' +
        '# Generated from packages imported by E3_Session2 ... E3_Session6\n' +
        '\n'.join(f"{p['distribution']}=={p['installed_version']}" for p in packages if p['status']=='INSTALLED')+'\n',encoding='utf-8')
    lic=root/'datasets/Bok_Choy_Dataset/LICENSE.txt'
    info=root/'datasets/Bok_Choy_Dataset/dataset_info.json'
    readme=root/'datasets/Bok_Choy_Dataset/README.md'
    license_text=safe_text(lic)
    readme_text=safe_text(readme,6000)
    info_text=safe_text(info,12000)
    parsed_info=None
    if info_text is not None:
        try: parsed_info=json.loads(info_text)
        except Exception: parsed_info=None
    evidence={
      'scope':'Read-only audit. NO legal conclusions or source license approval.',
      'source_licence_file_present':lic.is_file(),
      'source_licence_sha256':sha256(lic) if lic.is_file() else None,
      'source_licence_text':license_text,
      'source_dataset_info_json_text':info_text,
      'source_dataset_info_json_parsed':parsed_info,
      'source_readme_excerpt':readme_text,
      'questions_to_resolve':[
         'Identify original source dataset publisher and original public URL.',
         'Check whether the license actually applies to all 575 photos and annotations.',
         'Check whether the license permits redistribution, or requires link-only access.',
         'Check attribution wording and whether edits must be identified.',
         'Check how licence conditions relate to submitting dataset copies to Canvas.'
      ]
    }
    (out/'source_license_review_v5.json').write_text(json.dumps(evidence,indent=2,ensure_ascii=False),encoding='utf-8')
    report={
      'audit_version':'v5','read_only_project_inputs':True,
      'project_root_relative':'Napa_Cabbage_Project',
      'system':{'python_version':sys.version.split()[0], 'python_implementation':platform.python_implementation(),'platform':platform.platform(),'machine':platform.machine()},
      'required_files_found':sum(x['exists'] for x in checked),
      'required_files_total':len(checked),
      'packages_detected':len(packages),
      'packages_not_found':[x['distribution'] for x in packages if x['status']=='NOT_FOUND'],
      'parse_errors':parse_errors,
      'licence_decision':'PENDING_MANUAL_REVIEW',
      'source_validation_reinference':'NOT_PERFORMED',
      'model_retraining':'NOT_PERFORMED',
      'notes':[
        'Do not release source images before checking original author, licence and attribution.',
        'Do not confuse source validation rerun with LOHO results reproduction.',
        'Python package list is a machine snapshot, not a portable guarantee.',
        'Do not include unrelated CIFAR-10 data in final cabbage artifact.'
      ]
    }
    (out/'environment_and_license_summary_v5.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    md_lines=['# AI Model Artifact V5 — Environment and Source Licence Audit','',
      f"- Required paths: **{report['required_files_found']}/{report['required_files_total']}** present",
      f"- Imported third-party packages detected: **{len(packages)}**",
      f"- Packages not found in current environment: **{', '.join(report['packages_not_found']) or 'None'}**",
      '- Source dataset licence decision: **PENDING MANUAL REVIEW**',
      '- Source validation full inference rerun: **NOT PERFORMED**','',
      '## Follow-up','1. Send `source_license_review_v5.json` to verify the exact publisher, licence and permitted uses.',
      '2. Send `dependency_inventory_v5.csv` to check dependency versions and any missing packages.',
      '3. Keep all original source code, checkpoint and dataset files unchanged.',
      '4. Re-run source validation independently in a later step if TP/FP/FN verification is needed.','']
    (out/'environment_and_license_summary_v5.md').write_text('\n'.join(md_lines),encoding='utf-8')
    print('AI MODEL ARTIFACT V5: COMPLETE')
    print(f"Paths present: {report['required_files_found']}/{report['required_files_total']}")
    print(f"Third-party imports: {len(packages)}; missing distributions: {len(report['packages_not_found'])}")
    print('Source licence: PENDING MANUAL REVIEW (no redistribution approval inferred)')
    print('Output:',out)

if __name__=='__main__':main()
