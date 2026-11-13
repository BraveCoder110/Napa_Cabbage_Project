#!/usr/bin/env python3
"""V7 independent ZIP/extraction and CPU demo audit. Read-only ZIP; NEW output only.

Python 3.10+ stdlib for integrity; installed torch/torchvision/numpy/opencv/Pillow
required only when --run-demo used. No model training or internet access.
"""
from __future__ import annotations
import argparse,csv,hashlib,io,json,os,shutil,subprocess,sys,zipfile
from pathlib import Path, PurePosixPath

REQUIRED={
 'E3_Session3/models/unet.py','E3_Session4/models/resnet18_unet.py',
 'experiments/session3_4_baseline_v2/best_unet.pth',
 'experiments/session4_4_transfer_v1/best_resnet18_unet.pth',
 'experiments/session6_2_analysis_roi_candidates_v1/resnet18_target_analysis_only.pth',
 'submission_preparation/live_leaf_demo_v1.py',
 'experiments/session5_1_pilot20_review_v1/images/D3C01L01正.jpg',
 'README_START_HERE.md','SOURCE_DATASET_ACCESS_AND_LICENCE_REVIEW_REQUIRED.md',
 'FILE_MANIFEST_V6.csv','ARTIFACT_BUILD_SUMMARY_V6.json',
 'experiments/v5_1_source_custom_recheck_new/per_image_metrics.csv',
 'experiments/v5_1_source_resnet18_recheck_new/per_image_metrics.csv',
}

def sha256_file(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--zip',required=True,help='V6 ZIP file')
 ap.add_argument('--output',required=True,help='New directory for V7 logs and isolated extraction')
 ap.add_argument('--run-demo',action='store_true',help='Run CPU demo from extracted artifact using included existing photo')
 ap.add_argument('--timeout-seconds',type=int,default=240)
 a=ap.parse_args();zp=Path(a.zip).resolve(); out=Path(a.output).resolve()
 if not zp.is_file():ap.error('ZIP does not exist: '+str(zp))
 if out.exists():ap.error('Output already exists; choose a NEW folder: '+str(out))
 if a.timeout_seconds<10:ap.error('--timeout-seconds must be >=10')
 if out==zp.parent or out in zp.parents:ap.error('Invalid output path')
 out.mkdir(parents=True)
 checks=[]
 def ck(name,ok,detail):
  checks.append({'check':name,'status':'PASS' if ok else 'FAIL','detail':str(detail)})
  print(('PASS' if ok else 'FAIL')+' '+name+': '+str(detail),flush=True)
 try:
  with zipfile.ZipFile(zp) as z:
   members=[x for x in z.infolist() if not x.is_dir()]
   names=[x.filename for x in members]
   prefix=names[0].split('/')[0]+'/' if names else ''
   # Defense against archive traversal, absolute paths and symlinks
   safe=bool(prefix) and all(x.filename.startswith(prefix) and not PurePosixPath(x.filename).is_absolute() and not any(p in ('..','') for p in PurePosixPath(x.filename).parts) and ((x.external_attr>>16)&0o170000)!=0o120000 for x in members)
   ck('zip_paths_safe',safe,f'{len(members)} regular entries, root={prefix}')
   if not safe:raise RuntimeError('Unsafe ZIP entry; extraction cancelled')
   ck('zip_duplicate_entries',len(names)==len(set(names)),f'{len(names)} entries')
   bad=z.testzip();ck('zip_crc',bad is None,bad or 'all entries pass CRC')
   root=out/'isolated'/prefix.rstrip('/')
   root.mkdir(parents=True)
   for x in members:
    relative=PurePosixPath(x.filename[len(prefix):]);target=root.joinpath(*relative.parts)
    target.parent.mkdir(parents=True,exist_ok=True)
    with z.open(x) as src,target.open('xb') as dst:shutil.copyfileobj(src,dst,length=1024*1024)
   ck('isolated_extraction',len([p for p in root.rglob('*') if p.is_file()])==len(members),root)
  ck('required_paths',all((root/p).is_file() for p in REQUIRED),f'{sum((root/p).is_file() for p in REQUIRED)}/{len(REQUIRED)}')
  rows=list(csv.DictReader((root/'FILE_MANIFEST_V6.csv').open('r',newline='',encoding='utf-8-sig')))
  bad=[];bytes_sum=0
  for r in rows:
   rp=r['relative_path'];path=root/rp
   if not path.is_file():bad.append((rp,'missing'));continue
   size=path.stat().st_size;bytes_sum+=size
   if size!=int(r['bytes']) or sha256_file(path).lower()!=r['sha256'].lower():bad.append((rp,'hash/size mismatch'))
  ck('manifest_sha256',not bad,f'{len(rows)} tracked, {len(bad)} mismatch, payload={bytes_sum} bytes')
  ck('manifest_unique',len(rows)==len({r['relative_path'] for r in rows}),str(len(rows))+' rows')
  meta=json.loads((root/'ARTIFACT_BUILD_SUMMARY_V6.json').read_text(encoding='utf-8'))
  ck('build_summary',int(meta.get('files',-1))==len(rows) and int(meta.get('payload_bytes',-1))==bytes_sum,meta)
  included={x.relative_to(root).as_posix() for x in root.rglob('*') if x.is_file()}
  additional=included-{r['relative_path'] for r in rows}
  expected_add={'ARTIFACT_BUILD_SUMMARY_V6.json','FILE_MANIFEST_V6.csv','README_START_HERE.md','SOURCE_DATASET_ACCESS_AND_LICENCE_REVIEW_REQUIRED.md'}
  ck('untracked_files',additional==expected_add,'extra files: '+', '.join(sorted(additional)))
  forbidden=[f for f in included if f.startswith('datasets/Bok_Choy_Dataset/') or f in ('E3_Session2/Bok_Choy_Dataset.zip','E3_output/binary_masks.zip')]
  ck('source_dataset_exclusion',not forbidden,forbidden or 'no original public source images/labels/derived mask ZIP')
  for name in ('custom','resnet18'):
   f=root/f'experiments/v5_1_source_{name}_recheck_new/per_image_metrics.csv'
   if f.is_file():
    with f.open('r',newline='',encoding='utf-8-sig') as handle:count=sum(1 for _ in csv.DictReader(handle))
    ck('source_validation_'+name,count==115,str(count)+' rows')
  gt=list((root/'experiments/Session5_1_GT_Pilot20_Final_v1/masks').glob('*.png'))
  ck('reviewed_gt_masks',len(gt)==20,str(len(gt))+' PNG masks')
  image_files=list((root/'experiments/session5_1_pilot20_review_v1/images').glob('*.jpg'))
  ck('reviewed_photo_subset',len(image_files)==20,str(len(image_files))+' JPG photos')
  if a.run_demo and not any(x['status']=='FAIL' for x in checks):
   cmd=[sys.executable,'submission_preparation/live_leaf_demo_v1.py','--image','experiments/session5_1_pilot20_review_v1/images/D3C01L01正.jpg','--output','v7_demo_result_NEW']
   try:
    res=subprocess.run(cmd,cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=a.timeout_seconds)
    (out/'demo_console_v7.txt').write_text(res.stdout,encoding='utf-8')
    generated=root/'v7_demo_result_NEW'
    required=['demo_report.html','demo_metadata.json','feature_summary.csv','04_binary_mask.png','05_segmentation_overlay.png','07_colour_candidates.png']
    good=res.returncode==0 and all((generated/x).is_file() for x in required)
    ck('isolated_cpu_demo',good,f'returncode={res.returncode}; outputs={sum((generated/x).is_file() for x in required)}/{len(required)}; see demo_console_v7.txt')
    if good:
     rec=json.loads((generated/'demo_metadata.json').read_text(encoding='utf-8'))
     fv=rec.get('feature_metrics',{})
     ck('demo_metrics_finite',0<=float(fv.get('roi_fraction',-1))<=1 and all(v is None or 0<=float(v)<=1 for k,v in fv.items() if k in ('green_ratio','yellow_candidate_ratio','brown_candidate_ratio')),str({k:fv.get(k) for k in ('roi_fraction','green_ratio','yellow_candidate_ratio','brown_candidate_ratio')}))
   except subprocess.TimeoutExpired as e:
    ck('isolated_cpu_demo',False,f'timeout after {a.timeout_seconds} seconds')
  else:
   ck('isolated_cpu_demo_skipped',True,'--run-demo omitted (not a functional pass)')
 except Exception as e:
  ck('unexpected_exception',False,type(e).__name__+': '+str(e))
 summary={'result':'PASS' if all(c['status']=='PASS' for c in checks) else 'FAIL','zip':str(zp),'isolated_root':str(out/'isolated'),'checks':checks,'demo_executed':bool(a.run_demo),'note':'This proves packaged-file integrity and local isolated-folder execution, NOT a fresh environment installation, food safety, or calibrated freshness grading.'}
 (out/'v7_verification_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
 with (out/'v7_verification_checks.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=['check','status','detail']);w.writeheader();w.writerows(checks)
 print('V7 COMPLETE:',summary['result'],'checks',len(checks),'reports:',out,flush=True)
 return 0 if summary['result']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
