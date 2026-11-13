"""Read-only functional checks for the CS7002SR model artifact.

Run from Napa_Cabbage_Project root. All comments are intentionally in English.
This script does not train, change, rewrite, or save any model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_helper(path: Path):
    spec = importlib.util.spec_from_file_location('artifact_zero_shot_helper_v3', str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import helper: {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pick_columns(row, columns):
    return {key: row.get(key) for key in columns if key in row}


def csv_profile(root: Path, relpath: str):
    path = root / relpath
    result = {'path': relpath, 'exists': path.is_file()}
    if not path.is_file():
        return result
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        result['columns'] = list(reader.fieldnames or [])
        rows = list(reader)
    result['rows'] = len(rows)
    return result


def get_model_classes(helper, root):
    v = helper.get_models(root)
    if isinstance(v, (tuple, list)) and len(v) >= 2:
        return v[0], v[1]
    if isinstance(v, dict):
        return v.get('custom') or v.get('custom_unet'), v.get('resnet18') or v.get('resnet18_unet')
    raise TypeError(f'Unexpected get_models return type: {type(v).__name__}')


def model_tensor(model, x):
    y = model(x)
    if isinstance(y, (tuple, list)) and y:
        y = y[0]
    if isinstance(y, dict):
        for name in ('logits', 'out', 'mask', 'output'):
            if name in y:
                y = y[name]
                break
    return y


def check_checkpoint(root, helper, classes, kind, relpath, torch):
    path = root / relpath
    record = {'model': kind, 'checkpoint': relpath, 'status': 'NOT_CHECKED'}
    if not path.is_file():
        record.update(status='FAIL', error='Checkpoint file missing')
        return record
    record['size_bytes'] = path.stat().st_size
    record['sha256'] = sha256_file(path)
    try:
        # Load tensors only; never permit arbitrary Python object unpickling here.
        obj = torch.load(path, map_location='cpu', weights_only=True)
        if not isinstance(obj, dict):
            raise TypeError('Expected a dictionary checkpoint')
        record['checkpoint_keys'] = sorted(str(k) for k in obj.keys())[:40]
        record['weights_only_load'] = 'PASS'
    except Exception as exc:
        record.update(status='FAIL', weights_only_load='FAIL', error=f'{type(exc).__name__}: {exc}')
        return record
    try:
        model = helper.load_model('custom' if kind == 'custom_source' else 'resnet18', path, classes[0], classes[1])
        if isinstance(model, (tuple, list)):
            model = model[0]
        if not hasattr(model, 'eval'):
            raise TypeError('Helper did not return a torch model')
        model.eval()
        torch.manual_seed(2026)
        # A simple deterministic synthetic tensor checks inference, not accuracy.
        x = torch.rand((1, 3, 128, 128), dtype=torch.float32)
        with torch.inference_mode():
            logits = model_tensor(model, x)
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f'Model output is {type(logits).__name__}, not tensor')
        if list(logits.shape) != [1, 1, 128, 128]:
            raise ValueError(f'Expected [1,1,128,128], got {list(logits.shape)}')
        if not bool(torch.isfinite(logits).all().item()):
            raise ValueError('Model output contains NaN or infinite values')
        prob = torch.sigmoid(logits)
        record.update(status='PASS', output_shape=list(logits.shape),
                      probability_min=round(float(prob.min().item()), 6),
                      probability_max=round(float(prob.max().item()), 6),
                      foreground_fraction_at_0_5=round(float((prob >= 0.5).float().mean().item()), 6),
                      parameter_count=sum(p.numel() for p in model.parameters()))
    except Exception as exc:
        record.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
    return record


def main():
    ap = argparse.ArgumentParser(description='Read-only model checkpoint, CPU inference and experiment-file audit')
    ap.add_argument('--project-root', type=Path, default=Path.cwd())
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    root = args.project_root.resolve()
    output = args.output.resolve()
    if output.exists():
        ap.error(f'Output already exists: {output}; choose a NEW directory. Nothing overwritten.')
    if not (root / 'E3_Session5' / 'evaluate_zero_shot.py').is_file():
        ap.error('Run from Napa_Cabbage_Project root, or provide --project-root')
    # Preflight all input paths before creating output.
    import torch
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    helper = load_helper(root / 'E3_Session5' / 'evaluate_zero_shot.py')
    classes = get_model_classes(helper, root)
    if any(c is None for c in classes):
        ap.error('Could not identify Custom U-Net and ResNet18 U-Net model classes')
    checkpoints = [
        ('custom_source', 'experiments/session3_4_baseline_v2/best_unet.pth'),
        ('resnet18_source', 'experiments/session4_4_transfer_v1/best_resnet18_unet.pth'),
        ('resnet18_analysis_in_sample', 'experiments/session6_2_analysis_roi_candidates_v1/resnet18_target_analysis_only.pth'),
    ]
    records = []
    for kind, path in checkpoints:
        print(f'Checking {kind}: {path}', flush=True)
        rec = check_checkpoint(root, helper, classes, kind, path, torch)
        print(f"  {rec['status']}" + (f": {rec.get('error')}" if rec['status'] != 'PASS' else f"; parameters={rec['parameter_count']:,}"), flush=True)
        records.append(rec)
    metrics_paths = [
        'experiments/session3_4_evaluation_v2/per_image_metrics.csv',
        'experiments/session4_4_evaluation_v1/per_image_metrics.csv',
        'experiments/session5_2_zero_shot_v1/per_image_metrics.csv',
        'experiments/session5_3_custom_loho_v1/oof_per_image_metrics.csv',
        'experiments/session5_3_resnet18_loho_v1/oof_per_image_metrics.csv',
        'experiments/session5_3_model_comparison_v1/paired_oof_20_images.csv',
        'experiments/session6_2_analysis_roi_candidates_v1/candidate_roi_manifest_157.csv',
        'experiments/session6_3_visual_features_provisional_v1/visual_features_157_PROVISIONAL.csv',
    ]
    files = [csv_profile(root, p) for p in metrics_paths]
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'project_root': str(root),
        'python': platform.python_version(),
        'torch': str(torch.__version__),
        'cpu_only': True,
        'checkpoint_checks': records,
        'result_csv_profiles': files,
        'interpretation': ('A PASS means safe tensor deserialization and one valid CPU forward pass. '
                           'It does NOT reproduce a published Dice score or constitute independent validation. '
                           'The analysis-only model was trained on the 20 pilot GT images. '
                           'Stored LOHO fold CSVs do not prove saved fold checkpoints exist.'),
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / 'model_function_summary_v3.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    with (output / 'checkpoint_inventory_v3.csv').open('w', encoding='utf-8-sig', newline='') as f:
        fields = ['model','checkpoint','status','size_bytes','sha256','parameter_count','probability_min','probability_max','foreground_fraction_at_0_5','error']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(records)
    with (output / 'result_csv_inventory_v3.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['path','exists','rows','columns'], extrasaction='ignore');writer.writeheader()
        writer.writerows({**x,'columns':' | '.join(x.get('columns',[]))} for x in files)
    passed=sum(r['status']=='PASS' for r in records)
    print(f'COMPLETE: {passed}/{len(records)} checkpoints passed safe load and CPU inference',flush=True)
    print(f'Results: {output}',flush=True)
    print('NOTE: this is a functional smoke check, NOT full metric reproduction.',flush=True)
    if passed != len(records):
        sys.exit(1)

if __name__ == '__main__':
    main()
