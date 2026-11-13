#!/usr/bin/env python3
"""Read-only functional check for the Session 6.2 analysis-only checkpoint.

Do not change any existing project files/checkpoints. All output goes to a NEW folder.
"""
import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for piece in iter(lambda: f.read(1024 * 1024), b''):
            h.update(piece)
    return h.hexdigest()


def type_name(obj):
    return type(obj).__name__


def json_safe(obj, depth=0):
    if depth > 3:
        return '<nested object>'
    if isinstance(obj, (str, bool, int, float, type(None))):
        return obj
    if isinstance(obj, dict):
        return {str(k): json_safe(v, depth + 1) for k, v in obj.items() if k != 'model_state_dict'}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v, depth + 1) for v in obj[:30]]
    return f'<{type_name(obj)}>'


def get_state(checkpoint):
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get('model_state_dict'), dict):
        return checkpoint['model_state_dict']
    if isinstance(checkpoint, dict) and checkpoint and all(hasattr(v, 'shape') for v in checkpoint.values()):
        return checkpoint
    raise ValueError('Checkpoint does not contain a supported model_state_dict')


def import_model_module(project_root):
    file = project_root / 'E3_Session4' / 'models' / 'resnet18_unet.py'
    if not file.is_file():
        raise FileNotFoundError(str(file))
    # Project-root import support. This file is the existing trusted project model definition.
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    spec = importlib.util.spec_from_file_location('artifact_resnet18_unet_v31', file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def model_candidates(module):
    names = ['ResNet18UNet', 'ResNet18Unet', 'ResNetUNet', 'UNetResNet18', 'ResNet18_UNet']
    classes = [(n, c) for n, c in inspect.getmembers(module, inspect.isclass)
               if getattr(c, '__module__', '') == module.__name__]
    classes.sort(key=lambda pair: (0 if pair[0] in names else 1, 0 if 'resnet' in pair[0].lower() else 1, pair[0]))
    return classes


def build_model(cls):
    # The pretrained encoder weights are already in the checkpoint.
    # Never trigger an ImageNet download for a checkpoint loading test.
    sig = inspect.signature(cls)
    kwargs = {}
    for name, param in sig.parameters.items():
        lower = name.lower()
        if lower in ('pretrained', 'pretrained_encoder', 'encoder_pretrained', 'use_pretrained'):
            kwargs[name] = False
        elif lower in ('weights', 'encoder_weights'):
            kwargs[name] = None
        elif lower in ('num_classes', 'n_classes', 'out_channels', 'output_channels'):
            kwargs[name] = 1
        elif lower in ('in_channels', 'input_channels', 'n_channels'):
            kwargs[name] = 3
        elif param.default is inspect.Parameter.empty:
            raise TypeError(f'Unknown required constructor argument: {name}')
    return cls(**kwargs), kwargs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root', default='.', help='Napa_Cabbage_Project root')
    ap.add_argument('--output', required=True, help='New output directory (must not exist)')
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    out = Path(args.output).resolve()
    if out.exists():
        print('STOP: Output already exists; refusing to overwrite:', out)
        return 2
    if not (root / 'E3_Session4').is_dir():
        print('STOP: Run from Napa_Cabbage_Project root or use --project-root.')
        return 2
    import torch
    import numpy as np
    source_file = root / 'experiments/session4_4_transfer_v1/best_resnet18_unet.pth'
    analysis_file = root / 'experiments/session6_2_analysis_roi_candidates_v1/resnet18_target_analysis_only.pth'
    results = {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
               'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'device': 'cpu'},
               'checks': [], 'status': 'FAIL', 'interpretation': 'Smoke test only: not a Dice or IoU reproduction.'}
    try:
        if not source_file.is_file() or not analysis_file.is_file():
            raise FileNotFoundError('One of the two ResNet18 checkpoints is missing')
        src = torch.load(source_file, map_location='cpu', weights_only=True)
        tgt = torch.load(analysis_file, map_location='cpu', weights_only=True)
        ss, ts = get_state(src), get_state(tgt)
        source_keys, target_keys = set(ss), set(ts)
        missing, extra = sorted(source_keys - target_keys), sorted(target_keys - source_keys)
        shape_diff = sorted(k for k in source_keys & target_keys if tuple(ss[k].shape) != tuple(ts[k].shape))
        results['checkpoints'] = {
            'source': {'path': str(source_file.relative_to(root)), 'sha256': sha256(source_file),
                       'size_bytes': source_file.stat().st_size, 'metadata': json_safe(src)},
            'analysis': {'path': str(analysis_file.relative_to(root)), 'sha256': sha256(analysis_file),
                         'size_bytes': analysis_file.stat().st_size, 'metadata': json_safe(tgt)}
        }
        results['state_dict_comparison'] = {'source_key_count': len(source_keys),
             'analysis_key_count': len(target_keys), 'missing_in_analysis': missing[:30],
             'extra_in_analysis': extra[:30], 'shape_mismatch': shape_diff[:30],
             'exact_key_and_shape_match': not (missing or extra or shape_diff)}
        if missing or extra or shape_diff:
            raise ValueError('Source/analysis weight keys or tensor shapes differ; do not force metadata repair')
        results['checks'].append({'name': 'checkpoint_safe_load_and_tensor_schema', 'status': 'PASS'})
        module = import_model_module(root)
        candidates = model_candidates(module)
        if not candidates:
            raise RuntimeError('No model class in E3_Session4/models/resnet18_unet.py')
        trials = []
        model = None
        for name, cls in candidates:
            try:
                candidate, kwargs = build_model(cls)
                candidate.load_state_dict(ts, strict=True)
                model = candidate
                results['model_class'] = name
                results['constructor_kwargs'] = kwargs
                break
            except Exception as exc:
                trials.append({'class': name, 'error': f'{type(exc).__name__}: {str(exc)[:220]}'})
        if model is None:
            results['model_constructor_trials'] = trials
            raise RuntimeError('No available class matched the checkpoint tensors exactly')
        results['checks'].append({'name': 'analysis_checkpoint_strict_load_into_project_architecture', 'status': 'PASS'})
        model.eval().cpu()
        # Deterministic synthetic RGB input; not an evaluation image.
        x = torch.linspace(0, 1, steps=3*128*128, dtype=torch.float32).reshape(1, 3, 128, 128)
        with torch.inference_mode():
            raw = model(x)
        if isinstance(raw, (tuple, list)):
            raw = raw[0]
        if isinstance(raw, dict):
            raw = raw.get('out', next(iter(raw.values())))
        if not isinstance(raw, torch.Tensor):
            raise TypeError('Model output is not a tensor')
        if tuple(raw.shape) != (1, 1, 128, 128):
            raise ValueError(f'Unexpected output shape: {tuple(raw.shape)}')
        if not bool(torch.isfinite(raw).all()):
            raise ValueError('Model output has NaN or infinity')
        # Do not assume whether existing model returns logits or probabilities.
        results['inference'] = {'shape': list(raw.shape), 'raw_min': float(raw.min()),
             'raw_max': float(raw.max()), 'raw_mean': float(raw.mean()),
             'all_values_between_0_and_1': bool(((raw >= 0) & (raw <= 1)).all()),
             'parameter_count': sum(p.numel() for p in model.parameters())}
        results['checks'].append({'name': 'deterministic_cpu_forward_finite_output', 'status': 'PASS'})
        results['status'] = 'PASS'
    except Exception as exc:
        results['error'] = f'{type(exc).__name__}: {exc}'
        results['traceback_tail'] = traceback.format_exc(limit=4)
    out.mkdir(parents=True, exist_ok=False)
    (out/'analysis_checkpoint_check_v3_1.json').write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    md = [f"# Analysis-only ResNet18 Checkpoint Functional Check (V3.1)", '',
          f"**Status:** {results['status']}", '',
          'This is a read-only structural and CPU functional test, not independent validation.', '',
          'Existing checkpoints were never modified.', '',
          '## Checks', '']
    for check in results['checks']:
        md.append(f"- {check['name']}: {check['status']}")
    if results.get('error'):
        md.extend(['', '## Error', '', results['error']])
    if results.get('state_dict_comparison'):
        md.extend(['', '## Tensor schema', '', '```json',
                   json.dumps(results['state_dict_comparison'], ensure_ascii=False, indent=2), '```'])
    if results.get('inference'):
        md.extend(['', '## CPU inference', '', '```json',
                   json.dumps(results['inference'], ensure_ascii=False, indent=2), '```'])
    (out/'analysis_checkpoint_check_v3_1.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print('ANALYSIS CHECKPOINT V3.1:', results['status'])
    print('Source and analysis checkpoint tensor schema:',
          results.get('state_dict_comparison', {}).get('exact_key_and_shape_match', 'not checked'))
    if results.get('model_class'):
        print('Architecture class:', results['model_class'])
    if results.get('inference'):
        print('CPU forward shape:', results['inference']['shape'])
        print('Parameter count:', results['inference']['parameter_count'])
    if results.get('error'):
        print('ERROR:', results['error'])
    print('Output:', out)
    return 0 if results['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
