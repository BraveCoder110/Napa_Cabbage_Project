#!/usr/bin/env python3
"""V4 evidence consistency audit. Read-only inputs, new output only.

Run from Napa_Cabbage_Project root. Python standard library only.
This script recalculates global Dice/IoU from stored TP/FP/FN counts.
It does not retrain networks or claim independent reproduction of inference.
"""
import argparse
import csv
import json
import math
from pathlib import Path

DAYS = {0, 3, 5, 7}
HEADS = {f"C{i:02d}" for i in range(1, 6)}


def csv_rows(path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def as_int(value):
    return int(value)


def dice_iou(rows):
    tp = sum(as_int(r['tp']) for r in rows)
    fp = sum(as_int(r['fp']) for r in rows)
    fn = sum(as_int(r['fn']) for r in rows)
    dice = (2 * tp) / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    return {'tp': tp, 'fp': fp, 'fn': fn, 'global_dice': dice, 'global_iou': iou}


def close(a, b, tol=2e-6):
    return abs(float(a) - float(b)) <= tol


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='.', help='Project root containing experiments/')
    p.add_argument('--output', required=True, help='New output directory; refuses reuse')
    a = p.parse_args()
    root = Path(a.root).resolve()
    output = Path(a.output).resolve()
    if output.exists():
        p.error('Output already exists; select a NEW directory (no overwrite).')
    output.mkdir(parents=True)
    checks = []
    metrics = []

    def record(category, check, status, evidence, details=''):
        checks.append({'category':category,'check':check,'status':status,'evidence':str(evidence),'details':str(details)})
        print(f"{status:<5} {category}: {check}" + (f" | {details}" if status != 'PASS' and details else ''))

    def load_csv(category, label, rel):
        path = root / rel
        if not path.is_file():
            record(category, label, 'FAIL', rel, 'Required file missing')
            return None
        try:
            rows = csv_rows(path)
        except Exception as exc:
            record(category, label, 'FAIL', rel, f'CSV parse: {type(exc).__name__}: {exc}')
            return None
        return rows

    # Verify global counts using raw per-image confusion counts, not mean per-image Dice.
    zero = load_csv('zero_shot','per-image evaluation evidence','experiments/session5_2_zero_shot_v1/per_image_metrics.csv')
    zg = load_csv('zero_shot','grouped evaluation summary','experiments/session5_2_zero_shot_v1/group_metrics.csv')
    expected_zero = {'custom': 0.57088603, 'resnet18': 0.5868, 'all_foreground': 0.5692}
    if zero is not None:
        for model in ['custom','resnet18']:
            part = [r for r in zero if r.get('model') == model]
            names = [r.get('image') for r in part]
            ok = len(part)==20 and len(set(names))==20 and set(r.get('head') for r in part)==HEADS and set(int(r.get('day','-1')) for r in part)==DAYS
            record('zero_shot',f'{model}: 20 unique images / 5 heads / 4 days','PASS' if ok else 'FAIL','per_image_metrics.csv',f'{len(part)} rows')
            if not part: continue
            try:
                res=dice_iou(part)
                metrics.append({'experiment':'zero_shot','model':model,**res})
                orig=[r for r in (zg or []) if r.get('model')==model and r.get('group')=='ALL']
                summary_ok=len(orig)==1 and close(res['global_dice'],orig[0]['global_dice']) and close(res['global_iou'],orig[0]['global_iou'])
                record('zero_shot',f'{model}: global Dice/IoU recalculated','PASS' if summary_ok else 'FAIL','per_image_metrics.csv / group_metrics.csv',f"Dice={res['global_dice']:.8f}; IoU={res['global_iou']:.8f}")
                expected_ok=abs(res['global_dice']-expected_zero[model])<0.0001
                record('zero_shot',f'{model}: Technical Report reported Dice','PASS' if expected_ok else 'FAIL','report claim',f"computed {res['global_dice']:.8f}")
            except (KeyError,ValueError,ZeroDivisionError) as exc:
                record('zero_shot',f'{model}: counts','FAIL','per_image_metrics.csv',repr(exc))

    for model, target in [('custom',0.66453142),('resnet18',0.94829793)]:
        cat='loho_'+model
        folder=f'experiments/session5_3_{model}_loho_v1'
        per=load_csv(cat,'out-of-fold image evidence',folder+'/oof_per_image_metrics.csv')
        groups=load_csv(cat,'group summary',folder+'/oof_group_metrics.csv')
        folds=load_csv(cat,'fold membership',folder+'/fold_membership.csv')
        protocol_path=root/folder/'protocol_summary.json'
        if protocol_path.is_file():
            try:
                protocol=read_json(protocol_path)
                record(cat,'protocol JSON readable','PASS',str(protocol_path.relative_to(root)),f"epochs={protocol.get('epochs')}; lr={protocol.get('lr')}")
            except Exception as exc:
                record(cat,'protocol JSON readable','FAIL',str(protocol_path),repr(exc))
        else:
            record(cat,'protocol JSON readable','FAIL',str(protocol_path),'not found')
        if per is not None:
            try:
                heads={r['head'] for r in per}
                ids=[r['image'] for r in per]
                ok=len(per)==20 and len(set(ids))==20 and heads==HEADS and {int(r['day']) for r in per}==DAYS
                record(cat,'20 unique held-out image results','PASS' if ok else 'FAIL','oof_per_image_metrics.csv',f'{len(per)} rows')
                heldout_ok=all(r['head']==r['held_out_head'] for r in per)
                record(cat,'per-image held-out head equals image head','PASS' if heldout_ok else 'FAIL','oof_per_image_metrics.csv','')
                res=dice_iou(per)
                metrics.append({'experiment':'LOHO_5_fold','model':model,**res})
                orig=[r for r in (groups or []) if r.get('group')=='ALL']
                ok=len(orig)==1 and close(res['global_dice'],orig[0]['global_dice']) and close(res['global_iou'],orig[0]['global_iou'])
                record(cat,'pooled TP/FP/FN reproduce ALL metrics','PASS' if ok else 'FAIL','oof_per_image_metrics.csv + oof_group_metrics.csv',f"Dice={res['global_dice']:.8f}; IoU={res['global_iou']:.8f}")
                record(cat,'Technical Report Dice consistency','PASS' if close(res['global_dice'],target) else 'FAIL','report claim',f"computed {res['global_dice']:.8f}")
            except (KeyError,ValueError) as exc:
                record(cat,'out-of-fold result schema','FAIL','oof_per_image_metrics.csv',repr(exc))
        if folds is not None:
            try:
                good=True; reasons=[]
                if len(folds)!=100: good=False; reasons.append('expected 100 entries')
                for fold in sorted(HEADS):
                    part=[r for r in folds if r['fold']==fold]
                    train=[r for r in part if r['role']=='train']
                    test=[r for r in part if r['role']=='held_out_test']
                    if len(part)!=20 or len(train)!=16 or len(test)!=4:
                        good=False; reasons.append(f'{fold}: expected 16 train, 4 held out')
                    if any(r['head']==fold for r in train) or any(r['head']!=fold for r in test):
                        good=False; reasons.append(f'{fold}: head leakage')
                    if len({r['image'] for r in part})!=20:
                        good=False; reasons.append(f'{fold}: duplicated/missing images')
                record(cat,'five folds / 16 train + 4 held-out per fold / no head leakage','PASS' if good else 'FAIL','fold_membership.csv','; '.join(reasons))
            except (KeyError,ValueError) as exc:
                record(cat,'LOHO membership schema','FAIL','fold_membership.csv',repr(exc))

    # Longitudinal dataset: image counts, 40 tracked head/leaf/side series, 111 D0 pairs.
    folder='experiments/session6_1_longitudinal_audit_v1'
    obs=load_csv('longitudinal','observations',folder+'/observation_manifest_157.csv')
    pairs=load_csv('longitudinal','D0 followup matches',folder+'/D0_followup_pairs_111.csv')
    series=load_csv('longitudinal','tracked leaf-side series',folder+'/longitudinal_series_40.csv')
    missing=load_csv('longitudinal','missing observations',folder+'/missing_observations.csv')
    if obs is not None:
        ok=len(obs)==157 and {r['head_id'] for r in obs}==HEADS and {r['day'] for r in obs}=={'D0','D3','D5','D7'}
        record('longitudinal','157 images / five biological heads','PASS' if ok else 'FAIL','observation_manifest_157.csv',f'rows={len(obs)}')
        day_counts={d:sum(r['day']==d for r in obs) for d in ['D0','D3','D5','D7']}
        record('longitudinal','day counts D0=37, D3/5/7=40','PASS' if day_counts=={'D0':37,'D3':40,'D5':40,'D7':40} else 'FAIL','observation_manifest_157.csv',str(day_counts))
    if series is not None:
        completed=sum(str(r.get('complete_4_timepoints')) in ('1','True','true') for r in series)
        record('longitudinal','40 leaf-side series; 37 complete','PASS' if len(series)==40 and completed==37 else 'FAIL','longitudinal_series_40.csv',f'rows={len(series)}, complete={completed}')
    if pairs is not None:
        record('longitudinal','111 D0 follow-up pairs','PASS' if len(pairs)==111 else 'FAIL','D0_followup_pairs_111.csv',f'rows={len(pairs)}')
    if missing is not None:
        expected={('C04','L03','reverse','D0'),('C05','L03','reverse','D0'),('C05','L04','reverse','D0')}
        got={(r['head_id'],r['leaf_id'],r['side'],r['day']) for r in missing}
        record('longitudinal','three specific D0 reverse photos absent','PASS' if got==expected else 'FAIL','missing_observations.csv',str(got))

    # ROI diagnostic is in-sample; explicitly document this distinction.
    d='experiments/session6_4_gt_auto_sensitivity_v2'
    sens=load_csv('roi_sensitivity','manual GT versus analysis ROI',d+'/per_image_GT_vs_Auto_20.csv')
    summary_path=root/d/'diagnosis_summary.json'
    if sens is not None:
        ok=len(sens)==20 and {r['source_head'] for r in sens}==HEADS
        record('roi_sensitivity','20 in-sample GT/auto comparisons','PASS' if ok else 'FAIL','per_image_GT_vs_Auto_20.csv',f'rows={len(sens)}')
        try:
            mean=sum(float(r['mask_dice']) for r in sens)/len(sens)
            record('roi_sensitivity','mean in-sample mask Dice approximately 0.9576','PASS' if abs(mean-0.9576229564)<0.0001 else 'FAIL','per_image_GT_vs_Auto_20.csv',f'mean={mean:.9f}')
        except (ValueError,KeyError,ZeroDivisionError) as exc:
            record('roi_sensitivity','mean mask Dice calculation','FAIL','per_image_GT_vs_Auto_20.csv',repr(exc))
    if summary_path.is_file():
        try:
            summary=read_json(summary_path)
            record('roi_sensitivity','diagnosis explicitly not freshness validation','PASS' if 'NOT_FRESHNESS_VALIDATION' in str(summary.get('status','')) else 'WARN',str(summary_path.relative_to(root)),str(summary.get('status','')))
        except Exception as exc:
            record('roi_sensitivity','diagnosis JSON','FAIL',str(summary_path),repr(exc))

    # Additional evidence: not all automatic ROIs have reviewed ground truth.
    features=load_csv('visual_features','157 provisional feature records','experiments/session6_3_visual_features_provisional_v1/visual_features_157_PROVISIONAL.csv')
    if features is not None:
        record('visual_features','157 provisional features','PASS' if len(features)==157 else 'FAIL','visual_features_157_PROVISIONAL.csv',f'rows={len(features)}')
    photo=load_csv('colour_diagnostic','20 manual-GT leaf/wood checks','experiments/session6_4_2_photo_color_reproduction_v1/pilot20_leaf_wood_color.csv')
    if photo is not None:
        record('colour_diagnostic','20 samples in colour diagnostic','PASS' if len(photo)==20 else 'FAIL','pilot20_leaf_wood_color.csv',f'rows={len(photo)}')

    # Source-domain validation: source checkpoints already passed V3 smoke test;
    # no rerun of inference here. Track presence and defer source metric recalculation.
    for model,rel in [('custom','experiments/session3_4_evaluation_v2/per_image_metrics.csv'),('resnet18','experiments/session4_4_evaluation_v1/per_image_metrics.csv')]:
        rows=load_csv('source_validation',f'{model} source per-image evaluation',rel)
        if rows is not None:
            record('source_validation',f'{model} validation evaluation file has 115 images','PASS' if len(rows)==115 else 'FAIL',rel,f'rows={len(rows)}')
            # These may have different metric columns; without pixel counts do not recalculate pooled Dice.
            if rows and all(k in rows[0] for k in ('tp','fp','fn')):
                try:
                    res=dice_iou(rows)
                    metrics.append({'experiment':'source_validation','model':model,**res})
                    expected=0.9576 if model=='custom' else 0.9484
                    record('source_validation',f'{model} pooled source Dice','PASS' if abs(res['global_dice']-expected)<0.0001 else 'FAIL',rel,f"Dice={res['global_dice']:.8f}")
                except Exception as exc:
                    record('source_validation',f'{model} source counts','FAIL',rel,repr(exc))
            else:
                record('source_validation',f'{model} source metric reproduction','WARN',rel,'No TP/FP/FN in per-image table; needs original evaluation run for full verification')

    # License: file existence is not permission to redistribute.
    licence=root/'datasets/Bok_Choy_Dataset/LICENSE.txt'
    record('dataset_licence','source license review','WARN' if licence.exists() else 'FAIL','datasets/Bok_Choy_Dataset/LICENSE.txt','Manual review of attribution and redistribution is required')

    def write_csv(name, data, fields):
        with (output/name).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
            writer.writeheader(); writer.writerows(data)

    write_csv('evidence_checks_v4.csv',checks,['category','check','status','evidence','details'])
    write_csv('recomputed_metrics_v4.csv',metrics,['experiment','model','tp','fp','fn','global_dice','global_iou'])
    statuses={k:sum(r['status']==k for r in checks) for k in ['PASS','WARN','FAIL']}
    report={'audit_version':'v4','read_only_inputs':True,'retraining_performed':False,
            'metrics_from_stored_confusion_counts':True,'checks':statuses,
            'important_scope':'Consistency audit of saved outputs; not inference rerun or biological freshness validation',
            'result_checks':checks,'recomputed_metrics':metrics}
    (output/'evidence_summary_v4.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    md=['# V4 Experiment Evidence Audit','',
        f"PASS: {statuses['PASS']} | WARN: {statuses['WARN']} | FAIL: {statuses['FAIL']}",'',
        'This audit recomputes pooled Dice/IoU from stored TP/FP/FN. It does not rerun model prediction.',
        'Source dataset license and full independent run still need separate review.','',
        '| Category | Check | Status | Details |','|---|---|---|---|']
    for r in checks:
        md.append('| '+ ' | '.join(str(r[k]).replace('|','/').replace('\n',' ') for k in ['category','check','status','details']) + ' |')
    (output/'evidence_summary_v4.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(f"\nV4 COMPLETE: PASS={statuses['PASS']} WARN={statuses['WARN']} FAIL={statuses['FAIL']}")
    print(f'Output: {output}')
    print('This is a saved-evidence consistency audit, NOT model-training or new independent inference.')
    return 2 if statuses['FAIL'] else 0

if __name__=='__main__':
    raise SystemExit(main())
