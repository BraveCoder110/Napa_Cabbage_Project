"""Session 6.1: audit longitudinal JPEG archives without altering source images.

Run from Napa_Cabbage_Project project root; no third-party packages required.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
import re
import zipfile
from pathlib import Path

DAYS = (0, 3, 5, 7)
HEADS = tuple(f'C{i:02d}' for i in range(1, 6))
LEAVES = tuple(f'L{i:02d}' for i in range(1, 5))
SIDES = ('正', '反')
PATTERN = re.compile(r'^(?:D(?P<day>0|3|5|7))?(?P<head>C0[1-5])(?P<leaf>L0[1-4])(?P<side>[正反])\.jpe?g$', re.IGNORECASE)


def save_csv(path, rows, columns):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description='Audit 4 longitudinal ZIP datasets (no image rewriting).')
    for d in DAYS:
        p.add_argument(f'--d{d}', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    sources = {d: getattr(a, f'd{d}') for d in DAYS}
    for d, path in sources.items():
        if not path.is_file():
            p.error(f'D{d}: ZIP not found: {path}')
        if not zipfile.is_zipfile(path):
            p.error(f'D{d}: not a ZIP archive: {path}')
    if a.output.exists():
        p.error(f'Output already exists; refusing overwrite: {a.output}')

    observed = {}
    file_records = []
    unexpected = []
    # Hash ZIP member bytes so future analysis can verify that original JPGs were unchanged.
    for day in DAYS:
        path = sources[day]
        with zipfile.ZipFile(path) as z:
            for member in z.infolist():
                if member.is_dir():
                    continue
                basename = Path(member.filename).name
                if not basename.lower().endswith(('.jpg', '.jpeg')):
                    unexpected.append({'day': f'D{day}', 'zip_member': member.filename, 'issue': 'non_JPEG_member'})
                    continue
                match = PATTERN.fullmatch(basename)
                if not match:
                    unexpected.append({'day': f'D{day}', 'zip_member': member.filename, 'issue': 'unrecognised_filename'})
                    continue
                embedded_day = match['day']
                if day == 0:
                    if embedded_day not in (None, '0'):
                        raise ValueError(f'D0 archive includes mismatching day: {member.filename}')
                elif embedded_day != str(day):
                    raise ValueError(f'D{day} archive includes mismatching day: {member.filename}')
                head, leaf, side = match['head'], match['leaf'], match['side']
                key = (day, head, leaf, side)
                if key in observed:
                    raise ValueError(f'Duplicate observation for {key}: {member.filename}')
                with z.open(member) as f:
                    sha = hashlib.sha256()
                    while block := f.read(1024 * 1024):
                        sha.update(block)
                observed[key] = True
                file_records.append({
                    'day': f'D{day}', 'day_number': day, 'head_id': head, 'leaf_id': leaf,
                    'tracked_leaf_id': head + leaf, 'side': 'front' if side == '正' else 'reverse',
                    'filename': basename, 'source_zip': path.name,
                    'source_zip_member': member.filename, 'original_bytes': member.file_size,
                    'sha256_original_jpeg': sha.hexdigest(),
                    'gt_status': 'reviewed_pilot' if leaf == 'L01' and side == '正' else 'not_manually_verified',
                    'image_role': 'longitudinal_visual_observation',
                    'freshness_ground_truth': 'not_available',
                })
    if unexpected:
        raise ValueError(f'Unexpected archive entries: {unexpected[:5]}')

    expected = []
    for head in HEADS:
        for leaf in LEAVES:
            for side in SIDES:
                for day in DAYS:
                    present = (day, head, leaf, side) in observed
                    expected.append({
                        'tracked_leaf_id': head + leaf, 'head_id': head, 'leaf_id': leaf,
                        'side': 'front' if side == '正' else 'reverse', 'day': f'D{day}',
                        'expected_filename': ('' if day == 0 else f'D{day}') + head + leaf + side + '.jpg',
                        'status': 'PRESENT' if present else 'MISSING',
                        'reason': '' if present else 'not_collected_or_not_in_archive',
                    })

    seq = []
    pairs = []
    for head in HEADS:
        for leaf in LEAVES:
            for side in SIDES:
                dates = [d for d in DAYS if (d, head, leaf, side) in observed]
                seq.append({
                    'tracked_leaf_id': head + leaf, 'head_id': head, 'leaf_id': leaf,
                    'side': 'front' if side == '正' else 'reverse',
                    'observed_days': ','.join('D'+str(d) for d in dates),
                    'n_timepoints': len(dates), 'complete_4_timepoints': int(len(dates) == 4),
                    'd0_reference_available': int(0 in dates),
                    'valid_d0_pairs': sum(1 for d in (3,5,7) if 0 in dates and d in dates),
                })
                for day in (3,5,7):
                    if 0 in dates and day in dates:
                        pairs.append({
                            'tracked_leaf_id': head+leaf, 'head_id': head, 'leaf_id': leaf,
                            'side': 'front' if side == '正' else 'reverse',
                            'baseline_day': 'D0', 'followup_day': f'D{day}',
                            'baseline_filename': head+leaf+side+'.jpg',
                            'followup_filename': f'D{day}'+head+leaf+side+'.jpg',
                            'pair_status': 'valid_same_leaf_same_side',
                        })

    totals = collections.Counter(r['day'] for r in file_records)
    missing = [r for r in expected if r['status'] == 'MISSING']
    summary = {
        'study': 'longitudinal visual freshness-related appearance pilot',
        'independent_source_heads': len(HEADS), 'distinct_tracked_leaves': len(HEADS)*len(LEAVES),
        'timepoints': [f'D{d}' for d in DAYS], 'photographs_observed_total': len(file_records),
        'photographs_expected_if_complete': len(expected),
        'observed_per_day': {f'D{d}': totals[f'D{d}'] for d in DAYS},
        'missing_count': len(missing), 'missing_filenames': [r['expected_filename'] for r in missing],
        'tracked_leaf_side_series_total': len(seq),
        'complete_four_timepoint_series': sum(r['complete_4_timepoints'] for r in seq),
        'incomplete_series': sum(not r['complete_4_timepoints'] for r in seq),
        'complete_front_series': sum(r['complete_4_timepoints'] for r in seq if r['side']=='front'),
        'complete_reverse_series': sum(r['complete_4_timepoints'] for r in seq if r['side']=='reverse'),
        'valid_D0_to_followup_pairs': len(pairs),
        'biological_independence_warning': 'Images are repeated/nested observations from only five source heads.',
        'ground_truth_warning': 'Storage day is not a measured freshness or food-safety label.',
        'photography_warning': 'Colour and projected size may vary with lighting, pose, camera and possible chilling/freezing effects.',
        'metadata_warning': 'Actual capture timestamps should come from laboratory notes; image metadata was not used as proof of capture time.',
        'images_rewritten': False,
    }
    a.output.mkdir(parents=True, exist_ok=False)
    save_csv(a.output/'observation_manifest_157.csv', sorted(file_records, key=lambda r: (r['head_id'],r['leaf_id'],r['side'],r['day_number'])), list(file_records[0]))
    save_csv(a.output/'expected_grid_160.csv', expected, list(expected[0]))
    save_csv(a.output/'missing_observations.csv', missing, list(expected[0]))
    save_csv(a.output/'longitudinal_series_40.csv', seq, list(seq[0]))
    save_csv(a.output/'D0_followup_pairs_111.csv', pairs, list(pairs[0]))
    (a.output/'audit_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (a.output/'README_RESEARCH_BOUNDARIES.txt').write_text(
        'Session 6.1 longitudinal inventory.\n'
        'All 157 observations are original photographs, NOT independent biological replicates.\n'
        'Each leaf is nested within one of 5 cabbage heads.\n'
        'D0-to-followup pairs compare the same tracked leaf and the same photographed side.\n'
        'Missing D0 reverse shots must not be imputed.\n'
        'Storage day is not a validated freshness score.\n'
        'This inventory does NOT generate foreground masks or estimate freshness.\n'
        'A reviewed GT or separately validated model is required before segmentation-based indicators.\n', encoding='utf-8')
    assert len(file_records)==157 and len(missing)==3 and len(seq)==40 and len(pairs)==111, summary
    print(f"PASS: {len(file_records)}/160 observed images, {len(missing)} missing, {len(seq)} tracked side-series")
    print(f"PASS: {sum(r['complete_4_timepoints'] for r in seq)} complete 4-day series; {len(pairs)} valid D0/follow-up pairs")
    print('Missing: '+', '.join(r['expected_filename'] for r in missing))
    print('Output: '+str(a.output))

if __name__ == '__main__':
    main()
