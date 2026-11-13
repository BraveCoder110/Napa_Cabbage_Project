"""Read-only CS7002SR AI model artifact audit; Python standard library only.

Run from the Napa_Cabbage_Project root. Never modifies project inputs.
Output directory must not exist. Statuses are structural checks, not model validation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


CHECKS = [
    ("CRITICAL", "Custom U-Net architecture", "E3_Session3/models/unet.py", "file"),
    ("CRITICAL", "Custom U-Net source training", "E3_Session3/train_baseline.py", "file"),
    ("CRITICAL", "Custom U-Net source evaluation", "E3_Session3/evaluate_baseline.py", "file"),
    ("CRITICAL", "Custom U-Net saved checkpoint", "experiments/session3_4_baseline_v2/best_unet.pth", "file"),
    ("CRITICAL", "ResNet18 U-Net architecture", "E3_Session4/models/resnet18_unet.py", "file"),
    ("CRITICAL", "ResNet18 U-Net source training", "E3_Session4/train_transfer_baseline.py", "file"),
    ("CRITICAL", "ResNet18 U-Net source evaluation", "E3_Session4/evaluate_transfer_baseline.py", "file"),
    ("CRITICAL", "ResNet18 source checkpoint", "experiments/session4_4_transfer_v1/best_resnet18_unet.pth", "file"),
    ("CRITICAL", "Zero-shot target evaluation", "E3_Session5/evaluate_zero_shot.py", "file"),
    ("CRITICAL", "Leave-one-head-out few-shot script", "E3_Session5/run_grouped_few_shot.py", "file"),
    ("CRITICAL", "20 reviewed target masks", "experiments/Session5_1_GT_Pilot20_Final_v1/masks", "dir"),
    ("CRITICAL", "20 pilot target RGB", "experiments/session5_1_target_annotation_v1/images", "dir"),
    ("CRITICAL", "Target D0 archive", "datasets/Huang_Jing_Bai_Cai/Huang_Jing_Bai_Cai_D0.zip", "zip"),
    ("CRITICAL", "Target D3 archive", "datasets/Huang_Jing_Bai_Cai/Huang_Jing_Bai_Cai_D3.zip", "zip"),
    ("CRITICAL", "Target D5 archive", "datasets/Huang_Jing_Bai_Cai/Huang_Jing_Bai_Cai_D5.zip", "zip"),
    ("CRITICAL", "Target D7 archive", "datasets/Huang_Jing_Bai_Cai/Huang_Jing_Bai_Cai_D7.zip", "zip"),
    ("IMPORTANT", "Longitudinal data audit", "E3_Session6/audit_longitudinal.py", "file"),
    ("IMPORTANT", "Analysis-only model training", "E3_Session6/train_final_roi_model.py", "file"),
    ("IMPORTANT", "157 candidate mask manifest", "experiments/session6_2_analysis_roi_candidates_v1/candidate_roi_manifest_157.csv", "csv"),
    ("IMPORTANT", "157 full resolution candidate masks", "experiments/session6_2_analysis_roi_candidates_v1/masks_full_resolution", "dir"),
    ("IMPORTANT", "Visual feature extraction", "E3_Session6/extract_visual_freshness_features.py", "file"),
    ("IMPORTANT", "157 image features", "experiments/session6_3_visual_features_provisional_v1/visual_features_157_PROVISIONAL.csv", "csv"),
    ("IMPORTANT", "111 D0 follow-up pairs", "experiments/session6_3_visual_features_provisional_v1/D0_followup_deltas_111_PROVISIONAL.csv", "csv"),
    ("IMPORTANT", "GT-versus-auto comparison v2", "E3_Session6/compare_gt_auto_roi_features_v2.py", "file"),
    ("IMPORTANT", "20 GT/auto comparisons", "experiments/session6_4_gt_auto_sensitivity_v2/per_image_GT_vs_Auto_20.csv", "csv"),
    ("IMPORTANT", "Photo colour diagnostic", "E3_Session6/diagnose_photo_color_v1.py", "file"),
    ("SUPPORTING", "20 GT photo-colour diagnostic output", "experiments/session6_4_2_photo_color_reproduction_v1/pilot20_leaf_wood_color.csv", "csv"),
    ("SUPPORTING", "Report-ready figures", "experiments/session6_5_report_results_v1", "dir"),
]

EXPECTED = {"Huang_Jing_Bai_Cai_D0.zip": 37, "Huang_Jing_Bai_Cai_D3.zip": 40,
            "Huang_Jing_Bai_Cai_D5.zip": 40, "Huang_Jing_Bai_Cai_D7.zip": 40}
COUNT_EXPECTED = {
    "candidate_roi_manifest_157.csv": 157,
    "visual_features_157_PROVISIONAL.csv": 157,
    "D0_followup_deltas_111_PROVISIONAL.csv": 111,
    "per_image_GT_vs_Auto_20.csv": 20,
    "pilot20_leaf_wood_color.csv": 20,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            raise ValueError("CSV is empty")
        return sum(1 for _ in reader)


def inspect_zip(path: Path, expected: int | None, full_crc: bool):
    with zipfile.ZipFile(path) as z:
        bad = z.testzip() if full_crc else None
        # count jpeg images by suffix, not all archive entries
        images = [n for n in z.namelist() if n.lower().endswith((".jpg", ".jpeg"))]
        if bad:
            return "INVALID", f"ZIP CRC failed: {bad}"
        if expected is not None and len(images) != expected:
            return "CHECK", f"JPG count {len(images)}; expected {expected}"
        return "OK", f"JPG count {len(images)}" + ("; CRC passed" if full_crc else "; CRC not tested")


def inspect_one(root: Path, priority: str, name: str, rel: str, kind: str, hash_mode: bool, crc: bool):
    path = root / rel
    row = dict(priority=priority, item=name, relative_path=rel, kind=kind,
               status="MISSING", details="", size_bytes="", sha256="")
    if not path.exists():
        return row
    if kind == "dir":
        if not path.is_dir():
            row.update(status="INVALID", details="Expected directory")
        else:
            files = [p for p in path.iterdir() if p.is_file()]
            subdirs = [p for p in path.iterdir() if p.is_dir()]
            row.update(status="OK" if files or subdirs else "CHECK", details=f"{len(files)} immediate files, {len(subdirs)} subdirectories")
            if name == "20 reviewed target masks" and len(list(path.glob("*.png"))) != 20:
                row.update(status="CHECK", details=f"Expected 20 PNG GT masks; found {len(list(path.glob('*.png')))}")
            if name == "157 full resolution candidate masks" and len(list(path.glob("*.png"))) != 157:
                row.update(status="CHECK", details=f"Expected 157 PNG masks; found {len(list(path.glob('*.png')))}")
        return row
    if not path.is_file():
        row.update(status="INVALID", details="Expected file")
        return row
    row["size_bytes"] = path.stat().st_size
    row["status"] = "OK" if path.stat().st_size > 0 else "INVALID"
    if kind == "zip" and row["status"] == "OK":
        try:
            row["status"], row["details"] = inspect_zip(path, EXPECTED.get(path.name), crc)
        except Exception as e:
            row.update(status="INVALID", details=f"ZIP error: {type(e).__name__}: {e}")
    elif kind == "csv" and row["status"] == "OK":
        try:
            n = csv_rows(path)
            expected = COUNT_EXPECTED.get(path.name)
            row["details"] = f"{n} data rows" + (f"; expected {expected}" if expected is not None else "")
            if expected is not None and n != expected:
                row["status"] = "CHECK"
        except Exception as e:
            row.update(status="INVALID", details=f"CSV read error: {type(e).__name__}: {e}")
    else:
        row["details"] = "Exists; content/implementation not functionally tested"
    if hash_mode and row["status"] not in ("MISSING", "INVALID"):
        try:
            row["sha256"] = sha256_file(path)
        except OSError as e:
            row["status"] = "CHECK"
            row["details"] += f"; hash failed: {e}"
    return row


def main():
    ap = argparse.ArgumentParser(description="Read-only project deliverable audit")
    ap.add_argument("--project-root", type=Path, default=Path.cwd())
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--hash-files", action="store_true", help="SHA256 existing checked files and checkpoints")
    ap.add_argument("--check-zip-crc", action="store_true", help="Read all ZIP bytes to verify CRC; slower")
    args = ap.parse_args()
    root = args.project_root.expanduser().resolve()
    out = args.output.expanduser().resolve()
    if not root.is_dir():
        ap.error(f"Project root missing: {root}")
    if out.exists():
        ap.error(f"Output already exists; refusing to overwrite: {out}")
    if root == out:
        ap.error("Output cannot be project root")
    rows = [inspect_one(root, *spec, args.hash_files, args.check_zip_crc) for spec in CHECKS]
    summaries = Counter(r["status"] for r in rows)
    optional_note = (
        "This is an inventory, not a reproducibility or scientific validity test. "
        "Status OK confirms path/content structure only. "
        "Source dataset 575-image binary masks, full source training results, LOHO fold results, "
        "and all 157 ROI QA are not fully verified by this manifest. "
        "The 20-image analysis mask Dice 0.9576 is in-sample and NOT the five-fold out-of-fold Dice 0.9483."
    )
    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "project_root_checked": str(root),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "status_counts": dict(summaries),
        "critical_missing_or_invalid": [r["relative_path"] for r in rows if r["priority"] == "CRITICAL" and r["status"] != "OK"],
        "notes": optional_note,
        "user_confirmations_needed": [
            "Official public Bok Choy dataset URL, author, usage licence and redistribution permission",
            "Required source dataset/mask availability and clear access instructions",
            "Functional rerun evidence for source val, target zero-shot, and five LOHO folds",
            "Checkpoint safety/compatibility load verified in the user's own trusted environment",
            "Whether model checkpoints are retained in final submission and within platform upload limit",
            "Academic integrity / AI tool declaration and final Canvas packaging rules",
        ],
    }
    out.mkdir(parents=True, exist_ok=False)
    with (out / "artifact_inventory_v2.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "artifact_audit_summary_v2.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    md = ["# CS7002SR AI Model Artifact: structural audit", "", "Run from project root; no inputs were modified.", "", "| Priority | Item | Status | Details |", "|---|---|---|---|"]
    md.extend(f"| {r['priority']} | {r['item']} | {r['status']} | {r['details'].replace('|','/')} |" for r in rows)
    md += ["", "## Interpretation", optional_note, "", "## User verification tasks"]
    md.extend("- " + s for s in metadata["user_confirmations_needed"])
    (out / "artifact_audit_summary_v2.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("AI MODEL ARTIFACT STRUCTURAL AUDIT")
    print(f"Total checks: {len(rows)}; OK={summaries['OK']} MISSING={summaries['MISSING']} CHECK={summaries['CHECK']} INVALID={summaries['INVALID']}")
    for r in rows:
        if r["status"] != "OK":
            print(f"  [{r['status']}] {r['priority']}: {r['relative_path']} :: {r['details']}")
    print(f"Output: {out}")
    print("IMPORTANT: structural audit only. No checkpoints loaded, no code run, no dataset licence confirmed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
