"""Export a fixed, traceable target-domain annotation pilot without altering source ZIPs."""
import argparse
import csv
import hashlib
import re
import zipfile
from pathlib import Path

SCHEDULE = ("D0", "D3", "D5", "D7")
HEADS = tuple(f"C{i:02d}" for i in range(1, 6))


def main():
    parser = argparse.ArgumentParser(description="Prepare 20 front-side L01 images for independent polygon annotation.")
    for day in SCHEDULE:
        parser.add_argument(f"--{day.lower()}", required=True, help=f"Original {day} image ZIP")
    parser.add_argument("--output", required=True, help="A new, nonexistent output directory")
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        parser.error(f"Output already exists; refusing to overwrite: {out}")

    sources = {day: Path(getattr(args, day.lower())) for day in SCHEDULE}
    exported = []
    # Validate all 20 expected inputs before writing any outputs.
    for day, archive_path in sources.items():
        with zipfile.ZipFile(archive_path) as archive:
            names = [n for n in archive.namelist() if not n.endswith("/")]
            for head in HEADS:
                base = ("" if day == "D0" else day) + f"{head}L01正.jpg"
                matching = [name for name in names if Path(name).name == base]
                if len(matching) != 1:
                    raise ValueError(f"Expected exactly one {base} in {archive_path}; found {len(matching)}")
                info = archive.getinfo(matching[0])
                if info.file_size <= 0:
                    raise ValueError(f"Empty photo: {matching[0]}")
                exported.append((day, head, archive_path, matching[0], f"{day}{head}L01正.jpg"))
    out.mkdir(parents=True)
    image_dir = out / "images"
    image_dir.mkdir()
    rows = []
    for day, head, archive_path, member, target_name in exported:
        with zipfile.ZipFile(archive_path) as archive:
            payload = archive.read(member)
        (image_dir / target_name).write_bytes(payload)
        rows.append({
            "image": target_name,
            "day": day,
            "source_head": head,
            "leaf_id": f"{head}L01",
            "side": "front",
            "source_zip": archive_path.name,
            "source_zip_member": member,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "annotation_status": "PENDING",
            "annotation_file": "",
            "reviewer": "",
            "review_notes": "",
        })
    with (out / "annotation_manifest.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "ANNOTATION_POLICY.txt").write_text(
        "Pilot subset: 5 source heads x 4 time points, the FRONT of L01 only (20 photos).\n"
        "Goal: create independently reviewed masks for zero-shot target-domain evaluation.\n"
        "Label the visible tissue belonging to the numbered detached leaf; exclude the tabletop, hands,\n"
        "other detached leaves and background. Record any ambiguity or detached fragments.\n"
        "Do NOT use existing model predictions to draw ground truth. Do NOT label image quality/freshness.\n"
        "Do NOT edit original JPGs. Keep polygon JSON and record reviewer/revision information.\n"
        "This subset is an evaluation pilot, not an independent set of 20 biological specimens.\n"
        "All images from the same mother head must stay in one group in any future adaptation study.\n"
        "Do not fine-tune on these evaluation images and then report zero-shot accuracy on them.\n",
        encoding="utf-8",
    )
    print(f"PASS: Exported {len(rows)} photos from {len(HEADS)} source heads and {len(SCHEDULE)} dates")
    print(f"Output: {image_dir}")
    print(f"Manifest: {out / 'annotation_manifest.csv'}")
    print("NEXT: Independently annotate leaf foreground masks; do not use model predictions as labels.")

if __name__ == "__main__":
    main()
