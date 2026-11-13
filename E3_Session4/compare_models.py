"""Compare two source-domain segmentation models using saved per-image metrics.

Run from the Napa_Cabbage_Project root. This script neither trains nor loads models.
It reads CSV metrics only, never alters existing experiment results.
"""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median

def read_metrics(path):
    rows = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"image", "dice", "iou"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"{path}: CSV requires columns image,dice,iou; got {reader.fieldnames}")
        for row in reader:
            key = row["image"].strip()
            if not key or key in rows:
                raise ValueError(f"{path}: missing or repeated image ID {key!r}")
            dice, iou = float(row["dice"]), float(row["iou"])
            if not all(0 <= x <= 1 for x in (dice, iou)):
                raise ValueError(f"{path}: out-of-range metric for {key}")
            rows[key] = {"dice": dice, "iou": iou}
    return rows

def save_csv(path, fields, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def main():
    p = argparse.ArgumentParser(description="Read-only paired comparison on saved validation CSVs.")
    p.add_argument("--custom", required=True, help="Custom U-Net per_image_metrics.csv")
    p.add_argument("--transfer", required=True, help="ResNet18 U-Net per_image_metrics.csv")
    p.add_argument("--output", required=True, help="New empty output directory")
    p.add_argument("--top-k", type=int, default=10)
    a = p.parse_args()
    if a.top_k < 1:
        p.error("--top-k must be >= 1")
    out = Path(a.output)
    if out.exists():
        p.error(f"Refusing to overwrite existing path: {out}")
    c, t = read_metrics(a.custom), read_metrics(a.transfer)
    if set(c) != set(t):
        only_c = sorted(set(c) - set(t))
        only_t = sorted(set(t) - set(c))
        raise ValueError(f"Image IDs do not match. Custom-only: {only_c[:10]}; Transfer-only: {only_t[:10]}")
    if not c:
        raise ValueError("No metrics found.")
    paired = []
    for name in sorted(c):
        dc = c[name]["dice"]
        dt = t[name]["dice"]
        di = t[name]["iou"] - c[name]["iou"]
        delta = dt - dc
        verdict = "transfer" if delta > 1e-9 else "custom" if delta < -1e-9 else "tie"
        paired.append({
            "image": name,
            "custom_dice": dc,
            "transfer_dice": dt,
            "delta_dice_transfer_minus_custom": delta,
            "custom_iou": c[name]["iou"],
            "transfer_iou": t[name]["iou"],
            "delta_iou_transfer_minus_custom": di,
            "higher_dice": verdict
        })
    out.mkdir(parents=True)
    fields = list(paired[0])
    save_csv(out / "paired_metrics.csv", fields, paired)
    worst_custom = sorted(paired, key=lambda r: r["custom_dice"])[:a.top_k]
    worst_transfer = sorted(paired, key=lambda r: r["transfer_dice"])[:a.top_k]
    gain_transfer = sorted(paired, key=lambda r: r["delta_dice_transfer_minus_custom"], reverse=True)[:a.top_k]
    gain_custom = sorted(paired, key=lambda r: r["delta_dice_transfer_minus_custom"])[:a.top_k]
    for filename, rows in [
        ("worst_custom.csv", worst_custom),
        ("worst_transfer.csv", worst_transfer),
        ("largest_transfer_gains.csv", gain_transfer),
        ("largest_custom_gains.csv", gain_custom),
    ]:
        save_csv(out / filename, fields, rows)
    changes = [r["delta_dice_transfer_minus_custom"] for r in paired]
    wins = {v: sum(r["higher_dice"] == v for r in paired) for v in ("custom", "transfer", "tie")}
    summary = {
        "n_matched_validation_images": len(paired),
        "custom_mean_image_dice": mean(r["custom_dice"] for r in paired),
        "transfer_mean_image_dice": mean(r["transfer_dice"] for r in paired),
        "custom_mean_image_iou": mean(r["custom_iou"] for r in paired),
        "transfer_mean_image_iou": mean(r["transfer_iou"] for r in paired),
        "mean_paired_delta_dice_transfer_minus_custom": mean(changes),
        "median_paired_delta_dice_transfer_minus_custom": median(changes),
        "higher_dice_count": wins,
        "worst_k_overlap": sorted(set(r["image"] for r in worst_custom) & set(r["image"] for r in worst_transfer)),
        "interpretation_warning": (
            "Per-image averages are NOT global pixel-aggregated Dice/IoU. "
            "These 115 validation images are not a new independent test set. "
            "Paired comparisons are descriptive, not a biological-replicate significance test."
        )
    }
    (out / "comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sorted_rows = sorted(paired, key=lambda r: r["delta_dice_transfer_minus_custom"])
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.bar(range(len(sorted_rows)), [r["delta_dice_transfer_minus_custom"] for r in sorted_rows])
        ax.axhline(0, linewidth=1)
        ax.set_xlabel("Validation images sorted by paired Dice difference")
        ax.set_ylabel("ResNet18 Dice - Custom Dice")
        ax.set_title("Paired per-image Dice differences (source domain)")
        fig.tight_layout()
        fig.savefig(out / "paired_dice_differences.png", dpi=160)
        plt.close(fig)
    except ImportError:
        print("NOTE: matplotlib unavailable; numeric results saved, plot skipped.")
    print(f"PASS: Compared {len(paired)} matched validation images")
    print(f"Mean per-image Dice: Custom={summary['custom_mean_image_dice']:.4f} | ResNet18={summary['transfer_mean_image_dice']:.4f}")
    print(f"Mean paired Dice difference (ResNet18 - Custom): {mean(changes):+.4f}")
    print(f"Higher Dice counts: Custom={wins['custom']}, ResNet18={wins['transfer']}, Tie={wins['tie']}")
    print(f"PASS: New outputs saved to {out}")
    print("NOTE: Global Dice/IoU must be taken from the original full evaluation summaries, not these per-image means.")

if __name__ == "__main__":
    main()
