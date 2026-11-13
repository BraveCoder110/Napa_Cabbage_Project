# Postharvest Shanghai Bok Choy Dataset for Pre-Cutting Sorting

## Overview

This dataset contains RGB images and instance-level annotations of postharvest Shanghai bok choy (*Brassica rapa* subsp. *chinensis*) collected for visual perception in pre-cutting sorting. The samples cover two cultivars, 'Youmei No. 2' and 'Dongchun No. 22', and four postharvest states involving basal yellowed leaves and root residues.

The release contains the original fixed training and validation split used in the associated experiments. No independent test split was used. The dataset contains 575 images and 909 annotated instances.

## Directory Structure

```text
release_dataset/
|-- images/             # 575 RGB JPEG images
|-- labels/             # 575 instance-annotation JSON files
|-- train.json          # Fixed training split and image-label pairs
|-- val.json            # Fixed validation split and image-label pairs
|-- dataset_info.json   # Dataset metadata and class definitions
|-- LICENSE.txt         # CC BY 4.0 license notice
`-- README.md
```

All images use the JPEG format. Each image has one JSON annotation file with the same stem. Samples are sequentially named using the zero-padded `hik_NNNN` convention. For example, `images/hik_0001.jpg` corresponds to `labels/hik_0001.json`. The training samples range from `hik_0001` to `hik_0460`, and the validation samples range from `hik_0461` to `hik_0575`.

## Dataset Statistics

| Split | Images | Instances | Yellow leaves | Root residue | Qualified | Both defects |
|---|---:|---:|---:|---:|---:|---:|
| Train | 460 | 725 | 125 | 156 | 288 | 156 |
| Validation | 115 | 184 | 32 | 39 | 73 | 40 |
| Total | 575 | 909 | 157 | 195 | 361 | 196 |

The split was generated with seed 42 and a validation ratio of 0.20. Candidate allocations were searched to reduce class and source imbalance while keeping all instances from the same image in one split.

## Class Definitions

| ID | Internal name | Definition |
|---:|---|---|
| 0 | `bokchoy_yellow` | Bok choy with basal yellowed leaves and no root residue |
| 1 | `bokchoy_root` | Bok choy with root residue and no basal yellowed leaves |
| 2 | `bokchoy` | Qualified bok choy without either defect |
| 3 | `bokchoy_yellow_root` | Bok choy with both basal yellowed leaves and root residue |

## Annotation Format

Each file in `labels/` has the following structure:

```json
{
  "image": "hik_0001.jpg",
  "instances": [
    {
      "class_id": 1,
      "class_name": "bokchoy_root",
      "group_id": 1,
      "bbox_xywhn": [0.50, 0.50, 0.40, 0.30],
      "polygon_xyn": [[0.30, 0.40], [0.70, 0.40], [0.70, 0.60]],
      "keypoints": {
        "root": [0.45, 0.65, 2],
        "leaf": [0.55, 0.30, 2]
      }
    }
  ]
}
```

- `class_id`: zero-based class identifier.
- `class_name`: internal class name corresponding to `class_id`.
- `group_id`: instance identifier within the image.
- `bbox_xywhn`: normalized bounding box `[center_x, center_y, width, height]`.
- `polygon_xyn`: normalized instance-mask polygon vertices `[x, y]`.
- `keypoints.root`: normalized root-end keypoint `[x, y, visibility]`.
- `keypoints.leaf`: normalized leaf-end reference point `[x, y, visibility]`.
- `visibility = 2`: annotated and visible keypoint.
- `visibility = 0`: unavailable direction annotation; its coordinates must not be used for supervision.

Among the 909 instances, 864 contain valid root and leaf reference points and 45 use visibility value 0 for both points.

## Split Files

`train.json` and `val.json` contain explicit image-label pairs:

```json
{
  "split": "train",
  "num_samples": 460,
  "samples": [
    {
      "id": "hik_0001",
      "image": "images/hik_0001.jpg",
      "label": "labels/hik_0001.json"
    }
  ]
}
```

Paths are relative to the release root so that the dataset can be moved without modification.

## License and Citation

The dataset is released under the Creative Commons Attribution 4.0 International license (CC BY 4.0). See `LICENSE.txt` for details.

When using this dataset, please cite the corresponding Mendeley Data record and the associated research article after their DOI information becomes available.

## Data Integrity Notes

The release does not include framework-specific YOLO text labels, COCO conversions, generated cache files, trained weights, or model predictions. The JSON files in `labels/` are the canonical annotations for this release.
