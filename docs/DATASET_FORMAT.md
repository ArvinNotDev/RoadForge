# Dataset format and annotation contract

This document describes the on-disk contract consumed by RoadForge. Keeping the format explicit makes it possible to use prepared datasets with other compatible training pipelines.

## Raw collection layout

The CARLA collector writes timestamped collection folders under the selected output root:

```text
datasets/
└── collection_YYYYMMDD_HHMMSS/
    ├── collection_manifest.json
    ├── raw/
    │   └── Town10HD/
    │       ├── images/
    │       │   ├── frame_000001.png
    │       │   └── ...
    │       └── metadata.jsonl
    └── annotation_manifest.json
```

`metadata.jsonl` contains one JSON object per captured frame. Records include capture and simulation context such as the town, weather, route/spawn identifiers, camera settings, and collection-session identifier. Exact fields depend on the collector configuration and CARLA API values available at capture time.

The annotation manifest is written by the offline teacher-labeling stage. Each image entry records annotation status and the output paths, including the canonical combined mask.

## Canonical semantic mask

RoadForge uses a fixed four-class semantic label map:

| Pixel value | Class name | Meaning |
|---:|---|---|
| `0` | `background` | All pixels not assigned to either foreground target class |
| `1` | `ego_drivable_area` | Predicted drivable road area |
| `2` | `lane_lines` | Predicted lane-line pixels |
| `3` | `vehicles` | Combined car, motorcycle, bus, and truck masks |

Masks must be **single-channel, 8-bit grayscale PNGs** with the same width and height as the corresponding RGB image. Values other than `0`, `1`, `2`, and `3` are rejected unless an explicit ignore value is enabled for validation.

The teacher pipeline stores the two binary outputs in separate directories and the canonical combined target at:

```text
labels_yolopv2/<Town>/combined_4class/<frame-name>.png
```

The combined mask is the training target. The binary drivable/lane masks and visualization overlays are not interchangeable with it.

### Overlap rule

When the teacher masks overlap, the combined target uses this priority: vehicle class `3` overwrites lane class `2`, lane class `2` overwrites drivable class `1`, and all remaining pixels stay background class `0`. This prevents the student from learning the same vehicle pixels as drivable road or lane markings.


### Vehicle masks are pseudo-segmentation targets

YOLOPv2 provides vehicle bounding boxes, not pixel-accurate instance masks. RoadForge turns the selected detections into a single semantic vehicle mask. The recommended GrabCut mode estimates a foreground silhouette within each detector region; when it fails, the code falls back to the box and increments a counter. The alternative box mode intentionally labels the full detector rectangle. Inspect the red vehicle preview and fallback count before training. These masks are teacher-derived pseudo-labels, not human annotations.

Legacy collections with `combined_3class` masks remain on disk. They are discovered for audit but validation blocks them from a new four-class split until YOLOPv2 Teacher Labeling is rerun. Reannotation writes `vehicle_masks` and `combined_4class` files, then updates the annotation manifest only after all outputs succeed; the previous mask folder is not deleted.

## Teacher-generated labels are pseudo-labels

The annotation stage uses an ONNX teacher. Its output is automatically thresholded and post-processed; it is not a human-reviewed ground-truth annotation by default.

Consequences:

- Teacher false positives and false negatives may be inherited by the student.
- The threshold and morphology policy affect the target distribution and should be documented when comparing experiments.
- Agreement with teacher-generated labels is not equivalent to real-world driving safety or independently verified semantic accuracy.
- For publication-quality claims, use an independently human-annotated and quality-controlled test set where feasible.

## YOLOPv2 ONNX input/output contract

The current adapter expects a six-output ONNX graph. It uses the graph's input dimensions, resizes the RGB frame directly to that shape (without letterboxing), converts to RGB `float32`, scales to `[0, 1]`, and creates an `NCHW` batch tensor.

The adapter reads outputs 0–3 as the YOLOPv2 COCO-80 detection heads and anchor grids, output 4 as the drivable-area two-channel map, and output 5 as the lane-line probability map. It uses the existing `0.50` segmentation threshold, performs a 3×3 morphological opening and closing on lane predictions, restores the masks to the source resolution, and decodes class-aware vehicle boxes from the detection head. COCO IDs 2 (car), 3 (motorcycle), 5 (bus), and 7 (truck) are merged into semantic class `3`. The default detection confidence threshold is `0.30`; class-aware NMS uses IoU `0.45`. Vehicle boxes are converted to masks using GrabCut by default, with box-fill fallback recorded in the manifest when GrabCut cannot produce a stable component.

**Important:** this adapter is specific to the graph layout and tensor semantics it validates. A different YOLOPv2 export, output order, preprocessing pipeline, or post-processing convention requires an explicit adapter change and a parity check against the intended reference implementation.

## Prepared training dataset

After validation and splitting, RoadForge copies samples into this structure:

```text
prepared_dataset/
├── data.yaml
├── dataset_report.json
├── split_manifest.json
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── masks/
    ├── train/
    ├── val/
    └── test/
```

The YAML declares `images/train`, `images/val`, optional `images/test`, `masks_dir: masks`, and the four class names: `background`, `ego_drivable_area`, `lane_lines`, and `vehicles`. The test directory and YAML entry are present when a non-empty test split is requested.

`split_manifest.json` stores each prepared sample's exact image and mask paths, source paths, town, collection, session/group identifiers, weather, traffic metadata, route ID, and split assignment. Evaluation reads this manifest rather than trying to reconstruct the original filenames.

## Validation checks

The validation stage checks, among other things:

- image decode success and mask decode success;
- existence of source image and declared mask;
- correct single-channel grayscale PNG encoding;
- exact image/mask dimensions;
- allowed class IDs;
- annotation-manifest image/mask pairing;
- empty foreground masks;
- duplicate filenames and byte-identical images.

Validation issues are exported as JSON. Invalid pairs are not silently converted or coerced.
