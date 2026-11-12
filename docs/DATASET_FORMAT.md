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

RoadForge uses a fixed three-class semantic label map:

| Pixel value | Class name | Meaning |
|---:|---|---|
| `0` | `background` | All pixels not assigned to either foreground target class |
| `1` | `ego_drivable_area` | Predicted drivable road area |
| `2` | `lane_lines` | Predicted lane-line pixels |

Masks must be **single-channel, 8-bit grayscale PNGs** with the same width and height as the corresponding RGB image. Values other than `0`, `1`, and `2` are rejected unless an explicit ignore value is enabled for validation.

The teacher pipeline stores the two binary outputs in separate directories and the canonical combined target at:

```text
labels_yolopv2/<Town>/combined_3class/<frame-name>.png
```

The combined mask is the training target. The binary drivable/lane masks and visualization overlays are not interchangeable with it.

### Overlap rule

When the teacher's drivable-area and lane masks overlap, lane pixels take precedence in the combined target: class `2` overwrites class `1`. This is part of the current data contract and should be kept consistent in any alternative annotation exporter.

## Teacher-generated labels are pseudo-labels

The annotation stage uses an ONNX teacher. Its output is automatically thresholded and post-processed; it is not a human-reviewed ground-truth annotation by default.

Consequences:

- Teacher false positives and false negatives may be inherited by the student.
- The threshold and morphology policy affect the target distribution and should be documented when comparing experiments.
- Agreement with teacher-generated labels is not equivalent to real-world driving safety or independently verified semantic accuracy.
- For publication-quality claims, use an independently human-annotated and quality-controlled test set where feasible.

## YOLOPv2 ONNX input/output contract

The current adapter expects a six-output ONNX graph. It uses the graph's input dimensions, resizes the RGB frame directly to that shape (without letterboxing), converts to RGB `float32`, scales to `[0, 1]`, and creates an `NCHW` batch tensor.

The adapter reads outputs 4 and 5 as drivable-area and lane-line probability maps. It applies a `0.50` threshold, performs a 3×3 morphological opening and closing on lane predictions, restores masks to the source image resolution using nearest-neighbor interpolation, and combines them using the class mapping above.

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

The YAML declares `images/train`, `images/val`, `images/test`, `masks_dir: masks`, and the three class names. The test directory and YAML entry are present when a non-empty test split is requested.

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
