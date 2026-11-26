# Training and fine-tuning

RoadForge provides a GUI-based workflow for preparing data, training YOLO26 semantic models, evaluating checkpoints, and exporting deployment artifacts. The generated dataset is also a standard Ultralytics semantic dataset and can be consumed independently.

## 1. Prepare a dataset

1. Open **Dataset Manager**.
2. Add CARLA collection roots or AVIS capture folders (for example `AVIS_RoadForge_Captures`), then click **Rescan**.
3. Filter collections/towns/weather/traffic as needed.
4. Open **Dataset Quality & Splits** and run validation.
5. Review the full report, select only valid pairs, set train/validation/test ratios and a seed, and prepare a dataset copy.
6. For an unseen-town experiment, select one or more held-out towns. Ensure that enough independent collection sessions remain for every non-empty split.

The default split configuration is 80% train, 10% validation, and 10% test. Whole collection sessions are kept intact when there are enough independent sessions; exact duplicate image files stay together. For a single AVIS capture session, RoadForge falls back to deterministic image-level splitting and writes a warning because nearby frames may cross partitions.


## AVIS image-to-semantic fine-tuning

AVIS images can be used without converting them into fake CARLA collections.

1. In the Data Collector application, open **YOLOPv2 Teacher Labeling**. In the new **Teacher source** field, browse to Documents/AVIS_RoadForge_Captures (or the folder where you saved AVIS captures).
2. Select your six-output YOLOPv2 ONNX teacher model and choose Auto/CUDA/CPU. Preview one frame before starting batch annotation. The teacher creates four-class masks (background, drivable area, lane lines, vehicles) and writes an annotation manifest into the AVIS capture folder.
3. In the main RoadForge app, open **Dataset Manager**. Add the same AVIS capture folder as a source root and click **Rescan**. Select the AVIS image set and keep the default filters selected. Town/Weather/Traffic are optional metadata filters; the AVIS scope appears as AVIS and missing CARLA fields remain Unknown.
4. Open **Dataset Quality & Splits**, validate the image/mask pairs, set train/validation/test ratios, and prepare the dataset. Do not select a held-out town for an AVIS-only dataset.
5. In **Model & Training**, select your existing semantic checkpoint as the new fine-tuning checkpoint. Use **Start training** for a fresh fine-tuning run. Do not use **Resume selected last.pt** unless you are continuing the same interrupted run with its original dataset YAML.

When all AVIS images come from one running AVIS process, their capture rows share one session_id. RoadForge automatically falls back to deterministic image-level splitting so all three requested splits can still be created. It keeps byte-identical images in one split and writes a warning because nearby video frames can still leak across train/validation/test. For stronger evaluation, capture several independent AVIS runs and use session-isolated splitting; with enough sessions, RoadForge keeps whole AVIS sessions intact.

The YOLOPv2 output is a pseudo-label, not human ground truth. Review visualizations and masks, particularly vehicle silhouettes and challenging lighting. A checkpoint fine-tuned on these labels is still being trained to imitate YOLOPv2 on AVIS imagery; independent human-reviewed test labels are needed for a reliable accuracy claim.

## 2. Select a semantic model

Open **Model & Training** and choose a YOLO26 semantic model scale (`n`, `s`, `m`, `l`, or `x`).

- Select a local `yolo26*-sem.pt` checkpoint to fine-tune from pretrained weights.
- Use **Get official checkpoint…** to explicitly download an official semantic checkpoint after reviewing the prompt.
- To initialize from the architecture only, leave the checkpoint blank and use the corresponding `yolo26*-sem.yaml` architecture. This does not provide pretrained weights.

RoadForge rejects detection and instance-segmentation checkpoints. It never silently substitutes a `*-seg.pt` model for a semantic model. When source and target class-head dimensions differ, the application records which tensors match by name and shape and which class-dependent tensors require new initialization.

## 3. Configure and start a run

The default image size is 512 pixels. The configuration also records epochs, batch size, optimizer, learning-rate schedule, warm-up, patience, seed, determinism, worker count, cache, device, AMP, freezing, and supported augmentation controls.

Before starting, review the effective configuration in the GUI. Batch `-1` delegates batch selection to the installed trainer. A run receives a unique directory under:

```text
roadforge/workspace/runs/experiment_<timestamp>_<id>/
```

Each run includes:

- `config.json` — normalized training configuration snapshot;
- `split_manifest.json` — copy of the dataset split manifest when available;
- `training.log` — timestamped run log;
- `metrics.json` — metrics captured by the trainer;
- `run_metadata.json` — run identifiers, paths, timing, and status;
- `weights/best.pt` and `weights/last.pt` — trainer checkpoints when produced.

Pause and stop requests are honored at an epoch/checkpoint boundary.

## 4. Resume or fine-tune

Choose **Resume selected last.pt** to continue an interrupted run. A true resume must use the same prepared dataset YAML recorded by the original run.

Choose **Use best.pt as new fine-tune start** to start a fresh experiment, including when using a different dataset, image size, augmentation policy, or optimizer configuration.

## 5. Evaluate

Use validation metrics for checkpoint selection. For the final report, select the prepared dataset's `test` split. RoadForge does not silently substitute validation samples when the test split is empty.

The report includes per-class IoU, precision, recall, Dice, pixel accuracy, mean pixel accuracy, mIoU, foreground mIoU, and a confusion matrix. It also exports JSON/CSV data and a limited set of qualitative examples.

## 6. Standalone Ultralytics example

A prepared dataset can be used outside the GUI with the installed Ultralytics API:

```python
from ultralytics import YOLO

# Fine-tune a semantic checkpoint on a RoadForge-prepared dataset.
model = YOLO("yolo26l-sem.pt", task="semantic")
results = model.train(
    data="path/to/prepared_dataset/data.yaml",
    epochs=100,
    imgsz=512,
    batch=2,
    seed=42,
    deterministic=True,
)
```

This is an illustrative API example. Choose the model scale, image size, batch size, and learning schedule based on GPU memory and the experiment design. Model availability and supported arguments are governed by the pinned Ultralytics release.

## 7. ONNX export and deployment

In **ONNX Export**:

1. Select a trained semantic `.pt` checkpoint.
2. Choose image size and ONNX opset; export the model.
3. Review graph validation, input/output names, dimensions and data types.
4. Run PyTorch/ONNX prediction parity on a representative image.
5. Benchmark warmed-up inference on the target machine.
6. Create a deployment directory from the validated ONNX model.

The deployment bundle contains the model, class mapping, preprocessing metadata, and an inference example. The consuming application must follow the saved graph metadata and preprocessing contract; an ONNX file alone is not a complete deployment specification.
