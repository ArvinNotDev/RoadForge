# AVIS Classification Lab

RoadForge's Classification Lab collects image-level labels for the AVIS Engine. It is independent of the existing semantic-segmentation pipeline: it uses ordinary image classes, not pixel masks or YOLO label files.

## 1. Capture images in AVIS

Launch AVIS Engine with its browser dashboard enabled. In the live monitor, use **Capture frame for RoadForge**. The dashboard saves the latest available front-camera JPEG without pausing the control loop.

Default capture folder:

    %USERPROFILE%\Documents\AVIS_RoadForge_Captures\

The folder contains:

    AVIS_RoadForge_Captures/
    ├── images/
    │   ├── avis_<timestamp>_<id>.jpg
    │   └── ...
    └── manifest.jsonl

Every manifest row records the relative image path, capture timestamp, AVIS process/session ID, camera frame version, current drive mode, and the telemetry snapshot available at capture time. Captures are local data and are not committed to either Git repository.

A new AVIS process creates a new session ID. Collect multiple separate runs when you want session-isolated train/validation/test splits.

## 2. Define one to three tasks

In RoadForge → **Classification Lab**, choose **Create / Open Dataset** and configure one, two, or three tasks. Each task has a name and at least two comma-separated classes. Every annotated image receives exactly one class for every enabled task.

Suggested starting schema:

| Task | Classes |
|---|---|
| Obstacle ahead | no, yes |
| Return to right lane is safe | no, yes |
| Traffic sign | no_sign, stop, speed_limit, no_entry, turn_left, give_way |

Only enable the tasks you intend to train. The class names are editable. A task/class schema cannot be changed after labels have been recorded in that dataset; create a new dataset to preserve the meaning of existing annotations.

## 3. Import and annotate

Choose the AVIS capture folder and click **Import AVIS Captures**. Images are copied into the classification dataset; the original AVIS capture directory is left untouched. Exact SHA-256 duplicates are skipped. Each sample has a stable ID, source path, session ID, capture time, optional telemetry, and a label dictionary.

The annotation view presents one image at a time. Select one class for each configured task and click **Save Labels & Next**. **Skip for now** advances without writing labels. Partially annotated images are not included in training; all configured tasks must have a valid class for a sample to be used.

Dataset layout:

    classification_datasets/<dataset>/
    ├── dataset.json
    ├── samples.jsonl
    └── images/
        └── <sha256-prefix>.jpg

## 4. Train

Choose either:

- **ResNet18** — compact residual feature extractor with a 512-feature shared representation.
- **EfficientNet-B0** — compact compound-scaled feature extractor with a 1280-feature shared representation.

Both use a shared backbone and one independent linear classification head per task. Training uses cross-entropy for each head and averages the task losses. ImageNet pretrained weights are enabled by default and are recommended for small datasets; the first use may download the torchvision checkpoint. The input is resized to 224×224 and normalized with ImageNet statistics. Color jitter is used for training augmentation; horizontal flipping is deliberately avoided because it can reverse the meaning of road signs and driving manoeuvres.

The backbone learning rate is lower than the task-head learning rate. Training selects the best checkpoint by mean validation accuracy and uses early stopping. GPU is selected automatically when CUDA is available; otherwise training runs on CPU.

Outputs are stored under:

    roadforge/workspace/classification_runs/<dataset>_<backbone>_<timestamp>/
    ├── best.pt
    ├── last.pt
    ├── training_config.json
    └── training_report.json

The checkpoint contains the backbone, task names, class order, input size, normalization values, and model weights. Use load_classifier_checkpoint from roadforge.classification_training to reconstruct the matching architecture.

## 5. Split policy and limitations

RoadForge groups images by AVIS process/session whenever at least three sessions are available, keeping complete sessions in one split. With fewer than three sessions, it falls back to a deterministic image-level split and emits a warning; temporally adjacent frames can then leak across partitions. For meaningful evaluation, collect multiple separate AVIS runs and reserve unseen sessions for validation and test.

These are image-level classifiers. They do not return bounding boxes or locations. A traffic-sign classifier with one class per image cannot represent multiple different signs in one frame; use a detector or crop individual signs for that use case. Likewise, “obstacle present” and “safe to return to the right lane” labels are only as reliable as the annotation protocol and visual evidence. A camera-only classification output must not be treated as a safety guarantee or direct permission to execute a manoeuvre. Validate the model in simulation and combine it with independent sensor, geometry, and control checks.
