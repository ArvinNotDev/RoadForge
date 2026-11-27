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

Both use a shared backbone, a 0.30 dropout layer, and one independent classification head per task. ImageNet pretrained weights are enabled by default and are recommended for small datasets; the first use may download the torchvision checkpoint. The input is resized to 224×224 and normalized with ImageNet statistics. Training augmentation uses restrained color jitter and small affine changes. Horizontal flipping is deliberately avoided because it can reverse the meaning of road signs and left/right driving manoeuvres.

Training uses inverse-frequency class weights, light label smoothing, AdamW weight decay, gradient clipping, and a learning rate five times higher for the task heads than for the backbone. The learning-rate scheduler and early stopping monitor mean validation macro-F1 rather than raw accuracy, making model selection less sensitive to majority classes. Early stopping uses a patience of eight epochs. The training loop also tracks the train/validation macro-F1 gap and reports a persistent gap above 0.20 as a possible overfitting warning. These controls reduce overfitting risk but cannot replace more diverse, correctly annotated data.

Before training, use **Preview train / validation / test split** to inspect sample counts, session counts, class balance, excluded boundary frames, and any blockers or warnings. The same split audit is saved in the run report. CUDA is selected automatically when available; otherwise training runs on CPU.

The final report includes per-task accuracy, balanced accuracy, macro-F1, per-class precision/recall/F1, support counts, and confusion matrices. Validation is used for checkpoint selection; the held-out test split is evaluated only after the best validation checkpoint has been restored.

Outputs are stored under:

    roadforge/workspace/classification_runs/<dataset>_<backbone>_<timestamp>/
    ├── best.pt
    ├── last.pt
    ├── training_config.json
    └── training_report.json

The checkpoint contains the backbone, task names, class order, input size, normalization values, and model weights. Use load_classifier_checkpoint from roadforge.classification_training to reconstruct the matching architecture.

## 5. Export an AVIS classifier to ONNX

The **ONNX Export** page supports Classification Lab checkpoints as well as the existing Ultralytics semantic checkpoints. Select the AVIS run's best.pt file and click **Export + validate**. RoadForge detects the checkpoint format from its saved fields; no Ultralytics wrapper, retraining, or checkpoint conversion is required.

For AVIS classification, the exporter reads the backbone, task schema, input size, and ImageNet normalization values directly from the checkpoint. With the example configuration, the graph has:

- One input named images: float32 RGB in NCHW layout, shape [batch, 3, 224, 224].
- One output per task, in checkpoint task order. For the example schema, the outputs are logits_obstacle_ahead and logits_lane, each with shape [batch, 2].
- Raw logits, not probabilities. Apply softmax independently to each output when probabilities are needed; argmax selects a class index.

The input preprocessing must match training: convert the source image to RGB, resize directly to 224×224 (no letterbox), convert pixels to float32 in [0, 1], then normalize each channel with the checkpoint's mean and std values. The ONNX graph embeds the task/class order and preprocessing metadata in its model metadata. Dynamic export changes the batch dimension only; the trained spatial input remains fixed.

Before reporting success, RoadForge checks the ONNX graph, loads it with ONNX Runtime, runs a smoke inference, checks every task output's shape and finite values, and compares each ONNX output against the original PyTorch model. The semantic-only parity viewer and deployment-package generator remain specific to the semantic-segmentation workflow.

## 6. Split policy and limitations

The default target is 80% train, 10% validation, and 10% test.

- With at least three independent AVIS session groups, RoadForge keeps complete sessions in one partition. Assignment considers both split size and per-class distribution. If a class occurs in only one session, that session is forced into training so the model is not asked to learn a class it never sees.
- With fewer than three independent sessions, RoadForge does **not** randomly scatter individual frames. It creates chronological partitions and applies a two-second temporal embargo around same-session split boundaries. Boundary frames are excluded from all partitions. When timestamps are missing in older capture records, manifest/import order is used and the samples touching a same-session boundary are dropped.
- Temporal fallback reduces adjacent-frame leakage, but it is not equivalent to a session-independent test. Collect at least three separate AVIS runs for the strongest evaluation; ideally keep a completely unseen run for final testing.
- Training is blocked when any configured class is absent from the training partition or when the embargo leaves an empty partition. Missing classes in validation/test and classes with fewer than three training examples are reported as warnings because their metrics or learned decision boundaries may be unreliable.

The split audit records the strategy, actual partition ratios, excluded boundary samples, per-task class counts, and sessions represented in each partition.

These are image-level classifiers. They do not return bounding boxes or locations. A traffic-sign classifier with one class per image cannot represent multiple different signs in one frame; use a detector or crop individual signs for that use case. Likewise, “obstacle present” and “safe to return to the right lane” labels are only as reliable as the annotation protocol and visual evidence. A camera-only classification output must not be treated as a safety guarantee or direct permission to execute a manoeuvre. Validate the model in simulation and combine it with independent sensor, geometry, and control checks.
