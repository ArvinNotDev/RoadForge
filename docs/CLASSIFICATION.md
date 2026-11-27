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

The annotation view presents one image at a time. Select one class for each configured task and click **Save Labels & Next**. After a successful save, those labels become the defaults for the next unannotated frame, so repeated classes can be confirmed quickly. Unsaved changes are never carried forward. The last saved labels are restored from the dataset when Classification Lab is reopened. **Skip for now** advances without writing labels. Partially annotated images are not included in training; all configured tasks must have a valid class for a sample to be used.

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

Both use a shared backbone, a 0.15 dropout layer, and one independent classification head per task. ImageNet pretrained weights are enabled by default and are recommended for small datasets; the first use may download the torchvision checkpoint. The input is resized to 224×224 and normalized with ImageNet statistics. Training augmentation uses restrained color jitter and small affine changes. Horizontal flipping is deliberately avoided because it can reverse the meaning of road signs and left/right driving manoeuvres.

Training uses inverse-frequency class weights, no label smoothing, modest AdamW weight decay (1e-4), gradient clipping, and a learning rate five times higher for the task heads than for the backbone. Classification Lab defaults to up to 80 epochs and early stopping patience of 16 epochs. The scheduler and checkpoint selection monitor mean validation macro-F1 rather than raw accuracy, making selection less sensitive to majority classes. The training loop also tracks the train/validation macro-F1 gap and reports a persistent gap above 0.20 as a possible overfitting warning. This lighter regularization gives the model more room to fit the fixed AVIS environment; the stratified validation set still guards against simply memorizing the training subset.

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

The default target is 80% train, 10% validation, and 10% test. Classification Lab uses an **image-level, multi-task stratified split** (`image_stratified`) for the AVIS fixed-road workflow.

- The splitter balances both partition size and every configured task/class distribution. With 494 annotated images, the target is approximately 395 train, 49 validation, and 50 test images.
- Rare classes are deliberately distributed into validation and test whenever at least three distinct images containing that class are available. This prevents classes captured in only one AVIS session (for example, a particular sign or turn) from being trapped entirely in training.
- Exact SHA-256 duplicate images are grouped and always remain in one partition. Different frames from the same AVIS capture session are allowed to appear in more than one partition.
- No frames are discarded at temporal boundaries, so all valid annotations can be used. The preview and training report identify shared sessions and explicitly warn that these metrics are **in-domain**: they estimate performance on the same road/camera conditions, not generalization to a new route or unseen AVIS session.
- A class absent from the complete annotated dataset remains a training blocker. Missing validation/test classes are warned about when there are too few distinct images to place that class in every partition.

This policy matches deployments where the camera, route, and road environment are intentionally fixed. For a future deployment on unseen roads or capture conditions, collect independent AVIS runs and use a session-isolated evaluation protocol rather than comparing those results directly with this in-domain score.

