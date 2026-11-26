# AVIS image-to-semantic fine-tuning

This workflow uses frames captured from AVIS Engine as input to RoadForge's existing YOLOPv2 semantic teacher and YOLO26 semantic student trainer. It does not require CARLA Collector metadata, map names, or a CARLA server.

## 1. Capture images

Use the **Capture frame for RoadForge** button in the AVIS web dashboard. Captures are saved by default under:

    Documents/AVIS_RoadForge_Captures/
    ├── images/
    └── manifest.jsonl

The manifest records capture time, AVIS process/session ID, camera view/version, drive mode, and available telemetry. Images and metadata stay local; do not put generated capture data into Git.

## 2. Run YOLOPv2 Teacher Labeling

1. Launch RoadForge's Data Collector application using run_collector.bat.
2. Open **YOLOPv2 Teacher Labeling**. You do not need to connect to CARLA or start a new collection.
3. In **Teacher source**, browse to the AVIS_RoadForge_Captures folder. This is separate from the CARLA Collector's output-root setting.
4. Select the compatible YOLOPv2 six-output ONNX model. The adapter expects the graph layout documented in DATASET_FORMAT.md. Choose Auto, CUDA, or CPU.
5. Use **Preview Selected Image** and inspect Original RGB, Drivable Area, Lane Lines, Vehicle Mask, Combined classes, and Vehicle detections. Keep vehicle annotation enabled for the four-class student dataset.
6. Select the AVIS image scope and click **Start Annotation**.

The teacher writes annotation_manifest.json and the four-class mask files under labels_yolopv2/AVIS/. It does not require or generate fake CARLA metadata. The AVIS images and manifest.jsonl are not modified.

The current four-class map is:

| ID | Class |
|---:|---|
| 0 | background |
| 1 | ego_drivable_area |
| 2 | lane_lines |
| 3 | vehicles |

YOLOPv2 outputs are pseudo-labels. Inspect the saved visualizations and mask quality before training.

## 3. Discover, validate, and split the AVIS dataset

1. Launch the main RoadForge workbench using run.bat.
2. In **Dataset Manager**, add the same AVIS_RoadForge_Captures folder to **Read-only source roots** and click **Rescan**. The folder can also be selected through its parent directory.
3. Select the AVIS image set. Town/Weather/Traffic filters remain available for CARLA datasets; for AVIS the scope is shown as AVIS and absent CARLA fields are Unknown.
4. Open **Dataset Quality & Splits** and click **Validate selected samples**. A successful teacher run should produce four-class, single-channel PNG masks matching each source image's dimensions.
5. Choose the train/validation/test ratios (for example 80/10/10). Leave town holdout disabled for an AVIS-only dataset. Prepare the dataset.

RoadForge uses AVIS manifest session_id values as split groups. If there are enough independent AVIS runs, complete sessions remain intact. If every captured image belongs to one AVIS process/session, the splitter automatically uses deterministic image-level fallback so non-empty train/validation/test partitions can still be created. Exact byte-identical frames remain together. The dataset report records this fallback and warns that nearby consecutive video frames may appear in different partitions. For more reliable generalization testing, collect multiple independent AVIS runs and use the session-aware split.

## 4. Fine-tune an existing CARLA semantic checkpoint

1. Open **Model & Training**.
2. Select the existing compatible semantic PyTorch checkpoint (for example the previous CARLA run's best.pt) as the checkpoint to fine-tune from.
3. Select the newly prepared AVIS data.yaml, review training settings, and start a new fine-tuning run.
4. Use **Resume selected last.pt** only to continue an interrupted run with exactly its original prepared dataset. When switching from CARLA data to AVIS data, start a new fine-tuning run from best.pt instead.

The existing semantic class contract is four classes. When an older compatible semantic checkpoint has a different class-head size, RoadForge's checkpoint-transfer report shows which compatible weights transfer and which class-shaped tensors are initialized for the target class map. A detection checkpoint or unrelated architecture is not a drop-in semantic checkpoint.

## 5. Scientific notes

The student is trained against YOLOPv2 pseudo-labels, so this measures agreement with the teacher rather than independent real-world truth. Teacher errors can be learned by the student. The camera-only masks also do not provide ground-truth obstacle safety labels or guarantee a safe return manoeuvre. Inspect the labels, run held-out AVIS sessions, and use an independently human-reviewed test set for stronger claims.
