# Portfolio and résumé notes

This page provides a factual project summary that can be adapted for a CV, résumé, portfolio, or research-project application.

## One-line description

**RoadForge — Simulation-to-model semantic perception workbench:** a Windows desktop application for CARLA road-scene data acquisition, YOLOPv2 ONNX pseudo-labeling, dataset quality control, session-aware splitting, YOLO26 semantic-model fine-tuning, evaluation, and ONNX deployment validation.

## Résumé bullet templates

Use only statements that match the implementation and your actual contribution.

- Designed and implemented an end-to-end data-to-deployment workflow for road-scene semantic segmentation, combining CARLA simulation capture, offline ONNX teacher annotation, YOLO26 semantic fine-tuning, and ONNX export.
- Implemented dataset integrity checks for paired RGB images and semantic masks, including format/dimension validation, class-ID checks, annotation-manifest consistency, and SHA-256 duplicate detection.
- Developed collection-session-aware train/validation/test partitioning with duplicate-content group linking and optional held-out-town evaluation to reduce common forms of temporal and exact-duplicate leakage.
- Built a PySide6 desktop workbench with background workers, versioned experiment configurations, checkpoint resume/fine-tune workflows, and machine-readable run artifacts.
- Implemented semantic-segmentation evaluation with per-class IoU, precision, recall, Dice, pixel accuracy, mIoU, foreground mIoU, and confusion-matrix reporting.
- Added ONNX graph validation, PyTorch/ONNX prediction parity checks, warmed-up inference benchmarks, and a metadata-bearing deployment package.

## Skills demonstrated

**Computer vision and ML:** semantic segmentation, teacher–student workflows, pseudo-labeling, transfer learning, dataset curation, class-imbalance inspection, IoU/Dice evaluation, ONNX inference.

**Autonomous-driving simulation:** CARLA sensor configuration, controlled data collection, town/weather variation, simulation metadata, synthetic-domain evaluation.

**Software engineering:** Python, PySide6, Qt worker threads, modular architecture, file-format contracts, manifests, reproducibility, testing, CI, documentation.

## Evidence to add before claiming quantitative results

A strong portfolio entry should link to a reproducible experiment, not only the application code. Add the following only after collecting and independently verifying the results:

1. Dataset summary: number of frames, valid labeled samples, towns, sessions, weather/traffic configurations, and class-pixel distribution.
2. Teacher audit: representative manually reviewed masks and measured pseudo-label quality on a small annotated sample.
3. Student results: per-class validation/test IoU and Dice, foreground mIoU, confusion matrix, split strategy, and held-out-town results.
4. Deployment results: target GPU/CPU, model scale, input dimensions, warm-up count, benchmark iterations, latency percentiles, FPS, and ONNX parity.
5. Reproducibility: saved configuration, split manifest, run metadata, package versions, and model hashes.

Do not add placeholder metrics or describe a teacher-agreement score as ground-truth accuracy.
