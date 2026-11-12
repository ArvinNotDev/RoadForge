# CARLA data collector and teacher annotator

This folder contains the acquisition and offline pseudo-labeling component of [RoadForge](../README.md).

## Responsibilities

- Connect to a running CARLA server through its matching Python API.
- Capture raw RGB PNG frames using a configurable ego vehicle and camera.
- Save per-frame JSONL metadata and a collection manifest.
- Run an offline YOLOPv2 ONNX teacher over existing frames.
- Save drivable-area masks, lane-line masks, combined three-class masks, previews, and an annotation manifest.

The collection stage does not run perception or write labels. Teacher annotation is a separate operation so that the raw data remains available and the model/labeling settings can be inspected independently.

## Launch

From the repository root:

```bat
run_collector.bat
```

The root `setup.bat` installs the shared desktop/runtime dependencies. The CARLA Python API is not bundled and must be made available to the selected Python environment separately.

## CARLA connection

The default connection is `127.0.0.1:2000`. The CARLA server must be running, and the Python API must match the server release.

## Default capture profile

- Town01–Town07 and Town10HD
- 700 target frames per enabled town
- 640 × 384 RGB camera, 120° FOV
- Camera transform `(1.4, 0.0, 2.2)` and pitch `-25°`
- Sensor tick `0.03 s`
- Tesla Model 3 ego vehicle
- Optional background traffic, disabled by default

All values can be changed in the UI. The collection manifest and frame metadata are the authoritative record of the settings actually used.

For the complete workflow, mask schema, teacher graph assumptions, and scientific limitations, read [Dataset Format](../docs/DATASET_FORMAT.md), [CARLA Integration](../docs/CARLA_INTEGRATION.md), and [Experimental Protocol](../docs/EXPERIMENT_PROTOCOL.md).
