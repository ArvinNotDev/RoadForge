# CARLA data collector and teacher annotator

This folder contains the acquisition and offline pseudo-labeling component of [RoadForge](../README.md).

## Responsibilities

- Connect to a running CARLA server through its matching Python API.
- Capture raw RGB PNG frames using a configurable ego vehicle and camera.
- Save per-frame JSONL metadata and a collection manifest.
- Enumerate maps installed in the connected CARLA server instead of relying on a fixed Town list.
- Run the same offline YOLOPv2 ONNX teacher over existing frames.
- Save drivable-area, lane-line, and vehicle masks plus combined four-class masks, previews, and an annotation manifest.

The collection stage does not run perception or write labels. Teacher annotation is a separate operation so that the raw data remains available and the model/labeling settings can be inspected independently.

## Launch

From the repository root:

```bat
run_collector.bat
```

Run `setup.bat` once before launching the collector. It creates the project-local `.venv` and installs the dependencies from the root `requirements.txt`, including the CARLA 1.0 Python client. This keeps RoadForge's CARLA 1.0 API separate from any global CARLA 0.9.x installation. `run_collector.bat` automatically uses `.venv` when present.

## CARLA connection

The default connection is `127.0.0.1:2000`. The CARLA server must be running, and the Python API must match the server release. After connecting, RoadForge reads `Client.get_available_maps()` and populates town targets from the actual map list, including map variants supplied by the installed server.

## Default capture profile

- Map choices are empty until the user connects; only maps reported by the running CARLA server are shown. If the client API does not support map enumeration, only the currently loaded map is offered.
- 700 target frames per enabled map
- 640 × 384 RGB camera, 120° FOV
- Camera transform `(1.4, 0.0, 2.2)` and pitch `-25°`
- Sensor tick `0.03 s`
- Tesla Model 3 ego vehicle
- Optional background traffic, disabled by default

All values can be changed in the UI. The collection manifest and frame metadata are the authoritative record of the settings actually used.

For the complete workflow, mask schema, teacher graph assumptions, and scientific limitations, read [Dataset Format](../docs/DATASET_FORMAT.md), [CARLA Integration](../docs/CARLA_INTEGRATION.md), and [Experimental Protocol](../docs/EXPERIMENT_PROTOCOL.md).

## Vehicle semantic annotation

The YOLOPv2 Teacher Labeling tab uses outputs 0–3 of the existing six-output ONNX model to detect COCO cars, motorcycles, buses, and trucks. Those detections are merged into semantic class 3, `vehicles`. GrabCut is the default silhouette estimator; the box mode is an explicitly coarse alternative, and GrabCut failures fall back to the detector box with a count in the annotation manifest. The confidence, NMS threshold, class selection, and extraction mode are recorded per image. Vehicle detection defaults to 0.50 confidence and 0.40 NMS IoU. The bottom 22% of the image is excluded from vehicle detections to suppress the ego-vehicle hood; boxes crossing the cutoff are clipped at the boundary. The settings are recorded per image.

When an existing collection has three-class labels, rescanning marks those annotations as outdated for the current four-class schema. Running annotation creates `vehicle_masks` and `combined_4class` files without deleting the original `combined_3class` masks.
