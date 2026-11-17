# CARLA integration example

RoadForge is designed to support a simulation-to-model workflow for autonomous-driving perception.

## Companion project

The example downstream project is:

**[ArvinNotDev/CARLA_Autonomous_driving](https://github.com/ArvinNotDev/CARLA_Autonomous_driving)**

That repository contains an experimental CARLA autonomous-driving stack with lane perception, drivable-area reasoning, route planning, trajectory prediction, and a live debugging/control interface. RoadForge complements it by providing a repeatable path to collect road-scene RGB data, create semantic pseudo-labels, fine-tune a semantic segmentation model, and export a candidate ONNX artifact.

This relationship is an integration example. A specific checkpoint must still be tested against the consuming repository's model interface, camera geometry, color order, resizing, output layout, class mapping, and runtime requirements before it is used.

## Suggested end-to-end loop

1. Start the CARLA server with the version expected by the collector's Python API.
2. Use the RoadForge Data Collector to acquire raw RGB frames from selected towns, weather conditions, routes/spawn locations, and traffic settings.
3. Annotate the raw frames offline with a compatible YOLOPv2 ONNX teacher.
4. Audit labels and prepare session-aware train/validation/test data.
5. Fine-tune a YOLO26 semantic model on the prepared dataset.
6. Evaluate the final checkpoint on an independent test split.
7. Export the selected checkpoint to ONNX and inspect the graph metadata.
8. Integrate the ONNX artifact into the CARLA perception consumer, adapting its preprocessing and output decoder as required.
9. Test in simulation across unseen towns, weather, lighting, road geometry, and traffic scenarios. Keep a safety fallback and do not treat a successful model load as evidence of safe driving.

## Collector configuration reference

The collector's default reference camera is:

| Parameter | Default |
|---|---:|
| Resolution | 640 × 384 |
| Horizontal field of view | 120° |
| Mount position | `(x=1.4, y=0.0, z=2.2)` |
| Mount rotation | `(pitch=-25°, yaw=0°, roll=0°)` |
| Sensor tick | 0.03 s |

The default town list is Town01–Town07 and Town10HD, with a target of 700 images per enabled town. The configured ego vehicle defaults to `vehicle.tesla.model3`. Background traffic is optional and disabled by default. All of these are configurable; the values recorded in each collection manifest and frame metadata should be treated as the source of truth for an experiment.

The capture worker uses synchronous simulation stepping and records simulation timestamps. It restores the prior world settings when it exits or changes maps.

## CARLA Python API setup

The collector does not bundle CARLA. **The Python client API and running Simulator must come from the exact same CARLA release/build.** A mismatched API can connect and list maps but fail later while reading actor blueprints, sometimes with `ValueError: sticky_control: invalid value type`.

Compare versions using the same Python environment that launches RoadForge:

```powershell
python -c "import carla; c=carla.Client('127.0.0.1', 2000); print('module:', carla.__file__); print('client:', c.get_client_version()); print('server:', c.get_server_version())"
```

Use the client library that matches the server:
- For an official CARLA 1.0 server, install the CARLA 1.0 Python client in RoadForge's active environment (for example, `python -m pip install --upgrade carla==1.0`).
- For a custom/source build, install the matching wheel shipped with that server under `PythonAPI/carla/dist`, selecting the wheel for the active Python version and Windows architecture.
- For a CARLA 0.9.x server, use its corresponding 0.9.x client instead; do not mix the UE4 0.9.x client with the CARLA 1.0 UE5 client/server.

RoadForge logs both versions and the imported `carla` module path. It stops before collection if the reported versions differ and gives a targeted diagnostic for the `sticky_control` Blueprint error.

Run the collector from the repository root:

```bat
run_collector.bat
```

The collector connects to `127.0.0.1:2000` by default; host, port, timeout, and collection options are configurable in the application.
