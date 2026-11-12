# Troubleshooting

## The main application does not start

- Confirm that 64-bit Python 3.12 is available: `py -3.12 --version`.
- Run `setup.bat` and inspect the first dependency installation error.
- Launch through `run.bat` so the local `.venv` is used.
- For Qt platform plugin errors, reinstall PySide6 inside the environment:
  `.venv\Scripts\python.exe -m pip install --force-reinstall PySide6`.

## The collector cannot import `carla`

RoadForge does not redistribute the CARLA Python API. Install or expose the Python API package/egg from the CARLA release that matches the running server. Confirm that the API directory is on the Python import path used by `.venv\Scripts\python.exe`, then launch `run_collector.bat` again.

## The collector cannot connect

- Confirm the CARLA server is running.
- Check the host and port (defaults: `127.0.0.1:2000`).
- Confirm the server is reachable and the configured timeout is sufficient.
- Ensure the Python API and server versions are compatible.

## CUDA is unavailable

Open **Settings** and review the detected Python, PyTorch, CUDA, GPU, and ONNX Runtime providers. CPU execution remains available when supported. To use CUDA, install a PyTorch wheel compatible with the NVIDIA driver and a matching ONNX Runtime GPU package. Do not assume that installing a CUDA toolkit alone makes a CPU-only wheel GPU-enabled.

## YOLOPv2 annotation rejects the ONNX model

The current adapter expects a compatible six-output YOLOPv2 graph. It reads output index 4 as a two-channel drivable-area output and output index 5 as a one-channel lane-line probability map. A different export may use a different output count, ordering, tensor layout, or probability/logit convention. Do not disable the checks to force an incompatible graph through the pipeline; implement and test a matching adapter instead.

The annotator uses the ONNX graph's input dimensions, direct resize without letterboxing, RGB `float32` values in `[0, 1]`, and NCHW layout. Confirm that these choices match the checkpoint's reference inference pipeline.

## No dataset samples appear

- In **Dataset Manager**, confirm that the selected root contains `collection_*` folders.
- Check that raw images are under `raw/<Town>/images/`.
- The primary training target is the `combined_mask` entry in `annotation_manifest.json`.
- Binary mask folders and visualization overlays are not accepted as substitutes for combined semantic targets.
- For a manifest-free collection, the fallback is a single-channel PNG with the matching filename under `labels_yolopv2/<Town>/combined_3class/`.
- Click **Rescan** after adding or annotating collections.

## Dataset validation reports errors

Open **Dataset Quality & Splits** and inspect every issue. Check missing files, decode problems, PNG/grayscale mode, exact dimensions, class IDs 0/1/2, and manifest pairing. If value `255` intentionally marks an ignore region, enable the explicit ignore-label option before validation and preparation.

## Split creation reports too few independent groups

The splitter keeps collection sessions intact and merges groups containing byte-identical images. Each non-empty requested split needs at least one independent group after any town holdout. Add collection sessions, reduce the number of non-empty splits, or revise the holdout plan.

## Prepared split ratios differ from the requested ratios

The split algorithm assigns whole groups; it does not break sessions into individual frames just to satisfy an exact percentage. A held-out town can also move an entire session into test. Review `dataset_report.json` and `split_manifest.json` for actual counts and assignments.

## Training rejects an ignore label

The current integration supports an explicit ignore value during validation/evaluation but blocks training when train or validation masks contain ignored pixels. This is intentional: the current application build does not verify an ignore-index loss path for its installed Ultralytics semantic trainer. Remove or relabel those pixels using an explicit annotation policy, or use a trainer with documented ignore-index support.

## Training runs out of GPU memory

- Reduce the batch size.
- Reduce image size.
- Try batch `-1` to delegate selection to the installed trainer.
- Use a smaller model scale.
- Resume from the previous run's `last.pt` after checking the run snapshot and original dataset YAML.

Hardware use depends on model scale, image size, batch size, augmentations, and framework/runtime versions. The app does not change settings automatically.

## Resume and fine-tune are different

- **Resume from `last.pt`** continues the original experiment and requires the dataset YAML recorded by that run.
- **Fine-tune from `best.pt`** starts a new run with the current configuration and can use a different dataset.

Use the fine-tune path when changing datasets or experiment settings.

## ONNX export or inference fails

- Confirm that the source is a local semantic `.pt` checkpoint.
- Use supported image-size and opset settings.
- Review the graph input/output names, shapes, and data types reported by the exporter.
- Compare the exported model against PyTorch on the same image.
- For deployment, follow the bundle's `preprocessing.json`; do not assume every ONNX graph uses identical input dimensions or color handling.

## Video or webcam inference is unavailable

Confirm that OpenCV is installed in the environment. Close other applications that may be using the webcam and check Windows camera permissions. The default webcam index is 0.
