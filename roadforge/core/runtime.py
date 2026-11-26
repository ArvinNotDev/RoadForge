from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import shutil
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from .config import DEFAULT_CLASSES, snapshot_config
from .metrics import confusion_matrix, metrics_from_confusion

PALETTE = np.array([
    [0, 0, 0],       # background
    [40, 210, 120],  # ego_drivable_area
    [255, 210, 40],  # lane_lines
    [255, 70, 70],   # vehicles
], dtype=np.uint8)  # RGB


def atomic_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_report() -> dict:
    report = {"python": platform.python_version(), "platform": platform.platform(), "cpu": platform.processor() or "Unknown", "cuda": False}
    try:
        import torch
        report.update({"torch": torch.__version__, "cuda": bool(torch.cuda.is_available())})
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            free, total = torch.cuda.mem_get_info(0)
            report.update({"gpu": props.name, "vram_total_gb": round(total / 1024**3, 2), "vram_free_gb": round(free / 1024**3, 2), "cuda_version": torch.version.cuda, "compute_capability": f"{props.major}.{props.minor}", "amp_available": True})
        else:
            report.update({"gpu": "Unavailable", "amp_available": False})
    except Exception as exc:
        report.update({"torch": "Unavailable", "gpu": "Unavailable", "amp_available": False, "torch_error": str(exc)})
    try:
        import ultralytics
        report["ultralytics"] = ultralytics.__version__
    except Exception as exc:
        report["ultralytics"] = f"Unavailable: {exc}"
    try:
        import onnxruntime as ort
        report["onnxruntime"] = ort.__version__
        report["onnx_providers"] = ort.get_available_providers()
    except Exception as exc:
        report["onnxruntime"] = f"Unavailable: {exc}"
    try:
        import PySide6
        report["pyside6"] = PySide6.__version__
    except Exception as exc:
        report["pyside6"] = f"Unavailable: {exc}"
    return report


def official_checkpoint_name(size: str) -> str:
    size = size.lower().strip()
    if size not in {"n", "s", "m", "l", "x"}:
        raise ValueError("Model size must be one of n, s, m, l, x.")
    return f"yolo26{size}-sem.pt"


def download_official_checkpoint(size: str, destination: str | Path | None = None) -> str:
    """Explicitly obtain an official Ultralytics semantic checkpoint."""
    from ultralytics.utils.downloads import attempt_download_asset
    name = official_checkpoint_name(size)
    resolved = Path(attempt_download_asset(name))
    if not resolved.is_file():
        raise FileNotFoundError(f"Ultralytics did not return a checkpoint file for {name}.")
    if destination:
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.resolve() != resolved.resolve():
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite existing checkpoint: {target}")
            shutil.copy2(resolved, target)
            resolved = target
    return str(resolved)


def _ultralytics_yolo(path: str | Path, task: str = "semantic"):
    from ultralytics import YOLO
    path = Path(path)
    if path.suffix.lower() in {".pt", ".onnx"} and not path.is_file():
        raise FileNotFoundError(f"Model file does not exist: {path}\nDownload or select the checkpoint explicitly before loading it.")
    model = YOLO(str(path), task=task, verbose=False)
    model_task = getattr(model, "task", None)
    model_class = type(getattr(model, "model", None)).__name__
    if task == "semantic" and model_task != "semantic":
        raise TypeError(f"Selected model is not a YOLO semantic-segmentation model (task={model_task!r}, class={model_class}).")
    if task == "semantic" and path.suffix.lower() != ".onnx" and "SemanticSegmentationModel" not in model_class:
        raise TypeError(f"Selected model architecture is not semantic segmentation (task={model_task!r}, class={model_class}).")
    if task == "semantic" and path.suffix.lower() == ".onnx":
        import onnxruntime as ort
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        metadata = session.get_modelmeta().custom_metadata_map
        metadata_task = metadata.get("task") or metadata.get("task_type")
        if metadata_task and metadata_task != "semantic":
            raise TypeError(f"ONNX metadata declares task={metadata_task!r}; semantic segmentation is required.")
        if not session.get_outputs():
            raise TypeError("ONNX graph has no outputs.")
    return model


def verify_checkpoint(path: str | Path, expected_classes: int = 3) -> dict:
    model = _ultralytics_yolo(path, "semantic")
    names = getattr(model, "names", {})
    source_nc = len(names) if names is not None else 0
    config = getattr(model.model, "yaml", {}) or {}
    return {"path": str(Path(path).resolve()), "task": model.task, "architecture": type(model.model).__name__,
            "source_classes": source_nc, "target_classes": expected_classes,
            "class_adaptation_required": source_nc != expected_classes,
            "architecture_nc": config.get("nc"), "names": {int(k): str(v) for k, v in dict(names).items()},
            "sha256": file_hash(path) if Path(path).is_file() else None}


def transfer_report(path: str | Path, target_classes: int = 3) -> dict:
    """Compare checkpoint tensor names/shapes for the four-class semantic model."""
    import torch
    from ultralytics.nn.tasks import SemanticSegmentationModel

    source = _ultralytics_yolo(path, "semantic")
    source_net = source.model
    source_state = source_net.state_dict()
    target = SemanticSegmentationModel(source_net.yaml, nc=target_classes, ch=3, verbose=False)
    target_state = target.state_dict()
    transferred = [key for key, value in source_state.items() if key in target_state and value.shape == target_state[key].shape]
    changed = [{"key": key, "source_shape": list(value.shape), "target_shape": list(target_state[key].shape)}
               for key, value in source_state.items() if key in target_state and value.shape != target_state[key].shape]
    skipped = [key for key in source_state if key not in target_state]
    initialized = [key for key in target_state if key not in source_state]
    return {"source_checkpoint": str(Path(path).resolve()), "source_classes": len(source.names), "target_classes": target_classes,
            "transferred_count": len(transferred), "transferred": transferred, "modified_count": len(changed),
            "modified": changed, "initialized_count": len(initialized), "initialized": initialized,
            "skipped_count": len(skipped), "skipped": skipped,
            "note": "Only same-name tensors with identical shapes transfer. Class-dependent output tensors are newly initialized when their class count changes."}


def _semantic_array_to_mask(output, target_hw: tuple[int, int] | None = None) -> np.ndarray:
    """Decode common semantic logits/class-map tensor layouts into a uint8 class map."""
    values = output
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    values = np.asarray(values)

    if values.ndim == 4:
        if values.shape[0] != 1:
            raise RuntimeError(f"Expected a single-image semantic output batch, got {values.shape}.")
        values = values[0]

    if values.ndim == 3:
        # Ultralytics and some exported graphs expose a discrete class map as
        # [1, H, W]; it is not a one-channel logits tensor.
        if values.shape[0] == 1:
            values = values[0]
        elif values.shape[-1] == 1:
            values = values[..., 0]
    if values.ndim == 3:
        first_is_channels = values.shape[0] <= 256 and values.shape[1] > 1 and values.shape[2] > 1 and values.shape[0] < min(values.shape[1:])
        last_is_channels = values.shape[-1] <= 256 and values.shape[0] > 1 and values.shape[1] > 1 and values.shape[-1] < min(values.shape[:2])
        if first_is_channels and not last_is_channels:
            channel_axis = 0
        elif last_is_channels and not first_is_channels:
            channel_axis = -1
        elif first_is_channels:
            # Ambiguous tiny tensors are treated as channel-first, matching ONNX export.
            channel_axis = 0
        else:
            raise RuntimeError(f"Unrecognized 3D semantic output layout: {values.shape}")

        if target_hw is not None:
            target_h, target_w = map(int, target_hw)
            channels = values.shape[channel_axis]
            if channel_axis == 0:
                resized = [
                    np.asarray(Image.fromarray(np.asarray(values[i], dtype=np.float32), mode="F").resize(
                        (target_w, target_h), Image.Resampling.BILINEAR
                    ))
                    for i in range(channels)
                ]
                values = np.stack(resized, axis=0)
            else:
                resized = [
                    np.asarray(Image.fromarray(np.asarray(values[..., i], dtype=np.float32), mode="F").resize(
                        (target_w, target_h), Image.Resampling.BILINEAR
                    ))
                    for i in range(channels)
                ]
                values = np.stack(resized, axis=-1)
        values = values.argmax(axis=channel_axis)

    if values.ndim != 2:
        raise RuntimeError(f"Unrecognized semantic prediction shape: {values.shape}")

    if np.issubdtype(values.dtype, np.floating):
        if not np.all(np.isfinite(values)):
            raise RuntimeError("Semantic prediction contains NaN or infinite values.")
        rounded = np.rint(values)
        if not np.allclose(values, rounded, atol=1e-4):
            raise RuntimeError("A two-dimensional semantic output must contain discrete class IDs, not un-decoded scores.")
        values = rounded
    if values.size and (values.min() < 0 or values.max() > 255):
        raise RuntimeError(f"Semantic class IDs must fit in uint8; got range [{values.min()}, {values.max()}].")

    mask = values.astype(np.uint8, copy=False)
    if target_hw is not None and mask.shape != tuple(map(int, target_hw)):
        target_h, target_w = map(int, target_hw)
        mask = np.asarray(Image.fromarray(mask).resize((target_w, target_h), Image.Resampling.NEAREST))
    return mask


def _class_map_from_result(result) -> np.ndarray:
    mask = getattr(result, "semantic_mask", None)
    if mask is None:
        raise RuntimeError("Semantic predictor returned no semantic_mask. Verify the model task and Ultralytics version.")
    if hasattr(mask, "data"):
        mask = mask.data
    return _semantic_array_to_mask(mask)


class SemanticEngine:
    def __init__(self, model_path: str | Path, imgsz: int = 512, device: str = "auto", conf: float = 0.25):
        self.model_path = str(Path(model_path).expanduser())
        self.imgsz = int(imgsz)
        if self.imgsz < 32:
            raise ValueError("Inference image size must be at least 32 pixels.")
        self.conf = float(conf)
        if not 0.0 <= self.conf <= 1.0:
            raise ValueError("Confidence must be between 0 and 1.")
        import torch
        is_onnx = Path(self.model_path).suffix.lower() == ".onnx"
        if is_onnx:
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RuntimeError("ONNX inference requires onnxruntime. Install the project requirements first.") from exc
            available_providers = set(ort.get_available_providers())
            cuda_runtime = "CUDAExecutionProvider" in available_providers
            wants_gpu = torch.cuda.is_available() and cuda_runtime
            if device == "auto":
                self.device = "0" if wants_gpu else "cpu"
            elif str(device).lower() in {"cpu", "-1"}:
                self.device = "cpu"
            elif not cuda_runtime:
                raise RuntimeError(
                    "GPU inference was explicitly requested for this ONNX model, but ONNX Runtime "
                    "does not report CUDAExecutionProvider. Select CPU or install a compatible GPU runtime."
                )
            else:
                self.device = "0" if str(device).lower() in {"cuda", "cuda:0", "0"} else str(device)
        else:
            self.device = "0" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else str(device))
        self.model = _ultralytics_yolo(self.model_path, "semantic")

    def synchronize(self) -> None:
        """Wait for queued CUDA work so latency measurements are meaningful."""
        if self.device != "cpu":
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                # Synchronization is a benchmark aid, not a reason to break inference.
                pass

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[0] < 1 or rgb.shape[1] < 1:
            raise ValueError(f"Expected an HxWx3 RGB image, got shape {rgb.shape}.")
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        results = self.model.predict(source=bgr, imgsz=self.imgsz, device=self.device, conf=self.conf, verbose=False, save=False)
        if not results:
            raise RuntimeError("The model returned no prediction results.")
        prediction = _class_map_from_result(results[0])
        expected = rgb.shape[:2]
        if prediction.shape != expected:
            prediction = np.asarray(Image.fromarray(prediction).resize((expected[1], expected[0]), Image.Resampling.NEAREST))
        return prediction

def colorize(mask: np.ndarray) -> np.ndarray:
    result = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in enumerate(PALETTE):
        result[mask == class_id] = color
    return result


def overlay(rgb: np.ndarray, mask: np.ndarray, opacity: float = 0.42) -> np.ndarray:
    rgb = np.asarray(rgb).astype(np.uint8)
    mask = np.asarray(mask)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or mask.ndim != 2 or rgb.shape[:2] != mask.shape:
        raise ValueError("Overlay requires an RGB image and a same-sized single-channel class map.")
    opacity = float(np.clip(opacity, 0.0, 1.0))
    colors = colorize(mask)
    active = mask != 0
    result = rgb.copy()
    result[active] = np.rint(rgb[active] * (1 - opacity) + colors[active] * opacity).astype(np.uint8)
    return result


def run_image_inference(model_path: str, image_path: str, output_dir: str, imgsz: int = 512, device: str = "auto", opacity: float = 0.42) -> dict:
    image_path = Path(image_path).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    with Image.open(image_path) as im:
        rgb = np.asarray(im.convert("RGB"))
    engine = SemanticEngine(model_path, imgsz, device)
    predicted = engine.predict(rgb)

    output = Path(output_dir).expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(output)
    if output.exists() and any(output.iterdir()):
        output = output / f"{image_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    output.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    mask_path, overlay_path = output / f"{stem}_mask.png", output / f"{stem}_overlay.png"
    drive_path = output / f"{stem}_drivable.png"
    lane_path = output / f"{stem}_lane_lines.png"
    vehicles_path = output / f"{stem}_vehicles.png"
    Image.fromarray(predicted).save(mask_path)
    Image.fromarray(overlay(rgb, predicted, opacity)).save(overlay_path)
    Image.fromarray((predicted == 1).astype(np.uint8) * 255).save(drive_path)
    Image.fromarray((predicted == 2).astype(np.uint8) * 255).save(lane_path)
    Image.fromarray((predicted == 3).astype(np.uint8) * 255).save(vehicles_path)
    return {
        "image": str(image_path), "mask": str(mask_path), "overlay": str(overlay_path),
        "drivable": str(drive_path), "lane_lines": str(lane_path),
        "vehicles": str(vehicles_path),
        "classes_present": sorted(map(int, np.unique(predicted))),
    }


def run_folder_inference(model_path: str, folder: str, output_dir: str, imgsz: int = 512, device: str = "auto", opacity: float = 0.42, progress=None) -> dict:
    from .dataset import IMAGE_EXTENSIONS

    source = Path(folder).expanduser().resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)

    requested_output = Path(output_dir).expanduser().resolve()
    if requested_output.exists() and not requested_output.is_dir():
        raise NotADirectoryError(requested_output)
    output = requested_output
    if output.exists() and any(output.iterdir()):
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        output = output.with_name(f"{output.name}_{stamp}")
        suffix = 2
        while output.exists():
            output = output.with_name(f"{requested_output.name}_{stamp}_{suffix}")
            suffix += 1

    excluded_roots = [requested_output.resolve(), output.resolve()]

    def is_excluded(path: Path) -> bool:
        resolved = path.resolve()
        return any(resolved == root or root in resolved.parents for root in excluded_roots)

    images = sorted(
        path for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and not is_excluded(path)
    )
    if not images:
        raise ValueError(f"No supported images found in {source} (excluding the selected output folder).")

    # Load before creating output artifacts so a missing/bad model leaves no empty run folder.
    engine = SemanticEngine(model_path, imgsz, device)
    output.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(images, 1):
        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"))
        prediction = engine.predict(rgb)
        relative = path.relative_to(source)
        target_dir = output / relative.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        identity = hashlib.sha256(relative.as_posix().casefold().encode("utf-8", errors="replace")).hexdigest()[:8]
        output_stem = f"{path.stem}_{identity}"
        Image.fromarray(prediction).save(target_dir / f"{output_stem}_mask.png")
        Image.fromarray(overlay(rgb, prediction, opacity)).save(target_dir / f"{output_stem}_overlay.png")
        Image.fromarray((prediction == 1).astype(np.uint8) * 255).save(target_dir / f"{output_stem}_drivable.png")
        Image.fromarray((prediction == 2).astype(np.uint8) * 255).save(target_dir / f"{output_stem}_lane_lines.png")
        Image.fromarray((prediction == 3).astype(np.uint8) * 255).save(target_dir / f"{output_stem}_vehicles.png")
        if progress:
            progress(index, len(images), path.name)
    return {"processed": len(images), "output_dir": str(output), "model": str(Path(model_path).resolve())}


def run_video_inference(model_path: str, video_path: str, output_path: str, imgsz: int = 512, device: str = "auto", opacity: float = 0.42, save_masks: bool = False, progress=None) -> dict:
    from collections import deque
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"Could not open video: {video_path}")

    writer = None
    target: Path | None = None
    mask_dir: Path | None = None
    mask_dir_created = False
    completed = False
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            fps = 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise OSError("The input video reported invalid frame dimensions.")
        total = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))

        # Load the model before creating output artifacts; failure leaves no broken video.
        engine = SemanticEngine(model_path, imgsz, device)
        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite an existing video: {target}")

        mask_dir = target.with_name(target.stem + "_masks") if save_masks else None
        if mask_dir and mask_dir.exists():
            raise FileExistsError(f"Refusing to overwrite an existing mask directory: {mask_dir}")
        if mask_dir:
            mask_dir.mkdir(parents=True, exist_ok=False)
            mask_dir_created = True

        writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise OSError(f"Could not create output video: {target}")

        latency_samples = deque(maxlen=10000)
        total_processing = 0.0
        frames = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            start = time.perf_counter()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            prediction = engine.predict(rgb)
            vis = overlay(rgb, prediction, opacity)
            writer.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            if mask_dir:
                mask_file = mask_dir / f"frame_{frames:06d}.png"
                if not cv2.imwrite(str(mask_file), prediction):
                    raise OSError(f"Could not write prediction mask: {mask_file}")
            elapsed = time.perf_counter() - start
            total_processing += elapsed
            latency_samples.append(elapsed * 1000)
            frames += 1
            if progress:
                progress(frames, total, f"{1 / max(elapsed, 1e-9):.2f} FPS")

        if frames == 0:
            raise OSError("The input video contained no decodable frames.")
        latencies = np.asarray(latency_samples, dtype=np.float64)
        result = {
            "frames": frames, "source_fps": fps,
            "processing_fps": float(frames / total_processing) if total_processing else 0.0,
            "mean_frame_ms": float(total_processing / frames * 1000) if frames else 0.0,
            "median_frame_ms": float(np.median(latencies)) if latencies.size else 0.0,
            "p95_frame_ms": float(np.percentile(latencies, 95)) if latencies.size else 0.0,
            "latency_sample_count": int(latencies.size), "output": str(target),
            "mask_dir": str(mask_dir) if mask_dir else None,
        }
        completed = True
        return result
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not completed and target is not None:
            target.unlink(missing_ok=True)
        if not completed and mask_dir_created and mask_dir is not None:
            shutil.rmtree(mask_dir, ignore_errors=True)


def _dataset_manifest_file(dataset: Path, row: dict, field: str, split: str) -> Path:
    relative = row.get(field)
    if relative:
        candidate = (dataset / str(relative)).resolve()
        root = dataset.resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError(f"Prepared dataset manifest contains an unsafe {field} path: {relative}")
        return candidate

    # Backward compatibility for Studio manifests created before format version 2.
    source = Path(row["source_image"])
    prefix = f"{Path(row['collection']).name}__{row['town']}__{source.stem}"
    if field == "prepared_image":
        return dataset / "images" / split / f"{prefix}{source.suffix.lower()}"
    return dataset / "masks" / split / f"{prefix}.png"


def evaluate_dataset(
    model_path: str, prepared_dataset: str, split: str = "test", imgsz: int = 512,
    device: str = "auto", ignore_label: int | None = None, progress=None,
    max_examples: int = 64,
) -> dict:
    dataset = Path(prepared_dataset).expanduser().resolve()
    manifest_path = dataset / "split_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No split_manifest.json in {dataset}; independent test evaluation requires a prepared dataset.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("samples", {}).get(split, [])
    if not rows:
        raise ValueError(f"The {split} split is empty. Choose a prepared {split} split; it will not be replaced with validation data.")
    if max_examples < 0:
        raise ValueError("max_examples cannot be negative.")

    data_yaml_path = dataset / "data.yaml"
    if not data_yaml_path.is_file():
        raise FileNotFoundError(f"Prepared dataset is missing data.yaml: {data_yaml_path}")
    try:
        import yaml
        dataset_spec = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise ValueError(f"Could not read prepared data.yaml: {exc}") from exc
    raw_names = dataset_spec.get("names", {})
    if isinstance(raw_names, list):
        names = {index: str(name) for index, name in enumerate(raw_names)}
    elif isinstance(raw_names, dict):
        names = {int(key): str(value) for key, value in raw_names.items()}
    else:
        raise ValueError("Prepared data.yaml must declare semantic class names.")
    if names != DEFAULT_CLASSES:
        raise ValueError(
            "This RoadForge build expects the four-class semantic map "
            f"{DEFAULT_CLASSES}; found {names}. Re-annotate legacy samples and prepare a new dataset."
        )
    num_classes = len(names)

    engine = SemanticEngine(model_path, imgsz, device)
    total = np.zeros((num_classes, num_classes), dtype=np.int64)
    by_group: dict[str, np.ndarray] = {}
    examples = []
    preview_count = min(max_examples, len(rows))
    preview_indices = set(np.linspace(0, len(rows) - 1, preview_count, dtype=int).tolist()) if preview_count else set()
    run_stamp = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:4]}"
    example_dir = dataset / "evaluation_examples" / f"{split}_{run_stamp}"
    report_dir = dataset / "evaluations"
    report_dir.mkdir(parents=True, exist_ok=True)

    for index, row in enumerate(rows, 1):
        image = _dataset_manifest_file(dataset, row, "prepared_image", split)
        mask_file = _dataset_manifest_file(dataset, row, "prepared_mask", split)
        if not image.is_file() or not mask_file.is_file():
            missing = image if not image.is_file() else mask_file
            raise FileNotFoundError(f"Prepared dataset is incomplete; missing file: {missing}")
        with Image.open(image) as im:
            rgb = np.asarray(im.convert("RGB"))
        with Image.open(mask_file) as im:
            if im.mode != "L":
                raise ValueError(f"Expected a single-channel semantic PNG mask, got {im.mode}: {mask_file}")
            target = np.asarray(im)
        if target.shape != rgb.shape[:2]:
            raise ValueError(f"Image/mask dimensions differ for {image.name}: {rgb.shape[:2]} vs {target.shape}")

        prediction = engine.predict(rgb)
        matrix = confusion_matrix(target, prediction, num_classes, ignore_label)
        total += matrix
        for field in ("town", "weather", "traffic", "collection"):
            key = f"{field}:{row.get(field, 'Unknown')}"
            by_group.setdefault(key, np.zeros((num_classes, num_classes), dtype=np.int64))
            by_group[key] += matrix

        if index - 1 in preview_indices:
            example_dir.mkdir(parents=True, exist_ok=True)
            prefix = str(row.get("sample_id") or image.stem)
            prediction_path = example_dir / f"{prefix}_prediction.png"
            Image.fromarray(colorize(target)).save(example_dir / f"{prefix}_ground_truth.png")
            Image.fromarray(colorize(prediction)).save(prediction_path)
            Image.fromarray(overlay(rgb, target)).save(example_dir / f"{prefix}_gt_overlay.png")
            Image.fromarray(overlay(rgb, prediction)).save(example_dir / f"{prefix}_pred_overlay.png")
            Image.fromarray(np.where(target == prediction, 0, 255).astype(np.uint8)).save(example_dir / f"{prefix}_error.png")
            examples.append({
                "image": str(image), "target": str(mask_file), "prediction": str(prediction_path),
                "town": row.get("town"), "weather": row.get("weather"), "traffic": row.get("traffic"),
            })
        if progress:
            progress(index, len(rows), image.name)

    result = {
        "model": str(Path(model_path).expanduser().resolve()), "dataset": str(dataset), "split": split,
        "samples": len(rows), "ignore_label": ignore_label,
        "metrics": metrics_from_confusion(total, names),
        "by_group": {key: metrics_from_confusion(value, names) for key, value in by_group.items()},
        "examples": examples, "examples_generated": len(examples), "example_limit": max_examples,
        "evaluation_id": run_stamp,
    }
    atomic_json(report_dir / f"{split}_{run_stamp}.json", result)
    with (report_dir / f"{split}_{run_stamp}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["class_id", "class", "iou", "precision", "recall", "dice", "target_pixels", "predicted_pixels"])
        for class_id, row in result["metrics"]["per_class"].items():
            writer.writerow([class_id, row["name"], row["iou"], row["precision"], row["recall"], row["dice"], row["pixels"], row["predicted_pixels"]])
    result["report_json"] = str(report_dir / f"{split}_{run_stamp}.json")
    result["report_csv"] = str(report_dir / f"{split}_{run_stamp}.csv")
    return result


def _onnx_input_spec(input_meta, imgsz: int) -> tuple[tuple[int, int, int, int], np.dtype]:
    shape = list(input_meta.shape)
    if len(shape) != 4:
        raise ValueError(f"Expected a 4D NCHW ONNX input, got {shape}.")
    if shape[1] not in (3, "3"):
        raise ValueError(f"Expected NCHW input with 3 RGB channels, got {shape}.")
    height = int(shape[2]) if isinstance(shape[2], (int, np.integer)) and int(shape[2]) > 0 else int(imgsz)
    width = int(shape[3]) if isinstance(shape[3], (int, np.integer)) and int(shape[3]) > 0 else int(imgsz)
    batch = int(shape[0]) if isinstance(shape[0], (int, np.integer)) and int(shape[0]) > 0 else 1
    if batch != 1:
        raise ValueError(f"Only batch-1 ONNX validation is supported; model input shape is {shape}.")
    dtype_name = str(input_meta.type).lower()
    dtype = np.float16 if "float16" in dtype_name else np.float32
    return (batch, 3, height, width), dtype


def _smoke_test_onnx(session, imgsz: int, expected_classes: int | None = None) -> dict:
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or not outputs:
        raise ValueError(f"Expected one image input and at least one output, got {len(inputs)} input(s) and {len(outputs)} output(s).")
    shape, dtype = _onnx_input_spec(inputs[0], imgsz)
    dummy = np.zeros(shape, dtype=dtype)
    result = session.run(None, {inputs[0].name: dummy})
    if not result:
        raise RuntimeError("ONNX Runtime returned no output from the smoke inference.")
    raw = np.asarray(result[0])
    if not np.all(np.isfinite(raw)):
        raise RuntimeError("ONNX smoke inference produced NaN or infinite values.")
    decoded = _semantic_array_to_mask(raw, (shape[2], shape[3]))
    class_ids = sorted(map(int, np.unique(decoded)))
    if expected_classes is not None and class_ids and class_ids[-1] >= expected_classes:
        raise RuntimeError(f"ONNX output contains class ID {class_ids[-1]}, but the source checkpoint declares {expected_classes} classes.")
    return {"input_shape_used": list(shape), "output_shape": list(raw.shape), "smoke_test": "passed", "classes_observed_on_zero_input": class_ids}



def _read_avis_classifier_checkpoint(path: str | Path) -> dict | None:
    """Safely identify RoadForge's self-describing AVIS classification checkpoints."""
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # Ultralytics checkpoints may contain serialized model classes and are
        # handled by the existing Ultralytics exporter below.
        return None
    if not isinstance(payload, dict):
        return None
    if not (
        isinstance(payload.get("backbone"), str)
        and isinstance(payload.get("tasks"), list)
        and isinstance(payload.get("state_dict"), dict)
    ):
        return None
    return payload


def _export_avis_classifier_onnx(
    source: Path,
    payload: dict,
    opset: int = 17,
    dynamic: bool = False,
    simplify: bool = False,
    output_dir: str | Path | None = None,
) -> dict:
    """Export a RoadForge AVIS multi-head image classifier to a validated ONNX graph."""
    import tempfile

    import onnx
    import onnxruntime as ort
    import torch
    import torch.nn as nn

    from ..classification_training import build_multitask_classifier

    backbone = str(payload["backbone"]).strip().lower()
    tasks = payload["tasks"]
    if backbone not in {"resnet18", "efficientnet_b0"}:
        raise ValueError(f"Unsupported AVIS classification backbone in checkpoint: {backbone!r}.")
    if not tasks:
        raise ValueError("AVIS classification checkpoint contains no task heads.")

    input_size = int(payload.get("input_size", 224))
    if input_size < 32 or input_size > 4096:
        raise ValueError(f"Invalid classification input_size in checkpoint: {input_size}.")

    mean = [float(value) for value in payload.get("mean", (0.485, 0.456, 0.406))]
    std = [float(value) for value in payload.get("std", (0.229, 0.224, 0.225))]
    if len(mean) != 3 or len(std) != 3 or not np.all(np.isfinite(mean + std)) or any(value <= 0 for value in std):
        raise ValueError("Classification checkpoint must contain three finite RGB normalization means and positive standard deviations.")

    task_outputs = []
    output_names = []
    used_names = set()
    normalized_tasks = []
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError(f"Invalid task schema at index {index}.")
        key = str(task.get("key", "")).strip()
        name = str(task.get("name", key or f"Task {index + 1}")).strip()
        classes = task.get("classes")
        if not key or not isinstance(classes, list) or len(classes) < 2:
            raise ValueError(f"Task {index + 1} must declare a non-empty key and at least two classes.")
        classes = [str(value).strip() for value in classes]
        if any(not value for value in classes) or len(set(classes)) != len(classes):
            raise ValueError(f"Task {key!r} contains empty or duplicate class names.")
        safe_key = re.sub(r"[^A-Za-z0-9_]+", "_", key).strip("_") or f"task_{index}"
        output_name = f"logits_{safe_key}"
        if output_name in used_names:
            output_name = f"{output_name}_{index}"
        used_names.add(output_name)
        output_names.append(output_name)
        normalized = {"key": key, "name": name, "classes": classes}
        normalized_tasks.append(normalized)
        task_outputs.append({**normalized, "output_name": output_name, "output_index": index})

    output_root = Path(output_dir).expanduser().resolve() if output_dir else source.parent / "exports"
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / (
        f"{source.stem}_avis_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:5]}.onnx"
    )

    model = build_multitask_classifier(backbone, normalized_tasks, pretrained=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval().cpu()

    class ONNXExportWrapper(nn.Module):
        def __init__(self, classifier):
            super().__init__()
            self.classifier = classifier

        def forward(self, images):
            return tuple(self.classifier(images))

    wrapper = ONNXExportWrapper(model).eval()
    dummy = torch.linspace(
        -1.0, 1.0, steps=3 * input_size * input_size, dtype=torch.float32
    ).reshape(1, 3, input_size, input_size)
    dynamic_axes = None
    if dynamic:
        # The model is trained at a fixed spatial resolution. Only batch is dynamic.
        dynamic_axes = {"images": {0: "batch"}}
        dynamic_axes.update({name: {0: "batch"} for name in output_names})

    try:
        with tempfile.TemporaryDirectory(prefix="roadforge_avis_export_", dir=output_root) as temp_root:
            graph_path = Path(temp_root) / "classification.onnx"
            torch.onnx.export(
                wrapper,
                dummy,
                str(graph_path),
                export_params=True,
                opset_version=int(opset),
                do_constant_folding=True,
                input_names=["images"],
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                dynamo=False,  # Keep dynamic_axes semantics stable across newer PyTorch defaults.
            )

            graph = onnx.load(str(graph_path))
            if simplify:
                try:
                    from onnxsim import simplify as simplify_onnx
                except ImportError as exc:
                    raise RuntimeError(
                        "Graph simplification for AVIS classifiers requires onnxsim. "
                        "Install it with 'pip install onnxsim', or uncheck Simplify graph."
                    ) from exc
                graph, is_valid = simplify_onnx(graph)
                if not is_valid:
                    raise RuntimeError("onnxsim reported that the simplified AVIS graph is not equivalent.")
            metadata = {
                "task": "classification",
                "task_type": "classification",
                "roadforge_format": "avis_multitask_classification_v1",
                "backbone": backbone,
                "input_size": str(input_size),
                "input_layout": "NCHW",
                "input_dtype": "float32",
                "color_order": "RGB",
                "pixel_range_before_normalization": "[0, 1]",
                "normalization_mean": json.dumps(mean, ensure_ascii=False),
                "normalization_std": json.dumps(std, ensure_ascii=False),
                "resize": f"Direct RGB resize to {input_size}x{input_size}; no letterbox; bilinear interpolation.",
                "output_format": "Raw per-task logits. Apply softmax per output for probabilities; argmax selects the class index.",
                "tasks": json.dumps(task_outputs, ensure_ascii=False),
            }
            onnx.helper.set_model_props(graph, metadata)
            onnx.checker.check_model(graph)
            onnx.save(graph, str(graph_path))

            session = ort.InferenceSession(str(graph_path), providers=["CPUExecutionProvider"])
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            if len(inputs) != 1 or len(outputs) != len(task_outputs):
                raise RuntimeError(
                    f"Expected one image input and {len(task_outputs)} task outputs; "
                    f"got {len(inputs)} input(s) and {len(outputs)} output(s)."
                )
            if inputs[0].name != "images":
                raise RuntimeError(f"Unexpected ONNX input name: {inputs[0].name!r}.")
            if [item.name for item in outputs] != output_names:
                raise RuntimeError("ONNX graph output names/order do not match the checkpoint task schema.")

            with torch.inference_mode():
                torch_outputs = wrapper(dummy)
            ort_outputs = session.run(output_names, {"images": dummy.numpy()})
            parity_rows = []
            for task_output, expected_tensor, actual_value, output_meta in zip(
                task_outputs, torch_outputs, ort_outputs, outputs
            ):
                expected = expected_tensor.detach().cpu().numpy()
                actual = np.asarray(actual_value)
                expected_shape = (1, len(task_output["classes"]))
                if actual.shape != expected_shape or tuple(expected.shape) != expected_shape:
                    raise RuntimeError(
                        f"Task {task_output['key']!r} output has an unexpected shape: "
                        f"PyTorch={expected.shape}, ONNX={actual.shape}, expected={expected_shape}."
                    )
                if not np.all(np.isfinite(actual)):
                    raise RuntimeError(f"ONNX output for task {task_output['key']!r} contains NaN or infinity.")
                max_abs_diff = float(np.max(np.abs(expected - actual)))
                parity_passed = bool(np.allclose(expected, actual, rtol=1e-3, atol=1e-4))
                parity_rows.append({
                    "task": task_output["key"],
                    "output_name": output_meta.name,
                    "shape": list(actual.shape),
                    "max_absolute_difference": max_abs_diff,
                    "allclose": parity_passed,
                })
                if not parity_passed:
                    raise RuntimeError(
                        f"PyTorch/ONNX parity failed for task {task_output['key']!r} "
                        f"(maximum absolute difference {max_abs_diff:.6g})."
                    )

            onnx.checker.check_model(onnx.load(str(graph_path)))
            os.replace(graph_path, target)

        return {
            "path": str(target.resolve()),
            "size_bytes": target.stat().st_size,
            "task": "classification",
            "backbone": backbone,
            "input_size": input_size,
            "input_shape_used": [1, 3, input_size, input_size],
            "dynamic_batch": bool(dynamic),
            "dynamic_spatial": False,
            "simplified": bool(simplify),
            "task_outputs": task_outputs,
            "inputs": [{"name": item.name, "shape": item.shape, "type": item.type} for item in inputs],
            "outputs": [{"name": item.name, "shape": item.shape, "type": item.type} for item in outputs],
            "providers": session.get_providers(),
            "onnx_checker": "passed",
            "smoke_test": "passed",
            "pytorch_onnx_parity": parity_rows,
            "metadata": metadata,
        }
    except Exception:
        target.unlink(missing_ok=True)
        raise

def export_onnx(checkpoint: str, imgsz: int = 512, opset: int = 17, dynamic: bool = False, simplify: bool = False, output_dir: str | Path | None = None) -> dict:
    import tempfile
    source = Path(checkpoint).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() != ".pt":
        raise ValueError("ONNX export requires a local PyTorch .pt checkpoint.")
    if imgsz < 32 or opset < 12:
        raise ValueError("Image size must be at least 32 and ONNX opset must be 12 or newer.")

    # AVIS Classification Lab checkpoints are self-describing PyTorch state dictionaries.
    # Route them to the native Torch exporter; the existing Ultralytics semantic path stays intact.
    avis_payload = _read_avis_classifier_checkpoint(source)
    if avis_payload is not None:
        return _export_avis_classifier_onnx(
            source, avis_payload, opset=opset, dynamic=dynamic,
            simplify=simplify, output_dir=output_dir,
        )

    output_root = Path(output_dir).expanduser().resolve() if output_dir else source.parent / "exports"
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / f"{source.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:5]}.onnx"
    try:
        # Export from a temporary checkpoint copy so the original .pt remains untouched.
        with tempfile.TemporaryDirectory(prefix="roadforge_export_") as temp_root:
            temp_checkpoint = Path(temp_root) / source.name
            shutil.copy2(source, temp_checkpoint)
            model = _ultralytics_yolo(temp_checkpoint, "semantic")
            expected_classes = len(getattr(model, "names", {}) or {}) or None
            exported = model.export(format="onnx", imgsz=imgsz, opset=opset, dynamic=dynamic, simplify=simplify, task="semantic")
            exported_path = Path(exported)
            if not exported_path.is_file():
                raise FileNotFoundError(f"Ultralytics reported an export path that does not exist: {exported_path}")
            shutil.copy2(exported_path, target)

        import onnx
        graph = onnx.load(str(target))
        onnx.checker.check_model(graph)
        import onnxruntime as ort
        session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
        smoke = _smoke_test_onnx(session, imgsz, expected_classes)
        return {
            "path": str(target.resolve()), "size_bytes": target.stat().st_size,
            "inputs": [{"name": x.name, "shape": x.shape, "type": x.type} for x in session.get_inputs()],
            "outputs": [{"name": x.name, "shape": x.shape, "type": x.type} for x in session.get_outputs()],
            "providers": session.get_providers(), "onnx_checker": "passed",
            "expected_classes": expected_classes, **smoke,
        }
    except Exception:
        target.unlink(missing_ok=True)
        raise


def _timed_predictions(engine: SemanticEngine, rgb: np.ndarray, warmup: int = 3, iterations: int = 10) -> tuple[np.ndarray, np.ndarray]:
    if warmup < 0 or iterations < 1:
        raise ValueError("Warm-up must be non-negative and iterations must be at least one.")
    for _ in range(warmup):
        engine.predict(rgb)
    engine.synchronize()
    timings = []
    prediction = None
    for _ in range(iterations):
        engine.synchronize()
        start = time.perf_counter()
        prediction = engine.predict(rgb)
        engine.synchronize()
        timings.append((time.perf_counter() - start) * 1000)
    return prediction, np.asarray(timings, dtype=np.float64)


def compare_models(torch_model: str, onnx_model: str, image_path: str, imgsz: int = 512, device: str = "auto") -> dict:
    with Image.open(image_path) as im:
        rgb = np.asarray(im.convert("RGB"))

    torch_load_start = time.perf_counter()
    torch_engine = SemanticEngine(torch_model, imgsz, device)
    torch_load_ms = (time.perf_counter() - torch_load_start) * 1000
    onnx_load_start = time.perf_counter()
    onnx_engine = SemanticEngine(onnx_model, imgsz, "cpu")
    onnx_load_ms = (time.perf_counter() - onnx_load_start) * 1000

    torch_mask, torch_times = _timed_predictions(torch_engine, rgb, warmup=3, iterations=10)
    onnx_mask, onnx_times = _timed_predictions(onnx_engine, rgb, warmup=3, iterations=10)
    # Include all four classes even when the selected parity image contains
    # no vehicles. This catches accidental three-class exports.
    matrix = confusion_matrix(torch_mask, onnx_mask, num_classes=len(DEFAULT_CLASSES))
    parity = metrics_from_confusion(matrix)
    return {
        "image": str(Path(image_path).resolve()), "agreement": float(np.mean(torch_mask == onnx_mask)),
        "torch_model": str(Path(torch_model).resolve()), "onnx_model": str(Path(onnx_model).resolve()),
        "torch_device": torch_engine.device, "onnx_runtime_providers": _onnx_providers(onnx_engine),
        "torch_model_load_ms": torch_load_ms, "onnx_model_load_ms": onnx_load_ms,
        "torch_inference_ms": float(torch_times.mean()), "torch_median_ms": float(np.median(torch_times)),
        "onnx_inference_ms": float(onnx_times.mean()), "onnx_median_ms": float(np.median(onnx_times)),
        "iterations": 10, "warmup_iterations": 3, "timing_scope": "predict() end-to-end, excluding model loading",
        "torch_classes": sorted(map(int, np.unique(torch_mask))),
        "onnx_classes": sorted(map(int, np.unique(onnx_mask))),
        "prediction_parity_metrics": parity,
    }


def _onnx_providers(engine: SemanticEngine) -> list[str]:
    """Best-effort retrieval of the providers used by an Ultralytics ONNX predictor."""
    try:
        predictor = getattr(engine.model, "predictor", None)
        backend = getattr(predictor, "model", None)
        session = getattr(backend, "session", None) or getattr(backend, "ort_session", None)
        if session is not None and hasattr(session, "get_providers"):
            return list(session.get_providers())
    except Exception:
        pass
    return ["unknown; inspect the runtime backend"]


def benchmark_model(model_path: str, image_path: str, imgsz: int = 512, device: str = "auto", warmup: int = 3, iterations: int = 20) -> dict:
    with Image.open(image_path) as im:
        rgb = np.asarray(im.convert("RGB"))
    load_start = time.perf_counter()
    engine = SemanticEngine(model_path, imgsz, device)
    model_load_ms = (time.perf_counter() - load_start) * 1000
    _, values = _timed_predictions(engine, rgb, warmup=warmup, iterations=iterations)
    mean_ms = float(values.mean())
    return {
        "model": str(Path(model_path).expanduser().resolve()), "device": engine.device,
        "onnx_runtime_providers": _onnx_providers(engine) if Path(model_path).suffix.lower() == ".onnx" else None,
        "image_size": list(rgb.shape[:2]), "warmup": warmup, "iterations": len(values),
        "model_load_ms": model_load_ms, "mean_ms": mean_ms, "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)), "min_ms": float(values.min()), "max_ms": float(values.max()),
        "fps": float(1000 / mean_ms) if mean_ms > 0 else 0.0,
        "timing_scope": "model.predict() end-to-end, including preprocessing and postprocessing; excludes model loading",
    }


def create_deployment_package(model_path: str, output_dir: str, config: dict | None = None, evaluation_report: dict | None = None, benchmark_report: dict | None = None) -> dict:
    source = Path(model_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() != ".onnx":
        raise ValueError("Deployment packaging requires a validated ONNX model.")

    target = Path(output_dir).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Deployment destination already exists: {target}")

    raw_classes = (config or {}).get("model", {}).get("classes", DEFAULT_CLASSES)
    classes = {int(key): str(value) for key, value in dict(raw_classes).items()}
    if not classes or sorted(classes) != list(range(len(classes))):
        raise ValueError("Deployment class IDs must be contiguous and start at 0.")

    target.mkdir(parents=True, exist_ok=False)
    try:
        copied = target / "model.onnx"
        shutil.copy2(source, copied)
        import onnxruntime as ort
        session = ort.InferenceSession(str(copied), providers=["CPUExecutionProvider"])
        input_meta = session.get_inputs()[0]
        output_meta = session.get_outputs()
        smoke = _smoke_test_onnx(session, 512, expected_classes=len(classes))
        shapes = {
            "input": {"name": input_meta.name, "shape": input_meta.shape, "type": input_meta.type},
            "outputs": [{"name": item.name, "shape": item.shape, "type": item.type} for item in output_meta],
        }
        input_shape, _ = _onnx_input_spec(input_meta, 512)
        atomic_json(target / "classes.json", {str(key): value for key, value in classes.items()})
        atomic_json(target / "preprocessing.json", {
            "input_layout": "NCHW", "input_dtype": input_meta.type,
            "input_range": [0.0, 1.0], "color_order": "RGB",
            "resize": "Aspect-preserving letterbox with bilinear image resizing and RGB padding value 114.",
            "postprocessing": "The script decodes semantic logits with argmax (or accepts a discrete class map), removes letterbox padding, then resizes class IDs with nearest-neighbor interpolation.",
            "onnx_metadata": shapes,
            "dynamic_spatial_dimensions": not (isinstance(input_meta.shape[2], int) and isinstance(input_meta.shape[3], int)),
            "dynamic_default_size": 512,
        })
        atomic_json(target / "evaluation_report.json", evaluation_report or {"status": "not run", "note": "No evaluation report was attached to this package."})
        atomic_json(target / "benchmark_report.json", benchmark_report or {"status": "not run", "note": "No benchmark report was attached to this package."})
        metadata = {
            "format_version": 2, "model": copied.name, "sha256": file_hash(copied),
            "task": "semantic", "classes": {str(key): value for key, value in classes.items()},
            "config": snapshot_config(config) if config else None,
            "created_at": datetime.now().astimezone().isoformat(), "onnx_metadata": shapes,
            "smoke_test": smoke,
        }
        atomic_json(target / "metadata.json", metadata)
        script = _DEPLOYMENT_INFERENCE_SCRIPT.replace("DEFAULT_IMGSZ = 512", f"DEFAULT_IMGSZ = {max(input_shape[2], input_shape[3])}")
        (target / "inference.py").write_text(script, encoding="utf-8")
        (target / "requirements.txt").write_text("numpy>=1.26,<3\nPillow>=10,<13\nonnxruntime>=1.17,<2\n", encoding="utf-8")
        (target / "README.md").write_text(
            "# Road Perception Model Package\n\n"
            "Contains the exported semantic ONNX graph, class map, graph metadata, preprocessing details, and a standalone inference script.\n\n"
            "Install `requirements.txt`, then run `python inference.py input.png output_mask.png --overlay output_overlay.png`. "
            "For dynamic-spatial ONNX models, pass `--imgsz 512` (or another square size accepted by the model). "
            "The script uses aspect-preserving letterboxing and correctly removes padding before mapping class IDs back to the source image.\n",
            encoding="utf-8",
        )
        return {"path": str(target), "model": str(copied), "sha256": file_hash(copied), "smoke_test": smoke}
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise


_DEPLOYMENT_INFERENCE_SCRIPT = r'''# Standalone semantic ONNX inference for this deployment package.
import argparse
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image

DEFAULT_IMGSZ = 512


def letterbox(rgb, target_w, target_h):
    h, w = rgb.shape[:2]
    ratio = min(target_w / w, target_h / h)
    nw, nh = max(1, round(w * ratio)), max(1, round(h * ratio))
    resized = np.asarray(Image.fromarray(rgb).resize((nw, nh), Image.Resampling.BILINEAR))
    dw, dh = target_w - nw, target_h - nh
    left, top = round(dw / 2 - 0.1), round(dh / 2 - 0.1)
    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, (left, top, nw, nh)


def decode(output, target_hw):
    out = np.asarray(output)
    if out.ndim == 4:
        if out.shape[0] != 1:
            raise ValueError(f"Expected one output batch, got {out.shape}")
        out = out[0]
    if out.ndim == 3:
        if out.shape[0] == 1:
            out = out[0]  # batch-wrapped discrete class map
        elif out.shape[-1] == 1:
            out = out[..., 0]
        else:
            first_channels = out.shape[0] > 1 and out.shape[0] < min(out.shape[1:])
            last_channels = out.shape[-1] > 1 and out.shape[-1] < min(out.shape[:2])
            if first_channels:
                channels = [out[i] for i in range(out.shape[0])]
                axis = 0
            elif last_channels:
                channels = [out[..., i] for i in range(out.shape[-1])]
                axis = -1
            else:
                raise ValueError(f"Unsupported semantic output layout: {out.shape}")
            target_h, target_w = target_hw
            resized = [
                np.asarray(Image.fromarray(np.asarray(channel, dtype=np.float32), mode="F").resize(
                    (target_w, target_h), Image.Resampling.BILINEAR
                ))
                for channel in channels
            ]
            out = np.stack(resized, axis=axis).argmax(axis=axis)
    if out.ndim != 2:
        raise ValueError(f"Unsupported semantic output shape: {np.asarray(output).shape}")
    if np.issubdtype(out.dtype, np.floating):
        if not np.all(np.isfinite(out)) or not np.allclose(out, np.rint(out), atol=1e-4):
            raise ValueError("Two-dimensional output must contain finite discrete class IDs.")
        out = np.rint(out)
    if out.size and (out.min() < 0 or out.max() > 255):
        raise ValueError(f"Class IDs do not fit in uint8: [{out.min()}, {out.max()}]")
    mask = out.astype(np.uint8, copy=False)
    target_h, target_w = target_hw
    if mask.shape != (target_h, target_w):
        mask = np.asarray(Image.fromarray(mask).resize((target_w, target_h), Image.Resampling.NEAREST))
    return mask


def main():
    parser = argparse.ArgumentParser(description="Run semantic segmentation with the bundled ONNX model.")
    parser.add_argument("image")
    parser.add_argument("mask")
    parser.add_argument("--overlay")
    parser.add_argument("--model", default=str(Path(__file__).with_name("model.onnx")))
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="Square input size for dynamic-spatial models.")
    args = parser.parse_args()
    if args.imgsz < 32:
        parser.error("--imgsz must be at least 32")

    source = np.asarray(Image.open(args.image).convert("RGB"))
    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    if len(session.get_inputs()) != 1:
        raise ValueError("Expected exactly one ONNX input.")
    inp = session.get_inputs()[0]
    shape = inp.shape
    if len(shape) != 4 or shape[1] not in (3, "3"):
        raise ValueError(f"Expected a 4D NCHW RGB input; got {shape}.")
    ih = int(shape[2]) if isinstance(shape[2], int) and shape[2] > 0 else args.imgsz
    iw = int(shape[3]) if isinstance(shape[3], int) and shape[3] > 0 else args.imgsz
    padded, (left, top, nw, nh) = letterbox(source, iw, ih)
    dtype = np.float16 if "float16" in str(inp.type).lower() else np.float32
    tensor = padded.transpose(2, 0, 1)[None].astype(dtype) / 255.0
    outputs = session.run(None, {inp.name: tensor})
    if not outputs:
        raise RuntimeError("ONNX model returned no outputs.")
    predicted = decode(outputs[0], (ih, iw))
    predicted = predicted[top:top + nh, left:left + nw]
    predicted = np.asarray(Image.fromarray(predicted).resize((source.shape[1], source.shape[0]), Image.Resampling.NEAREST))
    Image.fromarray(predicted).save(args.mask)

    if args.overlay:
        colors = np.zeros((*predicted.shape, 3), dtype=np.uint8)
        colors[predicted == 1] = (40, 210, 120)
        colors[predicted == 2] = (255, 210, 40)
        active = predicted != 0
        result = source.copy()
        result[active] = np.rint(source[active] * 0.58 + colors[active] * 0.42).astype(np.uint8)
        Image.fromarray(result).save(args.overlay)


if __name__ == "__main__":
    main()
'''
