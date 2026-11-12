"""Offline YOLOPv2 annotation for existing DataCollector collections."""
import hashlib
import json
import os
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
INPUT_WIDTH, INPUT_HEIGHT = 640, 384  # fallback for dynamic ONNX input shapes


def discover_collections(dataset_root, stop_event=None):
    """Return collection -> town -> original image paths for actual raw trees."""
    result = {}
    root = Path(dataset_root)
    if not root.is_dir():
        return result
    for collection in sorted(root.glob("collection_*")):
        if stop_event is not None and stop_event.is_set():
            break
        if not collection.is_dir():
            continue
        towns = {}
        raw = collection / "raw"
        if raw.is_dir():
            for town_dir in sorted(p for p in raw.iterdir() if p.is_dir()):
                if stop_event is not None and stop_event.is_set():
                    break
                image_dir = town_dir / "images"
                if image_dir.is_dir():
                    images = sorted(p for p in image_dir.iterdir()
                                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
                    if images:
                        towns[town_dir.name] = images
        result[collection] = towns
    return result


def _atomic_png(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.png")
    try:
        if not cv2.imwrite(str(temp), image):
            raise OSError(f"Could not encode PNG: {path}")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
    os.replace(temp, path)


def _sha256_file(path, chunk_size=1024 * 1024):
    """Hash large models without loading the entire file into RAM."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def annotation_outputs_valid(collection, record):
    """Check that all committed masks decode, align, and use the documented values."""
    try:
        expected = (int(record["height"]), int(record["width"]))
        decoded = []
        for key in ("drivable_mask", "lane_mask", "combined_mask"):
            path = Path(collection) / record[key]
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != expected:
                return False
            decoded.append(mask)
        drive, lane, combined = decoded
        return (set(np.unique(drive)).issubset({0, 255})
                and set(np.unique(lane)).issubset({0, 255})
                and set(np.unique(combined)).issubset({0, 1, 2})
                and bool(np.all(combined[lane > 0] == 2)))
    except (KeyError, TypeError, ValueError, OSError):
        return False


class CollectionScanWorker(QThread):
    """Find collections and verify existing teacher outputs off the UI thread."""
    result = Signal(object)
    progress = Signal(str)

    def __init__(self, dataset_root, parent=None):
        super().__init__(parent)
        self.dataset_root = Path(dataset_root)
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        try:
            self.progress.emit("Discovering collection folders…")
            collections = discover_collections(self.dataset_root, self._stop)
            summaries = {}
            for collection_index, (collection, towns) in enumerate(collections.items(), start=1):
                if self._stop.is_set():
                    break
                self.progress.emit(f"Scanning {collection.name} ({collection_index}/{len(collections)})…")
                manifest_path = collection / "annotation_manifest.json"
                try:
                    data = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"images": {}}
                    entries = data.get("images", {})
                    if not isinstance(entries, dict):
                        entries = {}
                except Exception:
                    entries = {}

                summaries[collection] = {}
                for town, images in towns.items():
                    if self._stop.is_set():
                        break
                    self.progress.emit(f"Checking labels · {collection.name}/{town} ({len(images)} images)…")
                    done = failed = 0
                    for image in images:
                        if self._stop.is_set():
                            break
                        record = entries.get(image.relative_to(collection).as_posix(), {})
                        if record.get("status") == "success" and annotation_outputs_valid(collection, record):
                            done += 1
                        elif record.get("status") == "failed":
                            failed += 1
                    summaries[collection][town] = {"done": done, "failed": failed}

            self.result.emit({"collections": collections, "summaries": summaries,
                              "stopped": self._stop.is_set()})
        except Exception:
            self.result.emit({"error": traceback.format_exc()})


def create_session(model_path, provider="Auto"):
    import onnxruntime as ort
    available = ort.get_available_providers()
    wanted = ["CUDAExecutionProvider", "CPUExecutionProvider"] if provider in ("Auto", "CUDA") else ["CPUExecutionProvider"]
    if provider == "CUDA" and "CUDAExecutionProvider" not in available:
        raise RuntimeError(f"CUDA requested but unavailable. Installed providers: {available}")
    chosen = [p for p in wanted if p in available]
    if not chosen:
        chosen = ["CPUExecutionProvider"]
    fallback_message = ""
    try:
        session = ort.InferenceSession(str(model_path), providers=chosen)
    except Exception as exc:
        if "CUDAExecutionProvider" in chosen and provider != "CUDA":
            fallback_message = f"CUDA session initialization failed ({exc}); using CPUExecutionProvider."
            session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        else:
            raise RuntimeError(f"Could not initialize {chosen}: {exc}") from exc
    actual = session.get_providers()[0]
    inp = session.get_inputs()[0]
    shape = inp.shape
    width = shape[3] if isinstance(shape[3], int) else INPUT_WIDTH
    height = shape[2] if isinstance(shape[2], int) else INPUT_HEIGHT
    return session, actual, (int(width), int(height)), available, fallback_message


class YOLOPv2Predictor:
    """Adapt the ONNX preprocessing and mask extraction from Kazuhito00/YOLOPv2-ONNX-Sample."""
    def __init__(self, session, input_size):
        self.session = session
        self.input_width, self.input_height = input_size
        self.input_name = session.get_inputs()[0].name
        self.outputs = session.get_outputs()
        # Verified model outputs: 0 detection heads, 1-3 anchors, 4 drive(2ch), 5 lane(1ch).
        if len(self.outputs) != 6:
            raise RuntimeError(f"Expected verified YOLOPv2 six-output graph, found {len(self.outputs)}")

    def predict(self, rgb):
        h, w = rgb.shape[:2]
        # The reference implementation receives BGR, resizes directly (no letterbox), converts to RGB,
        # scales [0,1], then uses NCHW float32.
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        image = cv2.resize(bgr, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = np.expand_dims(np.transpose(image, (2, 0, 1)), 0)
        outputs = self.session.run(None, {self.input_name: tensor})
        if len(outputs) != 6:
            raise RuntimeError(f"Model returned {len(outputs)} outputs; expected 6")

        # Reference implementation: utils.utils_onnx.driving_area_mask: 1 - drive[0][0].
        drive_output = np.asarray(outputs[4], dtype=np.float32)
        if drive_output.ndim != 4 or drive_output.shape[:2] != (1, 2):
            raise RuntimeError(f"Drivable output shape mismatch: {drive_output.shape}")
        drive_prob = 1.0 - drive_output[0, 0]
        # Reference implementation: city_vision_processing._extract_prob_map(outputs, 5)
        lane_prob = np.squeeze(np.asarray(outputs[5], dtype=np.float32))
        if lane_prob.ndim != 2:
            raise RuntimeError(f"Lane output shape mismatch: {np.asarray(outputs[5]).shape}")
        if drive_prob.shape != (self.input_height, self.input_width) or lane_prob.shape != drive_prob.shape:
            raise RuntimeError(f"Segmentation spatial shape mismatch: drive={drive_prob.shape}, lane={lane_prob.shape}")
        # Outputs 4/5 are already sigmoid probabilities (verified model range [0,1]).
        drive = (drive_prob >= 0.50).astype(np.uint8) * 255
        lane = (lane_prob > 0.50).astype(np.uint8) * 255
        # Preserve the reference lane-mask morphology.
        kernel = np.ones((3, 3), np.uint8)
        lane = cv2.morphologyEx(lane, cv2.MORPH_OPEN, kernel, iterations=1)
        lane = cv2.morphologyEx(lane, cv2.MORPH_CLOSE, kernel, iterations=1)
        if (w, h) != (self.input_width, self.input_height):
            drive = cv2.resize(drive, (w, h), interpolation=cv2.INTER_NEAREST)
            lane = cv2.resize(lane, (w, h), interpolation=cv2.INTER_NEAREST)
        combined = np.zeros((h, w), dtype=np.uint8)
        combined[drive > 0] = 1
        combined[lane > 0] = 2
        return drive, lane, combined


def visualization(rgb, drive, lane, combined, opacity=0.42):
    overlay = rgb.copy()
    overlay[drive > 0] = (0, 255, 0)
    overlay[lane > 0] = (255, 220, 0)
    blended = cv2.addWeighted(rgb, 1.0 - opacity, overlay, opacity, 0)
    colors = np.zeros_like(rgb)
    colors[combined == 1] = (0, 200, 0)
    colors[combined == 2] = (255, 220, 0)
    combined_overlay = cv2.addWeighted(rgb, 0.58, colors, 0.42, 0)
    return {"Original RGB": rgb, "Drivable Area": cv2.cvtColor(drive, cv2.COLOR_GRAY2RGB),
            "Lane Lines": cv2.cvtColor(lane, cv2.COLOR_GRAY2RGB),
            "Combined classes": colors, "Combined overlay": blended if not np.any(combined == 2) else combined_overlay}


class AnnotationWorker(QThread):
    progress = Signal(dict)
    preview = Signal(object)
    log = Signal(str)
    done = Signal(dict)

    def __init__(self, jobs, model_path, provider="Auto", save_visualizations=True,
                 reprocess=False, preview_interval=1):
        super().__init__()
        self.jobs = jobs
        self.model_path = Path(model_path)
        self.provider = provider
        self.save_visualizations = save_visualizations
        self.reprocess = reprocess
        self.preview_interval = max(1, preview_interval)
        self._pause = threading.Event()
        self._pause.set()
        self._stop = threading.Event()

    def pause(self): self._pause.clear()
    def resume(self): self._pause.set()
    def stop(self): self._stop.set(); self._pause.set()

    def run(self):
        start = time.monotonic()
        stats = {"total": 0, "success": 0, "remaining": 0, "skipped": 0, "failed": 0, "provider": "—"}
        try:
            self.log.emit(f"Loading YOLOPv2 model: {self.model_path}")
            session, active, size, available, fallback_message = create_session(self.model_path, self.provider)
            predictor = YOLOPv2Predictor(session, size)
            model_hash = _sha256_file(self.model_path)
            stats["provider"] = active
            output_summary = "; ".join(f"{o.name} {o.shape}" for o in session.get_outputs())
            stats["model_info"] = f"Loaded · input {size[0]}×{size[1]} · outputs: {output_summary}"
            records = []
            manifests = {}
            self.log.emit(f"Preparing {len(self.jobs):,} selected images…")
            for collection, town, image in self.jobs:
                if self._stop.is_set():
                    stats["stopped"] = True
                    self.done.emit(stats)
                    return
                manifest_path = collection / "annotation_manifest.json"
                if collection not in manifests:
                    if manifest_path.exists():
                        with manifest_path.open("r", encoding="utf-8") as stream:
                            manifests[collection] = json.load(stream)
                    else:
                        manifests[collection] = {"format": "yolopv2-annotations-v1", "images": {}}
                manifest = manifests[collection]
                key = image.relative_to(collection).as_posix()
                entry = manifest["images"].get(key, {})
                valid = (entry.get("status") == "success"
                         and annotation_outputs_valid(collection, entry))
                records.append((collection, town, image, key, valid, entry.get("status")))
            stats["total"] = len(records)
            stats["skipped"] = sum(1 for record in records if record[4] and not self.reprocess)
            stats["remaining"] = stats["total"] - stats["skipped"]
            scopes = {}
            for collection, town, image_path, key, valid, previous_status in records:
                scope_key = f"{collection.name}|{town}"
                row = scopes.setdefault(scope_key, {"total": 0, "success": 0, "failed": 0})
                row["total"] += 1
                if valid and not self.reprocess:
                    row["success"] += 1
                elif not self.reprocess and previous_status == "failed":
                    row["failed"] += 1
            stats["scopes"] = scopes
            self.log.emit(f"ONNX Runtime providers: {available}; active: {active}; input: {size[0]}×{size[1]}; outputs: {output_summary}")
            if fallback_message:
                self.log.emit(fallback_message)
            self.progress.emit(dict(stats, current="", elapsed=0, rate=0))
            for collection, town, image_path, key, valid, previous_status in records:
                if self._stop.is_set(): break
                self._pause.wait()
                if self._stop.is_set(): break
                if valid and not self.reprocess:
                    self.log.emit(f"Skipped completed image: {key}")
                    continue
                manifest_path = collection / "annotation_manifest.json"
                manifest = manifests[collection]
                outroot = collection / "labels_yolopv2" / town
                stem = image_path.stem
                paths = {"drivable": outroot / "drivable_area" / f"{stem}.png",
                         "lane": outroot / "lane_lines" / f"{stem}.png",
                         "combined": outroot / "combined_3class" / f"{stem}.png"}
                images_record = manifest.setdefault("images", {})
                item = images_record.setdefault(key, {})
                if not self.reprocess and previous_status == "failed":
                    scopes[f"{collection.name}|{town}"]["failed"] -= 1
                item.update({"original_image": key, "status": "processing", "error": None})
                _atomic_json(manifest_path, manifest)
                self.log.emit(f"Annotating {collection.name}/{key}")
                try:
                    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                    if bgr is None: raise ValueError("Image could not be decoded by OpenCV")
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    h, w = rgb.shape[:2]
                    drive, lane, combined = predictor.predict(rgb)
                    if drive.shape != (h, w) or lane.shape != (h, w) or combined.shape != (h, w):
                        raise ValueError("Output masks do not match original dimensions")
                    _atomic_png(paths["drivable"], drive)
                    _atomic_png(paths["lane"], lane)
                    _atomic_png(paths["combined"], combined)
                    if self.save_visualizations:
                        views = visualization(rgb, drive, lane, combined)
                        visdir = collection / "labels_yolopv2" / town / "visualizations"
                        for name, arr in views.items():
                            if name == "Original RGB": continue
                            bgr_view = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                            _atomic_png(visdir / f"{stem}_{name.lower().replace(' ', '_')}.png", bgr_view)
                    item.update({"status": "success", "original_image": key,
                        "drivable_mask": paths["drivable"].relative_to(collection).as_posix(),
                        "lane_mask": paths["lane"].relative_to(collection).as_posix(),
                        "combined_mask": paths["combined"].relative_to(collection).as_posix(),
                        "width": w, "height": h, "annotated_at": datetime.now().astimezone().isoformat(),
                        "model_filename": self.model_path.name, "model_sha256": model_hash,
                        "execution_provider": active, "inference": {"input_width": size[0], "input_height": size[1], "resize": "direct linear resize", "drivable_output_index": 4, "drivable_channel": "1 - channel 0", "lane_output_index": 5, "threshold": 0.5}})
                    stats["success"] += 1
                    scope_key = f"{collection.name}|{image_path.parent.parent.name}"
                    scopes[scope_key]["success"] += 1
                    if stats["success"] % self.preview_interval == 0:
                        self.preview.emit({"rgb": rgb, "drive": drive, "lane": lane, "combined": combined,
                                           "path": str(image_path), "dims": (w, h)})
                    self.log.emit(f"Annotated {key} ({w}×{h})")
                except Exception as exc:
                    item.update({"status": "failed", "error": str(exc), "failed_at": datetime.now().astimezone().isoformat()})
                    stats["failed"] += 1
                    scope_key = f"{collection.name}|{image_path.parent.parent.name}"
                    scopes[scope_key]["failed"] += 1
                    self.log.emit(f"Failed {key}: {exc}")
                finally:
                    _atomic_json(manifest_path, manifest)
                    stats["remaining"] = max(0, stats["total"] - stats["skipped"] - stats["success"] - stats["failed"])
                    elapsed = time.monotonic() - start
                    completed = stats["success"] + stats["failed"]
                    stats.update({"current": f"{collection.name}/{key}", "elapsed": elapsed,
                                  "rate": completed / elapsed if elapsed else 0, "scopes": scopes})
                    self.progress.emit(dict(stats))
            stats["stopped"] = self._stop.is_set()
            self.done.emit(stats)
        except Exception as exc:
            error_text = str(exc).strip() or "<no exception message>"
            stats["error"] = f"{type(exc).__name__}: {error_text}"
            self.log.emit(f"Annotation startup error ({stats['error']}):\n{traceback.format_exc()}")
            self.done.emit(stats)
