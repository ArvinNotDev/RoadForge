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
    """Discover CARLA collections and ordinary image folders such as AVIS captures.

    An AVIS capture folder only needs an images/ directory. Its optional
    manifest.jsonl is kept as source metadata; no CARLA collection manifest,
    Town, Map, Weather, or route metadata is required.
    """
    result = {}
    root = Path(dataset_root).expanduser()
    if not root.is_dir():
        return result

    def has_images(folder):
        image_dir = folder / "images"
        if image_dir.is_dir() and any(
            path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            for path in image_dir.iterdir()
        ):
            return True
        return any(
            path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            for path in folder.iterdir()
        )

    if (
        (root / "collection_manifest.json").is_file()
        or (root / "raw").is_dir()
        or has_images(root)
    ):
        candidates = [root]
    else:
        candidates = sorted(root.glob("collection_*"))
        candidates.extend(
            child for child in sorted(root.iterdir())
            if child.is_dir()
            and child not in candidates
            and (
                (child / "collection_manifest.json").is_file()
                or (child / "raw").is_dir()
                or has_images(child)
            )
        )

    for collection in candidates:
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
                    images = sorted(
                        p for p in image_dir.iterdir()
                        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                    )
                    if images:
                        towns[town_dir.name] = images
        else:
            image_dir = collection / "images"
            if not image_dir.is_dir():
                # Also accept a folder containing image files directly.
                image_dir = collection
            images = sorted(
                p for p in image_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
            if images:
                towns["AVIS"] = images
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
    """Check a complete four-class annotation transaction, including vehicle masks."""
    try:
        expected = (int(record["height"]), int(record["width"]))
        decoded = {}
        for key in ("drivable_mask", "lane_mask", "vehicle_mask", "combined_mask"):
            relative = record.get(key)
            if not relative:
                return False
            path = Path(collection) / relative
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != expected:
                return False
            decoded[key] = mask

        drive = decoded["drivable_mask"]
        lane = decoded["lane_mask"]
        vehicle = decoded["vehicle_mask"]
        combined = decoded["combined_mask"]
        return (
            set(np.unique(drive)).issubset({0, 255})
            and set(np.unique(lane)).issubset({0, 255})
            and set(np.unique(vehicle)).issubset({0, 255})
            and set(np.unique(combined)).issubset({0, 1, 2, 3})
            # Vehicle class 3 intentionally overrides lane class 2 and drivable
            # class 1 where detected boxes overlap those regions.
            and bool(np.all(combined[(lane > 0) & (vehicle == 0)] == 2))
            and bool(np.all(combined[vehicle > 0] == 3))
            and bool(np.all(combined[(drive > 0) & (lane == 0) & (vehicle == 0)] == 1))
            and record.get("class_map", {}).get("3") == "vehicles"
            # Regenerate legacy vehicle annotations with the hood-exclusion
            # pipeline instead of silently skipping masks made with old settings.
            and int(record.get("vehicle_annotation", {}).get("pipeline_version", 0))
                == DEFAULT_VEHICLE_ANNOTATION_PIPELINE_VERSION
        )
    except (KeyError, TypeError, ValueError, OSError, AttributeError):
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


COCO_VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
DEFAULT_VEHICLE_CLASS_IDS = tuple(COCO_VEHICLE_CLASSES)
DEFAULT_VEHICLE_ANNOTATION_PIPELINE_VERSION = 2
DEFAULT_VEHICLE_SCORE_THRESHOLD = 0.50
DEFAULT_VEHICLE_NMS_THRESHOLD = 0.40
# Ignore the bottom 22% of the camera image for vehicle detections to exclude
# the ego-vehicle hood. Boxes crossing the boundary are clipped, not discarded.
DEFAULT_VEHICLE_BOTTOM_KEEP_RATIO = 0.78


def _sigmoid(values):
    values = np.asarray(values, dtype=np.float32)
    values = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-values))


def _detection_heads(output):
    """Normalize the upstream YOLOPv2 trace-model detection output into 3 BCHW heads."""
    if isinstance(output, (list, tuple)):
        heads = [np.asarray(item) for item in output]
    else:
        array = np.asarray(output)
        if array.ndim == 5 and array.shape[0] == 3:
            heads = [array[index] for index in range(3)]
        elif array.ndim == 5 and array.shape[1] == 3:
            heads = [array[:, index] for index in range(3)]
        elif array.ndim == 4 and array.shape[0] == 3:
            heads = [array[index:index + 1] for index in range(3)]
        else:
            raise RuntimeError(
                "Unsupported YOLOPv2 detection-head layout: "
                f"{array.shape}. Expected three anchor heads."
            )
    if len(heads) != 3:
        raise RuntimeError(f"Expected 3 YOLOPv2 detection heads, found {len(heads)}")
    normalized = []
    for index, head in enumerate(heads):
        head = np.asarray(head, dtype=np.float32)
        if head.ndim == 3:
            head = head[None, ...]
        if head.ndim != 4 or head.shape[0] != 1 or head.shape[1] != 255:
            raise RuntimeError(
                f"Detection head {index} has shape {head.shape}; expected (1, 255, H, W). "
                "This adapter requires the YOLOPv2 COCO-80 export."
            )
        normalized.append(head)
    return normalized


def decode_vehicle_detections(
    outputs,
    input_size,
    original_size,
    score_threshold=DEFAULT_VEHICLE_SCORE_THRESHOLD,
    iou_threshold=DEFAULT_VEHICLE_NMS_THRESHOLD,
    vehicle_class_ids=DEFAULT_VEHICLE_CLASS_IDS,
    max_detections=300,
    bottom_keep_ratio=DEFAULT_VEHICLE_BOTTOM_KEEP_RATIO,
):
    """Decode YOLOPv2's three anchor heads and return COCO vehicle boxes.

    Coordinates are mapped from the model's direct-resize input space to the
    source image. This deliberately matches RoadForge's existing RGB/semantic
    preprocessing and avoids changing historical drivable/lane pseudo-labels.
    """
    if len(outputs) != 6:
        raise RuntimeError(f"Model returned {len(outputs)} outputs; expected 6")
    score_threshold = float(score_threshold)
    iou_threshold = float(iou_threshold)
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("Vehicle confidence threshold must be between 0 and 1.")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("Vehicle NMS IoU threshold must be between 0 and 1.")
    bottom_keep_ratio = float(bottom_keep_ratio)
    if not 0.0 < bottom_keep_ratio <= 1.0:
        raise ValueError("Vehicle bottom keep ratio must be greater than 0 and at most 1.")

    input_width, input_height = map(int, input_size)
    original_width, original_height = map(int, original_size)
    if min(input_width, input_height, original_width, original_height) <= 0:
        raise ValueError("Input and source image dimensions must be positive.")

    heads = _detection_heads(outputs[0])
    strides = (8, 16, 32)
    decoded = []
    for index, (head, stride) in enumerate(zip(heads, strides)):
        batch, _channels, grid_h, grid_w = head.shape
        values = head.reshape(batch, 3, 85, grid_h, grid_w).transpose(0, 1, 3, 4, 2)
        values = _sigmoid(values)

        grid_x, grid_y = np.meshgrid(
            np.arange(grid_w, dtype=np.float32),
            np.arange(grid_h, dtype=np.float32),
        )
        grid = np.stack((grid_x, grid_y), axis=-1)[None, None, ...]
        values[..., 0:2] = (values[..., 0:2] * 2.0 - 0.5 + grid) * stride

        anchors = np.asarray(outputs[index + 1], dtype=np.float32)
        if anchors.size != 6:
            raise RuntimeError(
                f"YOLOPv2 anchor output {index + 1} has shape {anchors.shape}; expected 6 values."
            )
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        values[..., 2:4] = np.square(values[..., 2:4] * 2.0) * anchors
        decoded.append(values.reshape(batch, -1, 85))

    predictions = np.concatenate(decoded, axis=1)[0]
    objectness = predictions[:, 4]
    class_scores = predictions[:, 5:] * objectness[:, None]
    class_ids = np.argmax(class_scores, axis=1).astype(np.int32)
    scores = class_scores[np.arange(len(class_scores)), class_ids]
    allowed = {int(value) for value in vehicle_class_ids}
    candidate = (scores >= score_threshold) & np.isin(class_ids, list(allowed))
    if not np.any(candidate):
        return []

    rows = predictions[candidate]
    scores = scores[candidate]
    class_ids = class_ids[candidate]
    xywh = rows[:, :4]
    boxes = np.empty_like(xywh, dtype=np.float32)
    boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2.0
    boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2.0
    boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2.0
    boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2.0

    # Exclude the lower image strip from vehicle detections before NMS so an
    # ego-hood false positive cannot suppress a valid vehicle box. A box that
    # crosses the cutoff is retained but clipped to the visible upper region.
    bottom_cutoff = input_height * bottom_keep_ratio
    keep = boxes[:, 1] < bottom_cutoff
    boxes = boxes[keep].copy()
    scores = scores[keep]
    class_ids = class_ids[keep]
    if boxes.size == 0:
        return []
    boxes[:, 3] = np.minimum(boxes[:, 3], bottom_cutoff)
    keep = boxes[:, 3] > boxes[:, 1]
    boxes = boxes[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]
    if boxes.size == 0:
        return []

    # Class-aware NMS prevents a nearby car and bus from suppressing each other.
    kept = []
    for class_id in sorted(set(class_ids.tolist())):
        indices = np.flatnonzero(class_ids == class_id)
        order = indices[np.argsort(scores[indices])[::-1]]
        while order.size:
            current = int(order[0])
            kept.append(current)
            if order.size == 1:
                break
            rest = order[1:]
            left_top = np.maximum(boxes[current, :2], boxes[rest, :2])
            right_bottom = np.minimum(boxes[current, 2:], boxes[rest, 2:])
            intersection_wh = np.maximum(0.0, right_bottom - left_top)
            intersection = intersection_wh[:, 0] * intersection_wh[:, 1]
            area_current = max(0.0, boxes[current, 2] - boxes[current, 0]) * max(
                0.0, boxes[current, 3] - boxes[current, 1]
            )
            area_rest = np.maximum(0.0, boxes[rest, 2] - boxes[rest, 0]) * np.maximum(
                0.0, boxes[rest, 3] - boxes[rest, 1]
            )
            union = area_current + area_rest - intersection
            iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
            order = rest[iou <= iou_threshold]

    kept.sort(key=lambda index: float(scores[index]), reverse=True)
    scale_x = original_width / float(input_width)
    scale_y = original_height / float(input_height)
    result = []
    original_bottom_cutoff = max(
        1, min(original_height, int(round(original_height * bottom_keep_ratio)))
    )
    for index in kept[:max(1, int(max_detections))]:
        x1, y1, x2, y2 = boxes[index]
        x1 = int(np.clip(round(float(x1) * scale_x), 0, original_width - 1))
        y1 = int(np.clip(round(float(y1) * scale_y), 0, original_height - 1))
        if y1 >= original_bottom_cutoff:
            continue
        x2 = int(np.clip(round(float(x2) * scale_x), x1 + 1, original_width))
        y2 = int(np.clip(
            round(float(y2) * scale_y), y1 + 1, original_bottom_cutoff
        ))
        result.append({
            "bbox": (x1, y1, x2, y2),
            "score": float(scores[index]),
            "class_id": int(class_ids[index]),
            "class_name": COCO_VEHICLE_CLASSES.get(int(class_ids[index]), str(int(class_ids[index]))),
        })
    return result


def _grabcut_vehicle_instance(rgb, bbox):
    """Create a best-effort foreground silhouette inside one detector rectangle."""
    height, width = rgb.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    box_width, box_height = x2 - x1, y2 - y1
    if box_width < 8 or box_height < 8:
        return None

    margin = max(4, int(max(box_width, box_height) * 0.12))
    crop_x1 = max(0, x1 - margin)
    crop_y1 = max(0, y1 - margin)
    crop_x2 = min(width, x2 + margin)
    crop_y2 = min(height, y2 + margin)
    crop_rgb = np.ascontiguousarray(rgb[crop_y1:crop_y2, crop_x1:crop_x2])
    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    crop_h, crop_w = crop_bgr.shape[:2]

    local_x1, local_y1 = x1 - crop_x1, y1 - crop_y1
    local_x2, local_y2 = x2 - crop_x1, y2 - crop_y1
    if local_x2 - local_x1 < 2 or local_y2 - local_y1 < 2:
        return None

    mask = np.full((crop_h, crop_w), cv2.GC_PR_BGD, dtype=np.uint8)
    mask[local_y1:local_y2, local_x1:local_x2] = cv2.GC_PR_FGD
    core_x = max(1, int((local_x2 - local_x1) * 0.22))
    core_y = max(1, int((local_y2 - local_y1) * 0.22))
    inner_x1 = min(local_x2 - 1, local_x1 + core_x)
    inner_x2 = max(inner_x1 + 1, local_x2 - core_x)
    inner_y1 = min(local_y2 - 1, local_y1 + core_y)
    inner_y2 = max(inner_y1 + 1, local_y2 - core_y)
    mask[inner_y1:inner_y2, inner_x1:inner_x2] = cv2.GC_FGD

    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(
            crop_bgr, mask, None, background_model, foreground_model,
            4, cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        return None

    foreground = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    box_only = np.zeros_like(foreground)
    box_only[local_y1:local_y2, local_x1:local_x2] = 1
    foreground &= box_only
    if int(foreground.sum()) < max(8, int(box_width * box_height * 0.04)):
        return None

    # Keep the component touching the detection-box centre when possible.
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(foreground, 8)
    if count <= 1:
        return None
    center_label = int(labels[(local_y1 + local_y2) // 2, (local_x1 + local_x2) // 2])
    if center_label > 0:
        chosen = center_label
    else:
        chosen = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    instance = (labels == chosen).astype(np.uint8) * 255
    return instance, (crop_x1, crop_y1)


def build_vehicle_mask(rgb, detections, mask_method="grabcut"):
    """Merge instance masks into one binary vehicle semantic mask.

    GrabCut is the default because YOLOPv2 supplies boxes, not vehicle
    segmentation. Box masks remain available as an explicitly coarse fallback.
    """
    height, width = rgb.shape[:2]
    vehicle_mask = np.zeros((height, width), dtype=np.uint8)
    fallback_count = 0
    method = str(mask_method).strip().lower()
    if method not in {"grabcut", "box"}:
        raise ValueError("Vehicle mask method must be 'grabcut' or 'box'.")

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        if method == "box":
            vehicle_mask[y1:y2, x1:x2] = 255
            continue

        result = _grabcut_vehicle_instance(rgb, (x1, y1, x2, y2))
        if result is None:
            # A detector rectangle is safer than silently dropping a confident
            # vehicle, but its coarse nature is recorded in the manifest/log.
            vehicle_mask[y1:y2, x1:x2] = 255
            fallback_count += 1
            continue
        instance, (offset_x, offset_y) = result
        ih, iw = instance.shape
        region = vehicle_mask[offset_y:offset_y + ih, offset_x:offset_x + iw]
        region[instance > 0] = 255

    return vehicle_mask, fallback_count


class YOLOPv2Predictor:
    """Run the six-output YOLOPv2 ONNX graph for lane, drivable-area, and vehicle pseudo-labels."""
    def __init__(self, session, input_size):
        self.session = session
        self.input_width, self.input_height = input_size
        self.input_name = session.get_inputs()[0].name
        self.outputs = session.get_outputs()
        if len(self.outputs) != 6:
            raise RuntimeError(f"Expected verified YOLOPv2 six-output graph, found {len(self.outputs)}")

    def predict_full(
        self, rgb, *, annotate_vehicles=True,
        score_threshold=DEFAULT_VEHICLE_SCORE_THRESHOLD,
        iou_threshold=DEFAULT_VEHICLE_NMS_THRESHOLD,
        vehicle_class_ids=DEFAULT_VEHICLE_CLASS_IDS,
        mask_method="grabcut",
        bottom_keep_ratio=DEFAULT_VEHICLE_BOTTOM_KEEP_RATIO,
    ):
        h, w = rgb.shape[:2]
        # Keep the existing direct-resize RGB preprocessing unchanged for the
        # drivable/lane teacher masks. The detection boxes are scaled back with
        # independent x/y ratios to match this same input transform.
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        image = cv2.resize(
            bgr, (self.input_width, self.input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = np.expand_dims(np.transpose(image, (2, 0, 1)), 0)
        outputs = self.session.run(None, {self.input_name: tensor})
        if len(outputs) != 6:
            raise RuntimeError(f"Model returned {len(outputs)} outputs; expected 6")

        drive_output = np.asarray(outputs[4], dtype=np.float32)
        if drive_output.ndim != 4 or drive_output.shape[:2] != (1, 2):
            raise RuntimeError(f"Drivable output shape mismatch: {drive_output.shape}")
        drive_prob = 1.0 - drive_output[0, 0]
        lane_prob = np.squeeze(np.asarray(outputs[5], dtype=np.float32))
        if lane_prob.ndim != 2:
            raise RuntimeError(f"Lane output shape mismatch: {np.asarray(outputs[5]).shape}")
        if drive_prob.shape != (self.input_height, self.input_width) or lane_prob.shape != drive_prob.shape:
            raise RuntimeError(f"Segmentation spatial shape mismatch: drive={drive_prob.shape}, lane={lane_prob.shape}")

        drive = (drive_prob >= 0.50).astype(np.uint8) * 255
        lane = (lane_prob > 0.50).astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        lane = cv2.morphologyEx(lane, cv2.MORPH_OPEN, kernel, iterations=1)
        lane = cv2.morphologyEx(lane, cv2.MORPH_CLOSE, kernel, iterations=1)
        if (w, h) != (self.input_width, self.input_height):
            drive = cv2.resize(drive, (w, h), interpolation=cv2.INTER_NEAREST)
            lane = cv2.resize(lane, (w, h), interpolation=cv2.INTER_NEAREST)

        detections = []
        vehicle_mask = np.zeros((h, w), dtype=np.uint8)
        fallback_count = 0
        if annotate_vehicles:
            detections = decode_vehicle_detections(
                outputs, (self.input_width, self.input_height), (w, h),
                score_threshold, iou_threshold, vehicle_class_ids,
                bottom_keep_ratio=bottom_keep_ratio,
            )
            vehicle_mask, fallback_count = build_vehicle_mask(
                rgb, detections, mask_method=mask_method,
            )

        # Vehicle pixels take precedence over road/lane pseudo-labels.
        combined = np.zeros((h, w), dtype=np.uint8)
        combined[drive > 0] = 1
        combined[lane > 0] = 2
        combined[vehicle_mask > 0] = 3
        return {
            "drive": drive, "lane": lane, "vehicle": vehicle_mask,
            "combined": combined, "detections": detections,
            "vehicle_mask_fallbacks": fallback_count,
        }

    def predict(self, rgb):
        """Backward-compatible three-mask API used by older integrations."""
        result = self.predict_full(rgb, annotate_vehicles=False)
        return result["drive"], result["lane"], result["combined"]


def visualization(rgb, drive, lane, combined, vehicle=None, detections=None, opacity=0.42):
    overlay = rgb.copy()
    overlay[drive > 0] = (0, 255, 0)
    overlay[lane > 0] = (255, 220, 0)
    if vehicle is not None:
        overlay[vehicle > 0] = (255, 80, 80)
    blended = cv2.addWeighted(rgb, 1.0 - opacity, overlay, opacity, 0)

    colors = np.zeros_like(rgb)
    colors[combined == 1] = (0, 200, 0)
    colors[combined == 2] = (255, 220, 0)
    colors[combined == 3] = (255, 70, 70)
    detection_view = blended.copy()
    for detection in detections or ():
        x1, y1, x2, y2 = detection["bbox"]
        score = float(detection["score"])
        label = f"{detection['class_name']} {score:.2f}"
        cv2.rectangle(detection_view, (x1, y1), (x2, y2), (255, 70, 70), 2)
        cv2.putText(
            detection_view, label, (x1, max(14, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 70, 70), 1, cv2.LINE_AA,
        )
    views = {
        "Original RGB": rgb,
        "Drivable Area": cv2.cvtColor(drive, cv2.COLOR_GRAY2RGB),
        "Lane Lines": cv2.cvtColor(lane, cv2.COLOR_GRAY2RGB),
        "Vehicle Mask": cv2.cvtColor(vehicle if vehicle is not None else np.zeros_like(drive), cv2.COLOR_GRAY2RGB),
        "Combined classes": colors,
        "Combined overlay": cv2.addWeighted(rgb, 0.58, colors, 0.42, 0),
        "Vehicle detections": detection_view,
    }
    return views


class AnnotationWorker(QThread):
    progress = Signal(dict)
    preview = Signal(object)
    log = Signal(str)
    done = Signal(dict)

    def __init__(
        self, jobs, model_path, provider="Auto", save_visualizations=True,
        reprocess=False, preview_interval=1, annotate_vehicles=True,
        vehicle_score_threshold=DEFAULT_VEHICLE_SCORE_THRESHOLD,
        vehicle_nms_threshold=DEFAULT_VEHICLE_NMS_THRESHOLD,
        vehicle_class_ids=DEFAULT_VEHICLE_CLASS_IDS, vehicle_mask_method="grabcut",
        vehicle_bottom_keep_ratio=DEFAULT_VEHICLE_BOTTOM_KEEP_RATIO,
    ):
        super().__init__()
        self.jobs = jobs
        self.model_path = Path(model_path)
        self.provider = provider
        self.save_visualizations = save_visualizations
        self.reprocess = reprocess
        self.preview_interval = max(1, preview_interval)
        self.annotate_vehicles = bool(annotate_vehicles)
        self.vehicle_score_threshold = float(vehicle_score_threshold)
        self.vehicle_nms_threshold = float(vehicle_nms_threshold)
        self.vehicle_class_ids = tuple(int(value) for value in vehicle_class_ids)
        self.vehicle_mask_method = str(vehicle_mask_method)
        self.vehicle_bottom_keep_ratio = float(vehicle_bottom_keep_ratio)
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
            if self.annotate_vehicles and self.vehicle_class_ids:
                self.log.emit(
                    "Vehicle teacher enabled · COCO classes: "
                    + ", ".join(COCO_VEHICLE_CLASSES.get(value, str(value)) for value in self.vehicle_class_ids)
                    + f" · confidence {self.vehicle_score_threshold:.2f} · NMS {self.vehicle_nms_threshold:.2f} · "
                    + f"bottom excluded {(1.0 - self.vehicle_bottom_keep_ratio) * 100:.0f}% · mask {self.vehicle_mask_method}"
                )
            else:
                self.log.emit("Vehicle teacher disabled; new masks will contain an empty vehicles class.")
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
                paths = {
                    "drivable": outroot / "drivable_area" / f"{stem}.png",
                    "lane": outroot / "lane_lines" / f"{stem}.png",
                    "vehicle": outroot / "vehicle_masks" / f"{stem}.png",
                    "combined": outroot / "combined_4class" / f"{stem}.png",
                }
                images_record = manifest.setdefault("images", {})
                item = images_record.setdefault(key, {})
                previous_item = dict(item)
                if not self.reprocess and previous_status == "failed":
                    scopes[f"{collection.name}|{town}"]["failed"] -= 1
                # Keep the previous manifest pointer active until all four outputs
                # are written successfully. An interrupted upgrade therefore does
                # not hide an existing three-class annotation from the dataset index.
                self.log.emit(f"Annotating {collection.name}/{key}")
                try:
                    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                    if bgr is None: raise ValueError("Image could not be decoded by OpenCV")
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    h, w = rgb.shape[:2]
                    result = predictor.predict_full(
                        rgb,
                        annotate_vehicles=self.annotate_vehicles,
                        score_threshold=self.vehicle_score_threshold,
                        iou_threshold=self.vehicle_nms_threshold,
                        vehicle_class_ids=self.vehicle_class_ids,
                        mask_method=self.vehicle_mask_method,
                        bottom_keep_ratio=self.vehicle_bottom_keep_ratio,
                    )
                    drive = result["drive"]
                    lane = result["lane"]
                    vehicle = result["vehicle"]
                    combined = result["combined"]
                    detections = result["detections"]
                    if any(mask.shape != (h, w) for mask in (drive, lane, vehicle, combined)):
                        raise ValueError("Output masks do not match original dimensions")

                    _atomic_png(paths["drivable"], drive)
                    _atomic_png(paths["lane"], lane)
                    _atomic_png(paths["vehicle"], vehicle)
                    _atomic_png(paths["combined"], combined)
                    if self.save_visualizations:
                        views = visualization(
                            rgb, drive, lane, combined, vehicle=vehicle,
                            detections=detections,
                        )
                        visdir = collection / "labels_yolopv2" / town / "visualizations"
                        for name, arr in views.items():
                            if name == "Original RGB": continue
                            bgr_view = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                            _atomic_png(visdir / f"{stem}_{name.lower().replace(' ', '_')}.png", bgr_view)

                    item.update({
                        "status": "success", "original_image": key,
                        "drivable_mask": paths["drivable"].relative_to(collection).as_posix(),
                        "lane_mask": paths["lane"].relative_to(collection).as_posix(),
                        "vehicle_mask": paths["vehicle"].relative_to(collection).as_posix(),
                        "combined_mask": paths["combined"].relative_to(collection).as_posix(),
                        "class_map": {"0": "background", "1": "ego_drivable_area", "2": "lane_lines", "3": "vehicles"},
                        "num_classes": 4,
                        "vehicle_annotation": {
                            "pipeline_version": DEFAULT_VEHICLE_ANNOTATION_PIPELINE_VERSION,
                            "enabled": self.annotate_vehicles,
                            "model": "YOLOPv2-ONNX",
                            "class_ids": list(self.vehicle_class_ids),
                            "class_names": [COCO_VEHICLE_CLASSES.get(value, str(value)) for value in self.vehicle_class_ids],
                            "confidence_threshold": self.vehicle_score_threshold,
                            "nms_iou_threshold": self.vehicle_nms_threshold,
                            "bottom_keep_height_ratio": self.vehicle_bottom_keep_ratio,
                            "bottom_excluded_height_ratio": 1.0 - self.vehicle_bottom_keep_ratio,
                            "mask_method": self.vehicle_mask_method,
                            "detection_count": len(detections),
                            "grabcut_box_fallbacks": int(result["vehicle_mask_fallbacks"]),
                        },
                        "width": w, "height": h,
                        "annotated_at": datetime.now().astimezone().isoformat(),
                        "model_filename": self.model_path.name, "model_sha256": model_hash,
                        "execution_provider": active,
                        "inference": {
                            "input_width": size[0], "input_height": size[1],
                            "resize": "direct linear resize",
                            "detection_output_index": 0, "anchor_output_indices": [1, 2, 3],
                            "drivable_output_index": 4, "drivable_channel": "1 - channel 0",
                            "lane_output_index": 5, "segmentation_threshold": 0.5,
                        },
                    })
                    # Manifest is the commit marker: do not switch the dataset to
                    # the new mask until every output and visualization is complete.
                    images_record[key] = item
                    _atomic_json(manifest_path, manifest)

                    stats["success"] += 1
                    stats["vehicle_detections"] = stats.get("vehicle_detections", 0) + len(detections)
                    stats["vehicle_fallbacks"] = stats.get("vehicle_fallbacks", 0) + int(result["vehicle_mask_fallbacks"])
                    scope_key = f"{collection.name}|{town}"
                    scopes[scope_key]["success"] += 1
                    if stats["success"] % self.preview_interval == 0:
                        self.preview.emit({
                            "rgb": rgb, "drive": drive, "lane": lane,
                            "vehicle": vehicle, "combined": combined,
                            "detections": detections, "path": str(image_path),
                            "dims": (w, h),
                        })
                    self.log.emit(
                        f"Annotated {key} ({w}×{h}) · vehicle detections: {len(detections)}"
                        + (f" · GrabCut box fallbacks: {result['vehicle_mask_fallbacks']}" if result["vehicle_mask_fallbacks"] else "")
                    )
                except Exception as exc:
                    # Preserve the previous committed annotation when an upgrade
                    # fails, so users can still train from their previous masks.
                    if previous_item:
                        previous_item["last_vehicle_upgrade_error"] = str(exc)
                        images_record[key] = previous_item
                    else:
                        images_record[key] = {
                            "original_image": key, "status": "failed",
                            "error": str(exc),
                            "failed_at": datetime.now().astimezone().isoformat(),
                        }
                    stats["failed"] += 1
                    scope_key = f"{collection.name}|{town}"
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
