from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
VALID_CLASS_IDS = {0, 1, 2}


@dataclass
class Sample:
    image: str
    mask: str | None
    collection: str
    town: str
    session_id: str
    group_id: str
    filename: str
    weather: str = "Unknown"
    traffic: str = "Unknown"
    route_id: str = "Unknown"
    annotation_status: str = "missing"
    metadata: dict = field(default_factory=dict)
    content_hash: str = ""


@dataclass
class CollectionSummary:
    path: str
    name: str
    towns: list[str]
    image_count: int
    annotated_count: int
    missing_mask_count: int
    weather: list[str]
    traffic: list[str]
    dimensions: list[str]
    session_id: str
    samples: list[Sample] = field(default_factory=list, repr=False)


@dataclass
class ValidationIssue:
    severity: str
    code: str
    message: str
    image: str = ""
    mask: str = ""
    collection: str = ""


@dataclass
class ValidationReport:
    checked: int
    valid: list[Sample]
    issues: list[ValidationIssue]
    class_pixels: dict[int, int]
    class_images: dict[int, int]
    dimensions: dict[str, int]
    duplicate_image_hashes: dict[str, list[str]]

    @property
    def invalid_count(self) -> int:
        return len({x.image for x in self.issues if x.image and x.severity == "error"})


def _safe_manifest_path(collection: Path, value: str) -> Path | None:
    if not value:
        return None
    root = collection.resolve()
    target = (collection / value).resolve()
    return target if target == root or root in target.parents else None


def discover_collections(dataset_roots: Iterable[str | Path]) -> list[CollectionSummary]:
    """Discover raw CARLA collections and only use manifest-declared combined masks."""
    summaries: list[CollectionSummary] = []
    seen_paths: set[Path] = set()
    for root_value in dataset_roots:
        root = Path(root_value).expanduser()
        if not root.is_dir():
            continue
        for collection in sorted(root.glob("collection_*")):
            if not collection.is_dir() or collection.resolve() in seen_paths:
                continue
            seen_paths.add(collection.resolve())
            try:
                manifest = json.loads((collection / "collection_manifest.json").read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError):
                manifest = {}
            collection_config = manifest.get("config", {}) if isinstance(manifest.get("config", {}), dict) else {}
            try:
                annotations = json.loads((collection / "annotation_manifest.json").read_text(encoding="utf-8-sig"))
                annotation_entries = annotations.get("images", {})
                if not isinstance(annotation_entries, dict):
                    annotation_entries = {}
            except (OSError, json.JSONDecodeError):
                annotation_entries = {}

            metadata_by_path: dict[str, dict] = {}
            metadata_by_filename: dict[str, dict] = {}
            for metadata_file in collection.glob("raw/*/metadata.jsonl"):
                try:
                    with metadata_file.open("r", encoding="utf-8-sig") as stream:
                        for line in stream:
                            try:
                                record = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            rel = (Path("raw") / metadata_file.parent.name / "images" / str(record.get("filename", ""))).as_posix()
                            metadata_by_path[rel] = record
                            if record.get("filename"):
                                metadata_by_filename[str(record["filename"])] = record
                except OSError:
                    continue

            session_id = str(manifest.get("session_id") or collection.name)
            samples: list[Sample] = []
            seen_rel: set[str] = set()
            for town_dir in sorted((collection / "raw").glob("*")) if (collection / "raw").is_dir() else []:
                image_dir = town_dir / "images"
                if not image_dir.is_dir():
                    continue
                for image in sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS):
                    rel = image.relative_to(collection).as_posix()
                    seen_rel.add(rel)
                    entry = annotation_entries.get(rel, {})
                    mask = None
                    status = str(entry.get("status", "missing"))
                    # Primary labels come only from combined_mask in the teacher manifest.
                    combined = entry.get("combined_mask") if status == "success" else None
                    mask_path = _safe_manifest_path(collection, str(combined)) if combined else None
                    if mask_path and mask_path.is_file():
                        mask = str(mask_path)
                    elif not entry:
                        # A manifest-free import can still discover a uniquely named canonical
                        # combined mask. Binary directories and visualizations are never candidates.
                        candidate = collection / "labels_yolopv2" / town_dir.name / "combined_3class" / image.name
                        if candidate.is_file():
                            mask = str(candidate)
                            status = "discovered_combined_3class"
                    metadata = metadata_by_path.get(rel, metadata_by_filename.get(image.name, {}))
                    metadata = dict(metadata)
                    if entry:
                        metadata["_annotation_original_image"] = entry.get("original_image")
                    town = str(metadata.get("town", town_dir.name))
                    traffic_obj = metadata.get("traffic_manager", {})
                    traffic_value = traffic_obj.get("vehicle_count", traffic_obj.get("traffic_level", "Unknown")) if isinstance(traffic_obj, dict) else "Unknown"
                    if traffic_value == "Unknown" and "background_traffic" in collection_config:
                        traffic_value = (f"Background traffic ({collection_config.get('background_vehicle_count', 'unknown')} vehicles)"
                                         if collection_config.get("background_traffic") else "No background traffic")
                    route_id = str(metadata.get("route_id", metadata.get("spawn_point_id", "Unknown")))
                    samples.append(Sample(
                        image=str(image), mask=mask, collection=str(collection), town=town,
                        session_id=str(metadata.get("collection_session_id", session_id)),
                        group_id=str(metadata.get("collection_session_id", session_id)), filename=image.name,
                        weather=str(metadata.get("weather", "Unknown")), traffic=str(traffic_value),
                        route_id=route_id, annotation_status=status, metadata=metadata,
                    ))
            # Preserve annotation-manifest entries whose source image disappeared so
            # validation can report missing images instead of silently dropping them.
            for rel, entry in annotation_entries.items():
                if rel in seen_rel or not str(rel).replace("\\", "/").startswith("raw/"):
                    continue
                image = _safe_manifest_path(collection, str(rel))
                if image is None:
                    continue
                town = Path(str(rel)).parts[1] if len(Path(str(rel)).parts) > 1 else "Unknown"
                mask = _safe_manifest_path(collection, str(entry.get("combined_mask", ""))) if entry.get("status") == "success" else None
                metadata = {"_annotation_original_image": entry.get("original_image")}
                samples.append(Sample(str(image), str(mask) if mask else None, str(collection), town, session_id, session_id,
                                      image.name, "Unknown", "Unknown", "Unknown", str(entry.get("status", "missing")), metadata))
            weather = sorted({s.weather for s in samples})
            traffic = sorted({s.traffic for s in samples})
            towns = sorted({s.town for s in samples})
            dimensions: set[str] = set()
            for sample in samples:
                dims = sample.metadata.get("camera_resolution")
                if isinstance(dims, list) and len(dims) == 2:
                    dimensions.add(f"{dims[0]}×{dims[1]}")
            summaries.append(CollectionSummary(
                path=str(collection), name=collection.name, towns=towns,
                image_count=len(samples), annotated_count=sum(s.mask is not None for s in samples),
                missing_mask_count=sum(s.mask is None for s in samples), weather=weather,
                traffic=traffic, dimensions=sorted(dimensions), session_id=session_id, samples=samples,
            ))
    return summaries


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_samples(samples: Iterable[Sample], ignore_label: int | None = None) -> ValidationReport:
    if ignore_label is not None and (not isinstance(ignore_label, int) or ignore_label < 0 or ignore_label > 255 or ignore_label in VALID_CLASS_IDS):
        raise ValueError("ignore_label must be a uint8 value outside the semantic class IDs 0, 1, and 2.")
    samples = list(samples)
    issues: list[ValidationIssue] = []
    valid: list[Sample] = []
    pixels: Counter[int] = Counter()
    class_images: Counter[int] = Counter()
    dimensions: Counter[str] = Counter()
    hashes: dict[str, list[str]] = defaultdict(list)
    filename_locations: dict[str, list[str]] = defaultdict(list)
    invalid_images: set[str] = set()

    for sample in samples:
        image_path = Path(sample.image)
        filename_locations[sample.filename.casefold()].append(sample.image)
        declared_image = sample.metadata.get("_annotation_original_image")
        if declared_image:
            try:
                actual_rel = image_path.relative_to(Path(sample.collection)).as_posix()
            except ValueError:
                actual_rel = ""
            if str(declared_image).replace("\\", "/") != actual_rel:
                issues.append(ValidationIssue("error", "mismatched_pair", f"Annotation manifest pairs this mask with {declared_image}, not {actual_rel}.", sample.image, sample.mask or "", sample.collection))
                invalid_images.add(sample.image)
                continue
        if not image_path.is_file():
            issues.append(ValidationIssue("error", "missing_image", "Image file does not exist.", sample.image, sample.mask or "", sample.collection))
            invalid_images.add(sample.image)
            continue
        try:
            with Image.open(image_path) as im:
                im.verify()
            with Image.open(image_path) as im:
                image_size = im.size
                im.convert("RGB").load()
            dimensions[f"{image_size[0]}×{image_size[1]}"] += 1
            sample.content_hash = _sha256(image_path)
            hashes[sample.content_hash].append(sample.image)
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            issues.append(ValidationIssue("error", "invalid_image", f"Image cannot be decoded: {exc}", sample.image, sample.mask or "", sample.collection))
            invalid_images.add(sample.image)
            continue
        if not sample.mask:
            issues.append(ValidationIssue("error", "missing_mask", "No manifest-declared combined semantic mask was found.", sample.image, "", sample.collection))
            invalid_images.add(sample.image)
            continue
        mask_path = Path(sample.mask)
        if not mask_path.is_file():
            issues.append(ValidationIssue("error", "missing_mask", "Combined semantic mask file does not exist.", sample.image, sample.mask, sample.collection))
            invalid_images.add(sample.image)
            continue
        try:
            with Image.open(mask_path) as im:
                if im.format != "PNG":
                    issues.append(ValidationIssue("error", "mask_not_png", "Training masks must be lossless PNG files.", sample.image, sample.mask, sample.collection))
                    invalid_images.add(sample.image)
                    continue
                if im.mode != "L":
                    issues.append(ValidationIssue("error", "mask_not_single_channel", f"Expected single-channel grayscale PNG, got mode {im.mode}.", sample.image, sample.mask, sample.collection))
                    invalid_images.add(sample.image)
                    continue
                mask = np.asarray(im)
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            issues.append(ValidationIssue("error", "invalid_mask", f"Mask cannot be decoded: {exc}", sample.image, sample.mask, sample.collection))
            invalid_images.add(sample.image)
            continue
        if tuple(mask.shape[::-1]) != image_size:
            issues.append(ValidationIssue("error", "dimension_mismatch", f"Image size {image_size} does not match mask size {(mask.shape[1], mask.shape[0])}.", sample.image, sample.mask, sample.collection))
            invalid_images.add(sample.image)
            continue
        found = set(int(x) for x in np.unique(mask))
        allowed = VALID_CLASS_IDS | ({ignore_label} if ignore_label is not None else set())
        unexpected = found - allowed
        if unexpected:
            issues.append(ValidationIssue("error", "unexpected_class_ids", f"Mask contains values {sorted(unexpected)}; expected classes 0, 1, 2" + (f" and ignore {ignore_label}" if ignore_label is not None else "."), sample.image, sample.mask, sample.collection))
            invalid_images.add(sample.image)
            continue
        counts = np.bincount(mask.reshape(-1), minlength=max(VALID_CLASS_IDS) + 1)
        for class_id in VALID_CLASS_IDS:
            pixels[class_id] += int(counts[class_id])
            if counts[class_id] > 0:
                class_images[class_id] += 1
        if not found.intersection(VALID_CLASS_IDS):
            issues.append(ValidationIssue("warning", "empty_mask", "Mask contains no labeled foreground class pixels.", sample.image, sample.mask, sample.collection))
        valid.append(sample)

    for folded, paths in filename_locations.items():
        if len(paths) > 1:
            for path in paths:
                issues.append(ValidationIssue("warning", "duplicate_filename", f"Filename is shared by {len(paths)} samples across collections; preparation will make names unique.", path, "", str(Path(path).parents[3])))
    duplicates = {h: paths for h, paths in hashes.items() if len(paths) > 1}
    for paths in duplicates.values():
        for path in paths:
            issues.append(ValidationIssue("warning", "duplicate_image", f"Byte-identical image appears in {len(paths)} source locations; copies will stay in the same split group.", path, "", ""))
    return ValidationReport(len(samples), valid, issues, dict(pixels), dict(class_images), dict(dimensions), duplicates)


def export_validation_report(report: ValidationReport, destination: str | Path) -> Path:
    """Write the complete validation result, including every issue and valid sample, as JSON."""
    target = Path(destination).expanduser().resolve()
    if target.suffix.lower() != ".json":
        target = target.with_suffix(".json")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "checked": report.checked,
        "valid_count": len(report.valid),
        "invalid_count": report.invalid_count,
        "class_pixels": report.class_pixels,
        "class_images": report.class_images,
        "dimensions": report.dimensions,
        "duplicate_image_hashes": report.duplicate_image_hashes,
        "valid_samples": [asdict(sample) for sample in report.valid],
        "issues": [asdict(issue) for issue in report.issues],
    }
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def collection_rows(collections: Iterable[CollectionSummary]) -> list[dict]:
    return [{k: v for k, v in asdict(item).items() if k != "samples"} for item in collections]
