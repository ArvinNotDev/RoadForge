from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from roadforge.core.dataset import Sample, discover_collections, validate_samples
from roadforge.core.splits import prepare_dataset, split_samples


def _write_avis_capture_set(root: Path, count: int = 1, session_id: str = "avis-run-01"):
    image_dir = root / "images"
    image_dir.mkdir(parents=True)
    records = []
    for index in range(count):
        filename = f"avis_{index:04d}.jpg"
        rgb = np.zeros((12, 16, 3), dtype=np.uint8)
        rgb[..., 0] = 20 + index
        rgb[..., 1] = 80
        rgb[..., 2] = 140
        Image.fromarray(rgb, mode="RGB").save(image_dir / filename)
        records.append({
            "image": f"images/{filename}",
            "filename": filename,
            "captured_at": f"2026-11-26T14:{index % 60:02d}:00+03:30",
            "session_id": session_id,
            "camera_view": "front_rgb",
            "mode": "manual",
            "telemetry": {"speed_kmh": 8.0},
        })
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_avis_capture_folder_is_discovered_without_carla_metadata(tmp_path):
    capture_root = tmp_path / "AVIS_RoadForge_Captures"
    _write_avis_capture_set(capture_root, count=2)

    collections = discover_collections([tmp_path])
    assert len(collections) == 1
    collection = collections[0]
    assert collection.name == "AVIS_RoadForge_Captures"
    assert collection.towns == ["AVIS"]
    assert collection.image_count == 2
    assert collection.annotated_count == 0
    assert {sample.session_id for sample in collection.samples} == {"avis-run-01"}
    assert all(sample.metadata["camera_view"] == "front_rgb" for sample in collection.samples)
    assert all(sample.weather == "Unknown" for sample in collection.samples)


def test_avis_teacher_manifest_pairs_four_class_masks_without_collection_manifest(tmp_path):
    capture_root = tmp_path / "AVIS_RoadForge_Captures"
    _write_avis_capture_set(capture_root, count=1)
    image_rel = "images/avis_0000.jpg"

    mask_path = capture_root / "labels_yolopv2" / "AVIS" / "combined_4class" / "avis_0000.png"
    mask_path.parent.mkdir(parents=True)
    mask = np.tile(np.array([[0, 1, 2, 3]], dtype=np.uint8), (12, 4))
    Image.fromarray(mask, mode="L").save(mask_path)
    annotation = {
        "format": "yolopv2-annotations-v1",
        "images": {
            image_rel: {
                "status": "success",
                "original_image": image_rel,
                "combined_mask": "labels_yolopv2/AVIS/combined_4class/avis_0000.png",
                "class_map": {
                    "0": "background",
                    "1": "ego_drivable_area",
                    "2": "lane_lines",
                    "3": "vehicles",
                },
                "num_classes": 4,
            }
        },
    }
    (capture_root / "annotation_manifest.json").write_text(
        json.dumps(annotation, indent=2),
        encoding="utf-8",
    )

    collections = discover_collections([capture_root])
    assert len(collections) == 1
    sample = collections[0].samples[0]
    assert sample.mask == str(mask_path)
    assert sample.town == "AVIS"
    report = validate_samples([sample])
    assert len(report.valid) == 1
    assert not [issue for issue in report.issues if issue.severity == "error"]


def test_single_avis_session_falls_back_to_reproducible_image_level_splits():
    samples = [
        Sample(
            image=f"frame_{index}.jpg",
            mask=f"mask_{index}.png",
            collection="AVIS_RoadForge_Captures",
            town="AVIS",
            session_id="one-avis-process",
            group_id="one-avis-process",
            filename=f"frame_{index}.jpg",
            content_hash=f"unique-hash-{index}",
        )
        for index in range(12)
    ]
    first = split_samples(samples, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=42)
    second = split_samples(samples, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=42)

    assert first.strategy == "image_level_fallback"
    assert first.warning and "neighbouring video frames" in first.warning
    assert first.counts == second.counts
    assert {
        name: [sample.image for sample in getattr(first, name)]
        for name in ("train", "val", "test")
    } == {
        name: [sample.image for sample in getattr(second, name)]
        for name in ("train", "val", "test")
    }
    assert all(first.counts[name] > 0 for name in ("train", "val", "test"))


def test_duplicate_avis_frames_stay_in_one_fallback_split():
    samples = [
        Sample(
            image=f"frame_{index}.jpg",
            mask=f"mask_{index}.png",
            collection="AVIS_RoadForge_Captures",
            town="AVIS",
            session_id="one-avis-process",
            group_id="one-avis-process",
            filename=f"frame_{index}.jpg",
            content_hash=("same-frame-hash" if index in (0, 1) else f"hash-{index}"),
        )
        for index in range(10)
    ]
    result = split_samples(samples, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=17)
    locations = {}
    for part in ("train", "val", "test"):
        for sample in getattr(result, part):
            locations[sample.content_hash] = locations.get(sample.content_hash, part)
            assert locations[sample.content_hash] == part


def test_prepare_dataset_persists_avis_image_level_split_warning(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    samples = []
    for index in range(12):
        image_path = source / f"frame_{index:04d}.jpg"
        mask_path = source / f"mask_{index:04d}.png"
        rgb = np.zeros((12, 16, 3), dtype=np.uint8)
        rgb[..., 0] = 30 + index
        Image.fromarray(rgb, mode="RGB").save(image_path)
        mask = np.tile(np.array([[0, 1, 2, 3]], dtype=np.uint8), (12, 4))
        Image.fromarray(mask, mode="L").save(mask_path)
        samples.append(Sample(
            image=str(image_path),
            mask=str(mask_path),
            collection="AVIS_RoadForge_Captures",
            town="AVIS",
            session_id="one-avis-process",
            group_id="one-avis-process",
            filename=image_path.name,
        ))

    result = prepare_dataset(
        samples,
        tmp_path / "prepared" / "avis",
        {"train": 0.8, "val": 0.1, "test": 0.1},
        seed=42,
    )
    report = json.loads((Path(result["path"]) / "dataset_report.json").read_text(encoding="utf-8"))
    split_manifest = json.loads((Path(result["path"]) / "split_manifest.json").read_text(encoding="utf-8"))

    assert result["counts"]["train"] > 0
    assert result["counts"]["val"] > 0
    assert result["counts"]["test"] > 0
    assert report["split_strategy"] == "image_level_fallback"
    assert any("neighbouring video frames" in warning for warning in report["warnings"])
    assert split_manifest["strategy"] == "image_level_fallback"

def test_avis_postprocessed_vehicle_to_background_mask_is_valid(tmp_path):
    capture_root = tmp_path / "AVIS_RoadForge_Captures"
    _write_avis_capture_set(capture_root, count=1)
    image_rel = "images/avis_0000.jpg"

    # The source vehicle mask still records the Teacher's detections, but the
    # active combined training mask maps those pixels to Background.
    output_root = capture_root / "labels_yolopv2" / "AVIS"
    mask_path = output_root / "combined_4class_no_vehicles" / "avis_0000.png"
    mask_path.parent.mkdir(parents=True)
    mask = np.tile(np.array([[0, 1, 2, 0]], dtype=np.uint8), (12, 4))
    Image.fromarray(mask, mode="L").save(mask_path)

    vehicle_mask_path = output_root / "vehicle_masks" / "avis_0000.png"
    vehicle_mask_path.parent.mkdir(parents=True)
    vehicle_mask = np.tile(np.array([[0, 0, 0, 255]], dtype=np.uint8), (12, 4))
    Image.fromarray(vehicle_mask, mode="L").save(vehicle_mask_path)

    original_mask_path = output_root / "combined_4class" / "avis_0000.png"
    original_mask_path.parent.mkdir(parents=True)
    original_mask = np.tile(np.array([[0, 1, 2, 3]], dtype=np.uint8), (12, 4))
    Image.fromarray(original_mask, mode="L").save(original_mask_path)

    annotation = {
        "format": "yolopv2-annotations-v1",
        "images": {
            image_rel: {
                "status": "success",
                "original_image": image_rel,
                "combined_mask": "labels_yolopv2/AVIS/combined_4class_no_vehicles/avis_0000.png",
                "original_combined_mask": "labels_yolopv2/AVIS/combined_4class/avis_0000.png",
                "postprocessing": {
                    "operation": "map_class_to_background",
                    "source_class_id": 3,
                    "target_class_id": 0,
                },
                "class_map": {
                    "0": "background",
                    "1": "ego_drivable_area",
                    "2": "lane_lines",
                    "3": "vehicles",
                },
                "num_classes": 4,
            }
        },
    }
    (capture_root / "annotation_manifest.json").write_text(
        json.dumps(annotation, indent=2),
        encoding="utf-8",
    )

    collections = discover_collections([capture_root])
    assert len(collections) == 1
    sample = collections[0].samples[0]
    assert sample.mask == str(mask_path)
    assert sample.metadata["_annotation_legacy"] is False

    report = validate_samples([sample])
    assert len(report.valid) == 1
    assert not [issue for issue in report.issues if issue.severity == "error"]

def test_avis_collector_annotations_inside_images_subfolder_are_discovered_from_capture_root(tmp_path):
    capture_root = tmp_path / "AVIS_RoadForge_Captures"
    _write_avis_capture_set(capture_root, count=1)
    image_rel = "images/avis_0000.jpg"

    # This layout is produced when the Collector's Teacher source is selected
    # as <capture-root>/images instead of the capture root itself.
    annotation_root = capture_root / "images"
    mask_path = annotation_root / "labels_yolopv2" / "AVIS" / "combined_4class_no_vehicles" / "avis_0000.png"
    mask_path.parent.mkdir(parents=True)
    mask = np.tile(np.array([[0, 1, 2, 0]], dtype=np.uint8), (12, 4))
    Image.fromarray(mask, mode="L").save(mask_path)

    original_mask_path = annotation_root / "labels_yolopv2" / "AVIS" / "combined_4class" / "avis_0000.png"
    original_mask_path.parent.mkdir(parents=True)
    original_mask = np.tile(np.array([[0, 1, 2, 3]], dtype=np.uint8), (12, 4))
    Image.fromarray(original_mask, mode="L").save(original_mask_path)

    annotation = {
        "format": "yolopv2-annotations-v1",
        "images": {
            "avis_0000.jpg": {
                "status": "success",
                "original_image": "avis_0000.jpg",
                "combined_mask": "labels_yolopv2/AVIS/combined_4class_no_vehicles/avis_0000.png",
                "original_combined_mask": "labels_yolopv2/AVIS/combined_4class/avis_0000.png",
                "postprocessing": {
                    "operation": "map_class_to_background",
                    "source_class_id": 3,
                    "target_class_id": 0,
                },
                "class_map": {
                    "0": "background",
                    "1": "ego_drivable_area",
                    "2": "lane_lines",
                    "3": "vehicles",
                },
                "num_classes": 4,
            }
        },
    }
    (annotation_root / "annotation_manifest.json").write_text(
        json.dumps(annotation, indent=2),
        encoding="utf-8",
    )

    collections = discover_collections([capture_root])
    assert len(collections) == 1
    sample = collections[0].samples[0]
    assert Path(sample.image).relative_to(capture_root).as_posix() == image_rel
    assert sample.mask == str(mask_path)
    assert sample.metadata["_annotation_original_image"] == image_rel

    report = validate_samples([sample])
    assert len(report.valid) == 1
    assert not [issue for issue in report.issues if issue.severity == "error"]


def test_avis_collector_annotations_inside_images_subfolder_are_discovered_when_selected_directly(tmp_path):
    capture_root = tmp_path / "AVIS_RoadForge_Captures"
    _write_avis_capture_set(capture_root, count=1)
    image_root = capture_root / "images"
    mask_path = image_root / "labels_yolopv2" / "AVIS" / "combined_4class" / "avis_0000.png"
    mask_path.parent.mkdir(parents=True)
    mask = np.tile(np.array([[0, 1, 2, 3]], dtype=np.uint8), (12, 4))
    Image.fromarray(mask, mode="L").save(mask_path)
    annotation = {
        "images": {
            "avis_0000.jpg": {
                "status": "success",
                "original_image": "avis_0000.jpg",
                "combined_mask": "labels_yolopv2/AVIS/combined_4class/avis_0000.png",
                "class_map": {
                    "0": "background",
                    "1": "ego_drivable_area",
                    "2": "lane_lines",
                    "3": "vehicles",
                },
                "num_classes": 4,
            }
        }
    }
    (image_root / "annotation_manifest.json").write_text(json.dumps(annotation), encoding="utf-8")

    collections = discover_collections([image_root])
    sample = collections[0].samples[0]
    assert sample.mask == str(mask_path)
    assert sample.metadata["_annotation_original_image"] == "avis_0000.jpg"
    report = validate_samples([sample])
    assert len(report.valid) == 1

