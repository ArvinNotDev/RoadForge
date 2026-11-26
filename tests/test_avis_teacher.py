from __future__ import annotations

from pathlib import Path

import pytest


def test_teacher_discovery_accepts_avis_capture_folder_without_collection_manifest(tmp_path):
    pytest.importorskip("PySide6")
    from datacollector.teacher_labeling import discover_collections

    capture_root = tmp_path / "AVIS_RoadForge_Captures"
    image_dir = capture_root / "images"
    image_dir.mkdir(parents=True)
    (image_dir / "frame_0001.jpg").write_bytes(b"not decoded during discovery")
    (capture_root / "manifest.jsonl").write_text(
        '{"image":"images/frame_0001.jpg","session_id":"avis-session"}\n',
        encoding="utf-8",
    )

    discovered = discover_collections(capture_root)
    assert list(discovered) == [capture_root]
    assert set(discovered[capture_root]) == {"AVIS"}
    assert discovered[capture_root]["AVIS"] == [image_dir / "frame_0001.jpg"]


def test_teacher_discovery_accepts_parent_folder_containing_avis_capture_folder(tmp_path):
    pytest.importorskip("PySide6")
    from datacollector.teacher_labeling import discover_collections

    capture_root = tmp_path / "AVIS_RoadForge_Captures"
    image_dir = capture_root / "images"
    image_dir.mkdir(parents=True)
    (image_dir / "frame_0001.jpg").write_bytes(b"not decoded during discovery")

    discovered = discover_collections(tmp_path)
    assert capture_root in discovered
    assert discovered[capture_root]["AVIS"] == [image_dir / "frame_0001.jpg"]

def _make_four_class_teacher_annotation(tmp_path):
    pytest.importorskip("PySide6")
    import cv2
    import json
    import numpy as np

    from datacollector.teacher_labeling import (
        DEFAULT_VEHICLE_ANNOTATION_PIPELINE_VERSION,
    )

    collection = tmp_path / "AVIS_RoadForge_Captures"
    image_dir = collection / "images"
    image_dir.mkdir(parents=True)
    image = image_dir / "frame_0001.jpg"
    image.write_bytes(b"test image placeholder")

    output_root = collection / "labels_yolopv2" / "AVIS"
    drive = np.array([[255, 255, 255, 0], [0, 0, 0, 0], [0, 255, 0, 0]], dtype=np.uint8)
    lane = np.array([[0, 0, 255, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8)
    vehicle = np.array([[0, 0, 255, 0], [255, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8)
    combined = np.zeros_like(drive)
    combined[drive > 0] = 1
    combined[lane > 0] = 2
    combined[vehicle > 0] = 3

    paths = {
        "drivable_mask": output_root / "drivable_area" / "frame_0001.png",
        "lane_mask": output_root / "lane_lines" / "frame_0001.png",
        "vehicle_mask": output_root / "vehicle_masks" / "frame_0001.png",
        "combined_mask": output_root / "combined_4class" / "frame_0001.png",
    }
    for key, mask in (
        ("drivable_mask", drive), ("lane_mask", lane),
        ("vehicle_mask", vehicle), ("combined_mask", combined),
    ):
        paths[key].parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(paths[key]), mask)

    record = {
        "status": "success",
        "original_image": "images/frame_0001.jpg",
        **{key: path.relative_to(collection).as_posix() for key, path in paths.items()},
        "class_map": {
            "0": "background", "1": "ego_drivable_area",
            "2": "lane_lines", "3": "vehicles",
        },
        "num_classes": 4,
        "vehicle_annotation": {
            "pipeline_version": DEFAULT_VEHICLE_ANNOTATION_PIPELINE_VERSION,
        },
        "width": 4,
        "height": 3,
    }
    manifest = {"format": "yolopv2-annotations-v1", "images": {"images/frame_0001.jpg": record}}
    (collection / "annotation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return collection, image, paths


def test_vehicle_masks_can_be_mapped_to_background_and_restored(tmp_path):
    pytest.importorskip("PySide6")
    import cv2
    import json
    import numpy as np

    from datacollector.teacher_labeling import (
        MaskPostprocessWorker,
        annotation_outputs_valid,
    )

    collection, image, paths = _make_four_class_teacher_annotation(tmp_path)
    manifest_path = collection / "annotation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert annotation_outputs_valid(collection, manifest["images"]["images/frame_0001.jpg"])

    worker = MaskPostprocessWorker(
        [(collection, "AVIS", image)],
        MaskPostprocessWorker.VEHICLES_TO_BACKGROUND,
    )
    worker.run()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest["images"]["images/frame_0001.jpg"]
    processed_path = collection / record["combined_mask"]
    processed = cv2.imread(str(processed_path), cv2.IMREAD_GRAYSCALE)
    original = cv2.imread(str(paths["combined_mask"]), cv2.IMREAD_GRAYSCALE)
    assert processed is not None and original is not None
    assert np.count_nonzero(original == 3) == 2
    assert not np.any(processed == 3)
    assert np.array_equal(processed[original != 3], original[original != 3])
    assert record["original_combined_mask"] == paths["combined_mask"].relative_to(collection).as_posix()
    assert annotation_outputs_valid(collection, record)

    restore = MaskPostprocessWorker(
        [(collection, "AVIS", image)],
        MaskPostprocessWorker.RESTORE_ORIGINAL,
    )
    restore.run()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    restored_record = manifest["images"]["images/frame_0001.jpg"]
    assert restored_record["combined_mask"] == paths["combined_mask"].relative_to(collection).as_posix()
    assert "original_combined_mask" not in restored_record
    assert "postprocessing" not in restored_record
    assert annotation_outputs_valid(collection, restored_record)

