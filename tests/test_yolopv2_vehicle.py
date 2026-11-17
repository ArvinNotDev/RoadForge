import importlib
import sys
import types

import numpy as np
import pytest


def _teacher_module():
    pytest.importorskip("cv2")
    try:
        import PySide6.QtCore  # noqa: F401
    except ImportError:
        core = types.ModuleType("PySide6.QtCore")
        class DummySignal:
            def __init__(self, *args, **kwargs):
                pass
            def connect(self, *_args, **_kwargs):
                pass
            def emit(self, *_args, **_kwargs):
                pass
        class DummyThread:
            def __init__(self, *_args, **_kwargs):
                pass
        core.QThread = DummyThread
        core.Signal = lambda *args, **kwargs: DummySignal()
        package = types.ModuleType("PySide6")
        package.QtCore = core
        sys.modules["PySide6"] = package
        sys.modules["PySide6.QtCore"] = core
        try:
            module = importlib.import_module("datacollector.teacher_labeling")
        finally:
            sys.modules.pop("PySide6.QtCore", None)
            sys.modules.pop("PySide6", None)
        return module
    return importlib.import_module("datacollector.teacher_labeling")


def _detection_outputs():
    heads = []
    for grid_size, cell, class_id in (
        (4, (1, 1), 2),  # car
        (2, (0, 0), 7),  # truck
        (1, (0, 0), None),
    ):
        head = np.full((1, 255, grid_size, grid_size), -12.0, dtype=np.float32)
        if class_id is not None:
            row, column = cell
            head[0, 0:4, row, column] = 0.0
            head[0, 4, row, column] = 12.0
            head[0, 5 + class_id, row, column] = 12.0
        heads.append(head)
    anchors = np.asarray([12, 10, 30, 25, 40, 35], dtype=np.float32).reshape(1, 3, 1, 1, 2)
    return [heads, anchors, anchors, anchors, None, None]


def test_yolopv2_detection_decoder_extracts_only_supported_vehicle_classes():
    teacher = _teacher_module()
    outputs = _detection_outputs()

    detections = teacher.decode_vehicle_detections(
        outputs,
        input_size=(64, 64),
        original_size=(128, 96),
        score_threshold=0.30,
        iou_threshold=0.45,
    )

    assert {item["class_id"] for item in detections} == {2, 7}
    assert {item["class_name"] for item in detections} == {"car", "truck"}
    assert all(0.99 < item["score"] <= 1.0 for item in detections)
    for item in detections:
        x1, y1, x2, y2 = item["bbox"]
        assert 0 <= x1 < x2 <= 128
        assert 0 <= y1 < y2 <= 96


def test_vehicle_annotation_defaults_use_stricter_thresholds_and_bottom_crop():
    teacher = _teacher_module()

    assert teacher.DEFAULT_VEHICLE_SCORE_THRESHOLD == pytest.approx(0.50)
    assert teacher.DEFAULT_VEHICLE_NMS_THRESHOLD == pytest.approx(0.40)
    assert teacher.DEFAULT_VEHICLE_BOTTOM_KEEP_RATIO == pytest.approx(0.78)


def test_decoder_discards_lower_region_and_clips_boxes_crossing_cutoff():
    teacher = _teacher_module()
    heads = []
    for grid_size in (16, 8, 4):
        heads.append(np.full((1, 255, grid_size, grid_size), -12.0, dtype=np.float32))

    # A vehicle box crossing the 78% cutoff: retain its upper part.
    heads[0][0, 0:4, 12, 5] = 0.0
    heads[0][0, 4, 12, 5] = 12.0
    heads[0][0, 5 + 2, 12, 5] = 12.0

    # A hood-like detection entirely in the lower 22%: discard it.
    heads[0][0, 0:4, 15, 10] = 0.0
    heads[0][0, 4, 15, 10] = 12.0
    heads[0][0, 5 + 7, 15, 10] = 12.0

    anchors = np.asarray(
        [12, 10, 30, 25, 40, 35], dtype=np.float32
    ).reshape(1, 3, 1, 1, 2)
    outputs = [heads, anchors, anchors, anchors, None, None]

    detections = teacher.decode_vehicle_detections(
        outputs,
        input_size=(128, 128),
        original_size=(128, 128),
    )

    assert len(detections) == 1
    assert detections[0]["class_id"] == 2
    x1, y1, x2, y2 = detections[0]["bbox"]
    assert 0 <= x1 < x2 <= 128
    assert 0 <= y1 < y2 <= 100
    assert y1 < 100 and y2 == 100


def test_vehicle_mask_is_exported_as_semantic_class_three():
    teacher = _teacher_module()
    rgb = np.zeros((40, 60, 3), dtype=np.uint8)
    detections = [
        {"bbox": (10, 8, 30, 28), "score": 0.95, "class_id": 2, "class_name": "car"}
    ]

    vehicle, fallbacks = teacher.build_vehicle_mask(rgb, detections, mask_method="box")
    combined = np.zeros((40, 60), dtype=np.uint8)
    combined[vehicle > 0] = 3

    assert fallbacks == 0
    assert vehicle.shape == (40, 60)
    assert np.all(combined[8:28, 10:30] == 3)
    assert set(np.unique(combined)) == {0, 3}


def test_grabcut_small_vehicle_box_uses_reported_box_fallback():
    teacher = _teacher_module()
    rgb = np.zeros((24, 24, 3), dtype=np.uint8)
    detections = [
        {"bbox": (5, 5, 10, 10), "score": 0.9, "class_id": 3, "class_name": "motorcycle"}
    ]

    vehicle, fallbacks = teacher.build_vehicle_mask(rgb, detections, mask_method="grabcut")

    assert fallbacks == 1
    assert int(vehicle.sum()) > 0
    assert set(np.unique(vehicle)).issubset({0, 255})


def test_four_class_annotation_validator_accepts_vehicle_lane_overlap(tmp_path):
    teacher = _teacher_module()
    cv2 = __import__("cv2")
    height, width = 16, 20
    drive = np.zeros((height, width), dtype=np.uint8)
    lane = np.zeros_like(drive)
    vehicle = np.zeros_like(drive)
    combined = np.zeros_like(drive)

    drive[4:12, 3:17] = 255
    lane[7:9, 3:17] = 255
    vehicle[5:11, 8:14] = 255
    combined[drive > 0] = 1
    combined[lane > 0] = 2
    combined[vehicle > 0] = 3

    files = {
        "drivable_mask": "drivable.png",
        "lane_mask": "lane.png",
        "vehicle_mask": "vehicles.png",
        "combined_mask": "combined.png",
    }
    for key, name in files.items():
        mask = {
            "drivable_mask": drive,
            "lane_mask": lane,
            "vehicle_mask": vehicle,
            "combined_mask": combined,
        }[key]
        assert cv2.imwrite(str(tmp_path / name), mask)

    record = {
        **files,
        "width": width,
        "height": height,
        "class_map": {"0": "background", "1": "ego_drivable_area", "2": "lane_lines", "3": "vehicles"},
        "vehicle_annotation": {
            "pipeline_version": teacher.DEFAULT_VEHICLE_ANNOTATION_PIPELINE_VERSION,
            "confidence_threshold": 0.50,
            "nms_iou_threshold": 0.40,
            "bottom_keep_height_ratio": 0.78,
        },
    }
    assert teacher.annotation_outputs_valid(tmp_path, record)

    # Existing four-class labels without the new filter version must be marked
    # stale so the annotation worker regenerates them using the current defaults.
    legacy_record = dict(record)
    legacy_record["vehicle_annotation"] = {
        "confidence_threshold": 0.30,
        "nms_iou_threshold": 0.45,
    }
    assert not teacher.annotation_outputs_valid(tmp_path, legacy_record)
