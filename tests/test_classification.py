from __future__ import annotations

import json

import numpy as np

import pytest
from PIL import Image

from roadforge.classification_data import (
    analyze_classification_split,
    annotated_samples,
    ensure_dataset,
    import_avis_captures,
    load_dataset_config,
    load_samples,
    save_sample_labels,
    split_samples,
    validate_tasks,
)


TASKS = [
    {"name": "Obstacle ahead", "classes": ["no", "yes"]},
    {"name": "Traffic sign", "classes": ["no_sign", "stop", "speed_limit"]},
]


def _create_capture_set(root, sessions=3, images_per_session=2):
    images = root / "images"
    images.mkdir(parents=True)
    rows = []
    for session_index in range(sessions):
        for frame_index in range(images_per_session):
            filename = f"frame_{session_index}_{frame_index}.jpg"
            Image.new(
                "RGB",
                (48, 32),
                (session_index * 45 + 20, frame_index * 70 + 15, 90),
            ).save(images / filename)
            rows.append({
                "image": f"images/{filename}",
                "filename": filename,
                "session_id": f"avis-session-{session_index}",
                "captured_at": f"2026-11-{session_index + 1:02d}T10:00:00+03:30",
                "telemetry": {"speed_kmh": 8 + frame_index},
            })
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_task_schema_requires_one_to_three_tasks_and_unique_classes():
    assert len(validate_tasks(TASKS)) == 2
    with pytest.raises(ValueError):
        validate_tasks([])
    with pytest.raises(ValueError):
        validate_tasks([{"name": "bad", "classes": ["yes", "YES"]}])


def test_capture_import_annotation_and_exact_duplicate_handling(tmp_path):
    capture_root = tmp_path / "AVIS_RoadForge_Captures"
    dataset_root = tmp_path / "dataset"
    _create_capture_set(capture_root)
    ensure_dataset(dataset_root, TASKS)

    result = import_avis_captures(capture_root, dataset_root)
    assert result == {"discovered": 6, "added": 6, "duplicates_skipped": 0}
    assert len(load_samples(dataset_root)) == 6

    duplicate_result = import_avis_captures(capture_root, dataset_root)
    assert duplicate_result["added"] == 0
    assert duplicate_result["duplicates_skipped"] == 6

    for index, sample in enumerate(load_samples(dataset_root)):
        labels = {
            "obstacle_ahead": "yes" if index % 2 else "no",
            "traffic_sign": "stop" if index % 3 == 0 else "no_sign",
        }
        save_sample_labels(dataset_root, sample["sample_id"], labels)

    assert len(annotated_samples(dataset_root)) == 6
    assert len(load_dataset_config(dataset_root)["tasks"]) == 2


def test_image_stratified_split_balances_rare_classes_across_all_partitions():
    class_counts = {
        "right": 135,
        "left": 105,
        "straight": 116,
        "stop": 58,
        "90r": 27,
        "90l": 53,
    }
    samples = []
    sample_index = 0
    for class_name, count in class_counts.items():
        for frame in range(count):
            if class_name in {"90r", "90l"}:
                session_id = "rare-sign-session"
            else:
                session_id = "common-session-a" if frame % 2 else "common-session-b"
            samples.append({
                "sample_id": f"frame-{sample_index}",
                "session_id": session_id,
                "sha256": f"unique-hash-{sample_index}",
                "labels": {"sign": class_name},
            })
            sample_index += 1

    tasks = validate_tasks([{"name": "Sign", "classes": list(class_counts)}])
    split, audit = analyze_classification_split(samples, tasks, seed=7)

    assert audit["strategy"] == "image_stratified"
    assert not audit["errors"]
    assert audit["counts"]["train"] + audit["counts"]["val"] + audit["counts"]["test"] == 494
    assert 0.78 <= audit["actual_ratios"]["train"] <= 0.82
    assert 0.08 <= audit["actual_ratios"]["val"] <= 0.12
    assert 0.08 <= audit["actual_ratios"]["test"] <= 0.12
    for part in ("train", "val", "test"):
        assert all(
            audit["class_distribution"][part]["sign"][name] > 0
            for name in class_counts
        )

    # The rare classes are present in only one capture session, but are
    # intentionally distributed across partitions for the fixed-road use case.
    assert "rare-sign-session" in audit["session_distribution"]["train"]
    assert "rare-sign-session" in audit["session_distribution"]["val"]
    assert "rare-sign-session" in audit["session_distribution"]["test"]
    assert "rare-sign-session" in audit["shared_sessions_across_splits"]
    assert audit["excluded_near_boundaries"] == 0
    assert any("unseen-session generalization" in item for item in audit["warnings"])


def test_image_stratified_split_keeps_exact_duplicate_images_together():
    samples = []
    classes = ["right", "left", "straight"]
    for index in range(30):
        samples.append({
            "sample_id": f"frame-{index}",
            "session_id": "one-session",
            "sha256": f"hash-{index}",
            "labels": {"sign": classes[index % len(classes)]},
        })
    samples.append({
        "sample_id": "duplicate-of-frame-0",
        "session_id": "another-session",
        "sha256": "hash-0",
        "labels": {"sign": "right"},
    })

    first, warning = split_samples(samples, seed=21)
    second, second_warning = split_samples(samples, seed=21)
    assert warning and "Image-level stratification" in warning
    assert second_warning == warning

    membership = {}
    second_membership = {}
    for part, rows in first.items():
        for row in rows:
            membership[row["sample_id"]] = part
    for part, rows in second.items():
        for row in rows:
            second_membership[row["sample_id"]] = part
    assert membership["frame-0"] == membership["duplicate-of-frame-0"]
    assert membership == second_membership
    assert all(first.values())



def test_schema_cannot_be_changed_after_annotations_exist(tmp_path):
    dataset_root = tmp_path / "dataset"
    ensure_dataset(dataset_root, TASKS)
    image_path = dataset_root / "images" / "sample.jpg"
    Image.new("RGB", (16, 16)).save(image_path)
    sample = {
        "sample_id": "sample",
        "image": "images/sample.jpg",
        "session_id": "s1",
        "labels": {},
    }
    (dataset_root / "samples.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")
    save_sample_labels(
        dataset_root,
        "sample",
        {"obstacle_ahead": "no", "traffic_sign": "no_sign"},
    )
    with pytest.raises(ValueError):
        ensure_dataset(
            dataset_root,
            [{"name": "Different task", "classes": ["a", "b"]}],
            allow_task_update=True,
        )


@pytest.mark.parametrize("backbone", ["resnet18", "efficientnet_b0"])
def test_supported_backbones_produce_one_output_per_task(backbone):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from roadforge.classification_training import build_multitask_classifier

    model = build_multitask_classifier(backbone, validate_tasks(TASKS), pretrained=False)
    outputs = model(torch.rand(2, 3, 224, 224))
    assert len(outputs) == 2
    assert tuple(outputs[0].shape) == (2, 2)
    assert tuple(outputs[1].shape) == (2, 3)



def test_resnet18_multitask_checkpoint_exports_to_onnx(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("onnx")
    onnxruntime = pytest.importorskip("onnxruntime")

    from roadforge.classification_training import build_multitask_classifier
    from roadforge.core.runtime import export_onnx

    tasks = [
        {"key": "obstacle_ahead", "name": "Obstacle ahead", "classes": ["no", "yes"]},
        {"key": "lane", "name": "Lane", "classes": ["right", "left"]},
    ]
    model = build_multitask_classifier("resnet18", tasks, pretrained=False)
    checkpoint = tmp_path / "avis_resnet18_best.pt"
    torch.save(
        {
            "format_version": 1,
            "backbone": "resnet18",
            "tasks": tasks,
            "input_size": 224,
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
            "state_dict": model.state_dict(),
        },
        checkpoint,
    )

    result = export_onnx(
        str(checkpoint),
        imgsz=512,  # The classification checkpoint's saved 224px input must take precedence.
        opset=17,
        dynamic=True,
        output_dir=tmp_path / "exports",
    )

    assert result["task"] == "classification"
    assert result["backbone"] == "resnet18"
    assert result["input_size"] == 224
    assert result["dynamic_batch"] is True
    assert result["dynamic_spatial"] is False
    assert result["onnx_checker"] == "passed"
    assert result["smoke_test"] == "passed"
    assert len(result["pytorch_onnx_parity"]) == 2
    assert all(item["allclose"] for item in result["pytorch_onnx_parity"])
    assert [item["key"] for item in result["task_outputs"]] == ["obstacle_ahead", "lane"]
    assert [item["classes"] for item in result["task_outputs"]] == [["no", "yes"], ["right", "left"]]

    session = onnxruntime.InferenceSession(
        result["path"], providers=["CPUExecutionProvider"]
    )
    metadata = session.get_modelmeta().custom_metadata_map
    assert metadata["task"] == "classification"
    assert metadata["backbone"] == "resnet18"
    assert metadata["input_size"] == "224"
    assert [output.name for output in session.get_outputs()] == [
        "logits_obstacle_ahead",
        "logits_lane",
    ]
    values = session.run(
        None, {"images": np.zeros((1, 3, 224, 224), dtype=np.float32)}
    )
    assert [tuple(item.shape) for item in values] == [(1, 2), (1, 2)]
