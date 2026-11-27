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


def test_session_split_keeps_each_session_in_one_partition():
    samples = []
    for session in range(5):
        for index in range(3):
            samples.append({
                "sample_id": f"{session}-{index}",
                "session_id": f"session-{session}",
            })
    split, warning = split_samples(samples, seed=7)
    assert warning is None
    assert all(split.values())
    membership = {}
    for partition, rows in split.items():
        for row in rows:
            session = row["session_id"]
            assert session not in membership or membership[session] == partition
            membership[session] = partition


def test_single_session_uses_chronological_split_with_boundary_embargo():
    from datetime import datetime, timedelta, timezone

    start = datetime(2026, 11, 1, tzinfo=timezone.utc)
    samples = []
    for index in range(100):
        samples.append({
            "sample_id": f"frame-{index}",
            "session_id": "one-long-avis-session",
            "captured_at": (start + timedelta(seconds=index)).isoformat(),
            "labels": {
                "obstacle_ahead": "yes" if index % 2 else "no",
                "traffic_sign": "stop" if index % 5 == 0 else "no_sign",
            },
        })

    split, warning = split_samples(samples, seed=123)
    assert warning and "chronological" in warning
    indices = {
        part: {
            int(row["sample_id"].removeprefix("frame-"))
            for row in rows
        }
        for part, rows in split.items()
    }
    assert all(indices.values())
    assert not (indices["train"] & indices["val"])
    assert not (indices["train"] & indices["test"])
    assert not (indices["val"] & indices["test"])
    assert max(indices["train"]) < min(indices["val"])
    assert max(indices["val"]) < min(indices["test"])
    # The samples touching both same-session boundaries are embargoed.
    assert min(indices["val"]) - max(indices["train"]) >= 3
    assert min(indices["test"]) - max(indices["val"]) >= 3


def test_legacy_temporal_split_uses_a_multi_sample_embargo_without_timestamps():
    samples = []
    for index in range(100):
        samples.append({
            "sample_id": f"legacy-{index}",
            "session_id": "legacy-session",
            "labels": {
                "obstacle_ahead": "yes" if index % 2 else "no",
                "traffic_sign": "stop" if index % 5 == 0 else "no_sign",
            },
        })

    split, warning = split_samples(samples, seed=21)
    assert warning and "sequence-order window" in warning
    positions = {
        part: {
            int(row["sample_id"].removeprefix("legacy-"))
            for row in rows
        }
        for part, rows in split.items()
    }
    assert all(positions.values())
    assert max(positions["train"]) < min(positions["val"])
    assert max(positions["val"]) < min(positions["test"])
    assert min(positions["val"]) - max(positions["train"]) >= 7
    assert min(positions["test"]) - max(positions["val"]) >= 7


def test_temporal_fallback_deduplicates_identical_images_across_partitions():
    from datetime import datetime, timedelta, timezone

    start = datetime(2026, 11, 1, tzinfo=timezone.utc)
    samples = []
    for index in range(100):
        samples.append({
            "sample_id": f"frame-{index}",
            "session_id": "one-session",
            "captured_at": (start + timedelta(seconds=index)).isoformat(),
            "sha256": f"hash-{index}",
            "labels": {
                "obstacle_ahead": "yes" if index % 2 else "no",
                "traffic_sign": "stop" if index % 5 == 0 else "no_sign",
            },
        })
    samples[95]["sha256"] = samples[0]["sha256"]

    split, warning = split_samples(samples)
    assert warning
    occurrences = [
        row for rows in split.values()
        for row in rows
        if row["sha256"] == samples[0]["sha256"]
    ]
    assert len(occurrences) == 1
    assert any(row["sample_id"] == "frame-0" for row in split["train"])


def test_session_split_keeps_classes_unique_to_one_session_in_training():
    samples = []
    for session in range(5):
        for frame in range(10):
            samples.append({
                "sample_id": f"{session}-{frame}",
                "session_id": f"session-{session}",
                "labels": {
                    "obstacle_ahead": "yes" if frame % 2 else "no",
                    "traffic_sign": (
                        "stop" if session == 0 and frame < 2
                        else "speed_limit" if session > 0 and frame < 2
                        else "no_sign"
                    ),
                },
            })

    split, audit = analyze_classification_split(samples, validate_tasks(TASKS), seed=7)
    assert audit["strategy"] == "session_isolated"
    assert not audit["errors"]
    train_classes = {
        row["labels"]["traffic_sign"] for row in split["train"]
    }
    assert {"stop", "speed_limit", "no_sign"} <= train_classes

    membership = {}
    for part, rows in split.items():
        for row in rows:
            session = row["session_id"]
            assert session not in membership or membership[session] == part
            membership[session] = part
    assert set(membership.values()) == {"train", "val", "test"}


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




def test_resnet18_local_pt_checkpoint_initializes_backbone_and_ignores_old_head(tmp_path):
    torch = pytest.importorskip("torch")
    torchvision = pytest.importorskip("torchvision")
    from roadforge.classification_training import build_multitask_classifier

    source = torchvision.models.resnet18(weights=None)
    state = source.state_dict()
    state["conv1.weight"] = torch.full_like(state["conv1.weight"], 0.125)
    state["fc.weight"] = torch.randn(7, source.fc.in_features)
    state["fc.bias"] = torch.randn(7)
    checkpoint = tmp_path / "prepared_resnet18.pt"
    torch.save({"state_dict": {f"module.backbone.{key}": value for key, value in state.items()}}, checkpoint)

    tasks = [{"key": "sign", "name": "Sign", "classes": ["right", "left", "straight"]}]
    model = build_multitask_classifier(
        "resnet18", tasks, pretrained=False, initial_weights_path=checkpoint
    )

    torch.testing.assert_close(
        model.feature_extractor.conv1.weight,
        torch.full_like(model.feature_extractor.conv1.weight, 0.125),
    )
    assert model.heads[0].out_features == 3
    assert model.initialization_info["source"] == "local_resnet18_pt"
    assert model.initialization_info["checkpoint_sha256"]
    assert model.initialization_info["loaded_backbone_tensors"] > 100


def test_resnet18_local_pt_checkpoint_rejects_non_resnet_weights(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from roadforge.classification_training import build_multitask_classifier

    checkpoint = tmp_path / "not_resnet18.pt"
    torch.save({"state_dict": {"classifier.weight": torch.ones(3, 4)}}, checkpoint)
    with pytest.raises(ValueError, match="not a compatible ResNet18"):
        build_multitask_classifier(
            "resnet18",
            [{"key": "sign", "name": "Sign", "classes": ["right", "left"]}],
            pretrained=False,
            initial_weights_path=checkpoint,
        )


def test_local_pt_initialization_is_rejected_for_efficientnet(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from roadforge.classification_training import build_multitask_classifier

    checkpoint = tmp_path / "resnet18.pt"
    torch.save({"state_dict": {}}, checkpoint)
    with pytest.raises(ValueError, match="only for ResNet18"):
        build_multitask_classifier(
            "efficientnet_b0",
            [{"key": "sign", "name": "Sign", "classes": ["right", "left"]}],
            pretrained=False,
            initial_weights_path=checkpoint,
        )


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
