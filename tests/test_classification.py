from __future__ import annotations

import json

import pytest
from PIL import Image

from roadforge.classification_data import (
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
