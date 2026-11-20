import json
import pytest
from pathlib import Path

import numpy as np
from PIL import Image

from roadforge.core.config import default_config, load_config, save_config
from roadforge.core.dataset import Sample, discover_collections, export_validation_report, validate_samples
from roadforge.core.metrics import confusion_matrix, metrics_from_confusion
from roadforge.core.runtime import _DEPLOYMENT_INFERENCE_SCRIPT, _dataset_manifest_file, _semantic_array_to_mask, evaluate_dataset, run_folder_inference
from roadforge.core.registry import list_models, register_model, remove_model
from roadforge.core.splits import prepare_dataset, split_samples
from roadforge.core.selection import select_quality_balanced_samples, _allocate_quotas


def make_sample(tmp_path: Path, group: str, index: int, mask_values=None) -> Sample:
    collection = tmp_path / f"collection_{group}"
    image_path = collection / "raw" / "Town01" / "images" / f"frame_{index:04d}.png"
    mask_path = collection / "labels_yolopv2" / "Town01" / "combined_4class" / f"frame_{index:04d}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.full((12, 20, 3), index, dtype=np.uint8)
    mask = np.zeros((12, 20), dtype=np.uint8) if mask_values is None else np.asarray(mask_values, dtype=np.uint8)
    if mask_values is None:
        mask[4:10, 2:18] = 1
        mask[5, 4:16] = 2
        mask[7:9, 8:12] = 3
    Image.fromarray(rgb, "RGB").save(image_path)
    Image.fromarray(mask, "L").save(mask_path)
    return Sample(str(image_path), str(mask_path), str(collection), "Town01", group, group, image_path.name, "ClearNoon", "No traffic", f"route-{group}")


def test_validation_accepts_lossless_combined_mask_and_counts_classes(tmp_path):
    sample = make_sample(tmp_path, "a", 1)
    report = validate_samples([sample])
    assert len(report.valid) == 1
    assert not [issue for issue in report.issues if issue.severity == "error"]
    assert set(report.class_pixels) == {0, 1, 2, 3}
    assert report.class_images == {0: 1, 1: 1, 2: 1, 3: 1}


def test_validation_reports_missing_corrupt_dimension_and_unexpected_ids(tmp_path):
    missing = make_sample(tmp_path, "a", 1)
    missing.mask = None
    corrupt = make_sample(tmp_path, "b", 2)
    Path(corrupt.image).write_bytes(b"not an image")
    mismatch = make_sample(tmp_path, "c", 3)
    Image.fromarray(np.zeros((9, 10), dtype=np.uint8), "L").save(mismatch.mask)
    invalid_ids = make_sample(tmp_path, "d", 4, np.full((12, 20), 7, dtype=np.uint8))
    report = validate_samples([missing, corrupt, mismatch, invalid_ids])
    codes = {issue.code for issue in report.issues}
    assert {"missing_mask", "invalid_image", "dimension_mismatch", "unexpected_class_ids"}.issubset(codes)
    ignored = make_sample(tmp_path, "e", 5, np.full((12, 20), 255, dtype=np.uint8))
    assert not validate_samples([ignored]).valid
    assert len(validate_samples([ignored], ignore_label=255).valid) == 1


def test_discovery_uses_manifest_combined_mask_only(tmp_path):
    collection = tmp_path / "collection_20260101"
    image = collection / "raw" / "Town01" / "images" / "frame.png"
    binary = collection / "labels_yolopv2" / "Town01" / "drivable_area" / "frame.png"
    combined = collection / "labels_yolopv2" / "Town01" / "combined_3class" / "frame.png"
    image.parent.mkdir(parents=True); binary.parent.mkdir(parents=True); combined.parent.mkdir(parents=True)
    Image.fromarray(np.zeros((4, 6, 3), dtype=np.uint8), "RGB").save(image)
    Image.fromarray(np.full((4, 6), 255, dtype=np.uint8), "L").save(binary)
    Image.fromarray(np.zeros((4, 6), dtype=np.uint8), "L").save(combined)
    (collection / "collection_manifest.json").write_text(json.dumps({"session_id": "session-a", "config": {"background_traffic": False}}), encoding="utf-8")
    (collection / "annotation_manifest.json").write_text(json.dumps({"images": {"raw/Town01/images/frame.png": {"status": "success", "original_image": "raw/Town01/images/frame.png", "combined_mask": "labels_yolopv2/Town01/combined_3class/frame.png"}}}), encoding="utf-8")
    (collection / "raw" / "Town01" / "metadata.jsonl").write_text(json.dumps({"filename": "frame.png", "town": "Town01", "weather": "ClearNoon", "collection_session_id": "session-a"}) + "\n", encoding="utf-8")
    results = discover_collections([tmp_path])
    assert len(results) == 1
    assert results[0].samples[0].mask == str(combined)
    assert results[0].samples[0].traffic == "No background traffic"


def test_split_is_reproducible_and_keeps_sessions_together(tmp_path):
    samples = [make_sample(tmp_path, f"group-{g}", i) for g in range(7) for i in range(2)]
    ratios = {"train": 0.6, "val": 0.2, "test": 0.2}
    first = split_samples(samples, ratios, seed=42)
    second = split_samples(samples, ratios, seed=42)
    assert first.group_assignment == second.group_assignment
    for part, name in ((first.train, "train"), (first.val, "val"), (first.test, "test")):
        assert part
        assert {s.group_id for s in part} == {key for key, assigned in first.group_assignment.items() if assigned == name}


def test_duplicate_images_merge_groups_and_town_holdout_expands_to_session(tmp_path):
    a = make_sample(tmp_path, "a", 1)
    b = make_sample(tmp_path, "b", 2)
    a.content_hash = b.content_hash = "same-bytes"
    c = make_sample(tmp_path, "c", 3); d = make_sample(tmp_path, "d", 4)
    e = make_sample(tmp_path, "e", 5); f = make_sample(tmp_path, "f", 6)
    for sample in (c, d): sample.town = "Town02"
    result = split_samples([a, b, c, d, e, f], {"train": .5, "val": .25, "test": .25}, 2, holdout_towns={"Town02"})
    assert {s.town for s in result.test} == {"Town02"}
    assert result.group_assignment["a"] == result.group_assignment["b"]
    assert result.group_assignment["c"] == result.group_assignment["d"] == "test"


def test_prepare_dataset_applies_town_holdout_and_counts_groups(tmp_path):
    samples = [make_sample(tmp_path / "source", f"group-{g}", g) for g in range(5)]
    samples[3].town = "Town02"
    samples[4].town = "Town02"
    result = prepare_dataset(samples, tmp_path / "prepared" / "holdout", {"train": .6, "val": .2, "test": .2}, seed=7, holdout_towns={"Town02"})
    manifest = json.loads((Path(result["path"]) / "split_manifest.json").read_text(encoding="utf-8"))
    assert {item["town"] for item in manifest["samples"]["test"]} == {"Town02"}
    assert result["report"]["group_count"] == 5


def test_prepare_copies_data_yaml_and_preserves_source(tmp_path):
    samples = [make_sample(tmp_path / "source", f"group-{g}", g) for g in range(3)]
    original_bytes = {s.image: Path(s.image).read_bytes() for s in samples}
    result = prepare_dataset(samples, tmp_path / "prepared" / "demo", {"train": 1/3, "val": 1/3, "test": 1/3}, seed=4)
    assert result["counts"] == {"train": 1, "val": 1, "test": 1}
    assert Path(result["data_yaml"]).is_file()
    assert "masks_dir: masks" in Path(result["data_yaml"]).read_text(encoding="utf-8")
    yaml_text = Path(result["data_yaml"]).read_text(encoding="utf-8")
    assert "ego_drivable_area" in yaml_text
    assert "vehicles" in yaml_text
    assert result["report"]["num_classes"] == 4
    assert (Path(result["path"]) / "dataset_report.json").is_file()
    assert all(Path(sample.image).read_bytes() == original_bytes[sample.image] for sample in samples)


def test_quality_selection_is_deterministic_and_balances_towns_and_sessions(tmp_path):
    samples = []
    for town in ("Town01", "Town02"):
        for session in range(3):
            for frame in range(2):
                index = len(samples) + 1
                sample = make_sample(tmp_path / "source", f"{town}-session-{session}", index)
                sample.town = town
                sample.group_id = f"{town}-session-{session}"
                sample.session_id = sample.group_id
                samples.append(sample)

    first, report, scores = select_quality_balanced_samples(samples, 6, seed=19, balance_towns=True)
    second, second_report, _ = select_quality_balanced_samples(samples, 6, seed=19, balance_towns=True)

    assert len(first) == 6
    assert [sample.image for sample in first] == [sample.image for sample in second]
    assert report == second_report
    assert report["town_counts_after"] == {"Town01": 3, "Town02": 3}
    assert len(scores) == 6
    assert report["class_images_after"]["1"] > 0
    assert report["class_images_after"]["2"] > 0
    assert report["class_images_after"]["3"] > 0
    for town in ("Town01", "Town02"):
        selected_sessions = {sample.group_id for sample in first if sample.town == town}
        assert len(selected_sessions) == 3


def test_session_quota_ties_are_seeded_and_not_alphabetically_biased():
    import random

    capacities = {"session-a": 10, "session-b": 10, "session-c": 10, "session-d": 10}
    first = _allocate_quotas(2, capacities, equal=True, rng=random.Random(7))
    repeated = _allocate_quotas(2, capacities, equal=True, rng=random.Random(7))
    other_seed = _allocate_quotas(2, capacities, equal=True, rng=random.Random(19))
    assert first == repeated
    assert sum(first.values()) == 2
    assert sum(1 for value in first.values() if value) == 2
    assert first != other_seed


def test_quality_selection_refuses_to_drop_a_town_or_accept_invalid_target(tmp_path):
    first = make_sample(tmp_path, "a", 1)
    second = make_sample(tmp_path, "b", 2)
    second.town = "Town02"
    with pytest.raises(ValueError, match="smaller than the number of selected towns"):
        select_quality_balanced_samples([first, second], 1)
    with pytest.raises(ValueError, match="positive integer"):
        select_quality_balanced_samples([first, second], 0)


def test_prepare_dataset_quality_reduction_is_exact_and_non_destructive(tmp_path):
    samples = []
    for town in ("Town01", "Town02"):
        for index in range(4):
            sample = make_sample(tmp_path / "source", f"{town}-session-{index}", len(samples) + 1)
            sample.town = town
            sample.group_id = f"{town}-session-{index}"
            sample.session_id = sample.group_id
            samples.append(sample)
    source_bytes = {sample.image: Path(sample.image).read_bytes() for sample in samples}

    result = prepare_dataset(
        samples,
        tmp_path / "prepared" / "reduced",
        {"train": 0.5, "val": 0.25, "test": 0.25},
        seed=23,
        max_samples=4,
        balance_towns=True,
    )
    manifest = json.loads((Path(result["path"]) / "split_manifest.json").read_text(encoding="utf-8"))
    report = json.loads((Path(result["path"]) / "dataset_report.json").read_text(encoding="utf-8"))

    assert sum(result["counts"].values()) == 4
    assert result["selection"]["applied"] is True
    assert result["selection"]["town_counts_after"] == {"Town01": 2, "Town02": 2}
    assert report["samples"] == 4
    assert report["source_samples"] == 8
    assert manifest["selection"]["selected_samples"] == 4
    assert sum(len(rows) for rows in manifest["samples"].values()) == 4
    assert all(Path(path).read_bytes() == content for path, content in source_bytes.items())
    assert len(list((Path(result["path"]) / "images").glob("*/*"))) == 4


def test_metrics_include_foreground_and_handle_ignore_label():
    target = np.array([[0, 1, 2, 255]], dtype=np.uint8)
    pred = np.array([[0, 2, 2, 1]], dtype=np.uint8)
    matrix = confusion_matrix(target, pred, num_classes=3, ignore_label=255)
    assert matrix.tolist() == [[1, 0, 0], [0, 0, 1], [0, 0, 1]]
    metrics = metrics_from_confusion(matrix, {0: "background", 1: "drivable", 2: "lane"})
    assert metrics["pixel_accuracy"] == 2 / 3
    assert metrics["miou"] == 0.5
    assert metrics["foreground_miou"] == 0.25
    assert metrics["per_class"]["1"]["recall"] == 0.0


def test_config_save_load_and_migration(tmp_path):
    config = default_config(tmp_path)
    path = tmp_path / "config.json"
    save_config(path, config)
    loaded = load_config(path)
    assert loaded["config_version"] == 3
    assert loaded["dataset"] == {"data_yaml": "", "prepared_dataset": ""}
    assert loaded["export"]["imgsz"] == 512
    assert loaded["model"]["task"] == "semantic"
    assert loaded["model"]["classes"] == {
        0: "background", 1: "ego_drivable_area",
        2: "lane_lines", 3: "vehicles",
    }


def test_validation_report_exports_all_samples_and_issues(tmp_path):
    valid = make_sample(tmp_path, "good", 1)
    missing = make_sample(tmp_path, "bad", 2)
    missing.mask = None
    report = validate_samples([valid, missing])
    destination = export_validation_report(report, tmp_path / "report.json")
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["checked"] == 2
    assert len(payload["valid_samples"]) == 1
    assert any(issue["code"] == "missing_mask" for issue in payload["issues"])


def test_model_registry_deduplicates_by_hash_and_removes_entry_only(tmp_path):
    model = tmp_path / "semantic.pt"
    model.write_bytes(b"fake weights")
    workspace = tmp_path / "workspace"
    first = register_model(workspace, model, {"run_id": "exp-1"})
    second = register_model(workspace, model, {"run_id": "exp-2"})
    assert first["sha256"] == second["sha256"]
    assert len(list_models(workspace)) == 1
    assert list_models(workspace)[0]["details"]["run_id"] == "exp-2"
    assert remove_model(workspace, model)
    assert list_models(workspace) == []
    assert model.is_file()



def test_config_v1_migration_preserves_values_and_adds_new_fields(tmp_path):
    legacy = {"config_version": 1, "project_root": str(tmp_path), "training": {"epochs": 17},
              "split": {"train": 0.7, "val": 0.2, "test": 0.1, "seed": 9}}
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    loaded = load_config(path)
    assert loaded["config_version"] == 3
    assert loaded["model"]["classes"][3] == "vehicles"
    assert loaded["training"]["epochs"] == 17
    assert loaded["training"]["batch"] == 2
    assert loaded["split"]["train"] == 0.7
    assert loaded["split"]["holdout_towns"] == []
    assert loaded["dataset"]["data_yaml"] == ""


def test_prepare_manifest_uses_collision_safe_exact_paths(tmp_path):
    sample = make_sample(tmp_path / "source", "same-session", 10)
    # A duplicate annotation row can occur in merged collection indexes; preparation
    # must not overwrite the first copy or create a non-deterministic filename.
    result = prepare_dataset([sample, sample], tmp_path / "prepared" / "duplicates", {"train": 1, "val": 0, "test": 0}, seed=1)
    manifest = json.loads((Path(result["path"]) / "split_manifest.json").read_text(encoding="utf-8"))
    rows = manifest["samples"]["train"]
    assert len(rows) == 2
    assert rows[0]["prepared_image"] != rows[1]["prepared_image"]
    assert rows[0]["prepared_mask"] != rows[1]["prepared_mask"]
    assert all((Path(result["path"]) / row["prepared_image"]).is_file() for row in rows)
    assert all((Path(result["path"]) / row["prepared_mask"]).is_file() for row in rows)
    assert _dataset_manifest_file(Path(result["path"]), rows[1], "prepared_image", "train").name == Path(rows[1]["prepared_image"]).name


def test_dataset_manifest_rejects_paths_outside_dataset(tmp_path):
    with pytest.raises(ValueError, match="unsafe"):
        _dataset_manifest_file(tmp_path, {"prepared_mask": "../outside.png"}, "prepared_mask", "train")


def test_semantic_decoder_supports_discrete_batch_map_and_logits_layouts():
    class_map = np.arange(42, dtype=np.uint8).reshape(6, 7) % 4
    assert np.array_equal(_semantic_array_to_mask(class_map[None, ...]), class_map)
    nchw = np.stack([class_map == value for value in range(4)]).astype(np.float32)
    assert np.array_equal(_semantic_array_to_mask(nchw), class_map)
    nhwc = np.moveaxis(nchw, 0, -1)
    assert np.array_equal(_semantic_array_to_mask(nhwc), class_map)
    resized = _semantic_array_to_mask(class_map[None, ...], (12, 14))
    assert resized.shape == (12, 14)
    assert _semantic_array_to_mask(np.array([[[3]]], dtype=np.uint8)).tolist() == [[3]]
    assert set(np.unique(resized)).issubset({0, 1, 2, 3})


def test_metrics_reject_out_of_range_predictions_instead_of_dropping_pixels():
    with pytest.raises(ValueError, match="Prediction contains"):
        confusion_matrix(np.array([[0, 1, 2]]), np.array([[0, 1, 255]]), num_classes=3)


def test_evaluation_processes_all_rows_but_limits_saved_previews(tmp_path, monkeypatch):
    import roadforge.core.runtime as runtime

    samples = [make_sample(tmp_path / "source", f"session-{i}", i) for i in range(4)]
    prepared = prepare_dataset(samples, tmp_path / "prepared" / "eval", {"train": 0, "val": 0, "test": 1}, seed=3)

    class FakeEngine:
        def __init__(self, *args, **kwargs):
            pass
        def predict(self, rgb):
            return np.zeros(rgb.shape[:2], dtype=np.uint8)

    monkeypatch.setattr(runtime, "SemanticEngine", FakeEngine)
    report = evaluate_dataset("fake.pt", prepared["path"], split="test", max_examples=2)
    assert report["samples"] == 4
    assert report["metrics"]["num_pixels"] == 4 * 12 * 20
    assert report["examples_generated"] == 2
    assert report["example_limit"] == 2
    assert Path(report["report_json"]).is_file()


def test_deployment_inference_script_is_valid_python():
    compile(_DEPLOYMENT_INFERENCE_SCRIPT, "deployment_inference.py", "exec")


def test_folder_inference_skips_its_output_and_avoids_same_stem_collisions(tmp_path, monkeypatch):
    import roadforge.core.runtime as runtime

    source = tmp_path / "input"
    source.mkdir()
    rgb = np.full((8, 10, 3), 80, dtype=np.uint8)
    Image.fromarray(rgb, "RGB").save(source / "frame.png")
    Image.fromarray(rgb, "RGB").save(source / "frame.jpg")
    output = source / "predictions"

    class FakeEngine:
        def __init__(self, *args, **kwargs):
            pass
        def predict(self, image):
            return np.zeros(image.shape[:2], dtype=np.uint8)

    monkeypatch.setattr(runtime, "SemanticEngine", FakeEngine)
    first = run_folder_inference("fake.pt", str(source), str(output))
    assert first["processed"] == 2
    generated = list(Path(first["output_dir"]).rglob("*_mask.png"))
    assert len(generated) == 2

    # A second run with the same output target must neither read its own prior
    # outputs nor overwrite the previous run.
    second = run_folder_inference("fake.pt", str(source), str(output))
    assert second["processed"] == 2
    assert Path(first["output_dir"]).is_dir()
    assert Path(second["output_dir"]).is_dir()
    assert second["output_dir"] != first["output_dir"]


def test_undefined_class_metrics_are_json_safe_nulls():
    metrics = metrics_from_confusion(np.array([[10, 0, 0], [0, 0, 0], [0, 0, 0]]))
    assert metrics["miou"] == 1.0
    assert metrics["foreground_miou"] is None
    assert metrics["per_class"]["1"]["iou"] is None
    # Strict JSON encoding rejects NaN and Infinity; metrics should serialize cleanly.
    json.dumps(metrics, allow_nan=False)


def test_ignore_pixels_are_counted_and_class_id_is_preserved_in_manifest(tmp_path):
    values = np.zeros((12, 20), dtype=np.uint8)
    values[0, :4] = 255
    values[4:10, 2:18] = 1
    sample = make_sample(tmp_path / "source", "ignore-session", 11, values)
    result = prepare_dataset([sample], tmp_path / "prepared" / "ignore", {"train": 1, "val": 0, "test": 0}, ignore_label=255)
    manifest = json.loads((Path(result["path"]) / "split_manifest.json").read_text(encoding="utf-8"))
    assert manifest["ignore_label"] == 255
    assert result["report"]["class_distribution"]["train"]["ignored_pixel_count"] == 4
    assert result["report"]["ignore_label"] == 255
    with pytest.raises(ValueError, match="outside the semantic class IDs"):
        validate_samples([sample], ignore_label=1)


def test_legacy_three_class_annotation_is_discovered_but_rejected_for_training(tmp_path):
    collection = tmp_path / "collection_legacy"
    image = collection / "raw" / "Town01" / "images" / "old.png"
    mask = collection / "labels_yolopv2" / "Town01" / "combined_3class" / "old.png"
    image.parent.mkdir(parents=True)
    mask.parent.mkdir(parents=True)
    Image.fromarray(np.zeros((4, 6, 3), dtype=np.uint8), "RGB").save(image)
    Image.fromarray(np.ones((4, 6), dtype=np.uint8), "L").save(mask)
    (collection / "collection_manifest.json").write_text(
        json.dumps({"session_id": "legacy-session", "config": {}}), encoding="utf-8"
    )
    (collection / "annotation_manifest.json").write_text(
        json.dumps({"images": {
            "raw/Town01/images/old.png": {
                "status": "success",
                "original_image": "raw/Town01/images/old.png",
                "combined_mask": "labels_yolopv2/Town01/combined_3class/old.png",
            }
        }}), encoding="utf-8"
    )
    found = discover_collections([tmp_path])
    sample = found[0].samples[0]
    assert sample.mask == str(mask)
    report = validate_samples([sample])
    assert not report.valid
    assert any(issue.code == "legacy_class_map" for issue in report.issues)


def test_config_v2_three_class_upgrade_appends_vehicles(tmp_path):
    legacy = {
        "config_version": 2,
        "project_root": str(tmp_path),
        "model": {"classes": {0: "background", 1: "ego_drivable_area", 2: "lane_lines"}},
    }
    path = tmp_path / "roadforge-v2.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    loaded = load_config(path)
    assert loaded["config_version"] == 3
    assert loaded["model"]["classes"] == {
        0: "background", 1: "ego_drivable_area",
        2: "lane_lines", 3: "vehicles",
    }
