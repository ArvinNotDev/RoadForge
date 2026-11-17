from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

CONFIG_VERSION = 3
DEFAULT_CLASSES = {
    0: "background",
    1: "ego_drivable_area",
    2: "lane_lines",
    3: "vehicles",
}


def default_config(project_root: Path) -> dict[str, Any]:
    root = Path(project_root)
    collector = root / "datacollector"
    dataset_roots = sorted((p for p in collector.glob("datasets*") if p.is_dir()), key=lambda p: p.name) if collector.is_dir() else []
    if not dataset_roots:
        dataset_roots = [root / "datacollector" / "datasets"]
    return {
        "config_version": CONFIG_VERSION,
        "project_root": str(root),
        "dataset_roots": [str(p) for p in dataset_roots],
        "workspace": str(root / "roadforge" / "workspace"),
        "selected_collections": [],
        "dataset": {"data_yaml": "", "prepared_dataset": ""},
        "model": {"family": "yolo26", "size": "l", "checkpoint": "", "task": "semantic", "classes": DEFAULT_CLASSES},
        "training": {
            "epochs": 100, "imgsz": 512, "batch": 2, "optimizer": "auto", "lr0": 0.01,
            "lrf": 0.01, "momentum": 0.937, "weight_decay": 0.0005,
            "warmup_epochs": 3.0, "warmup_momentum": 0.8, "warmup_bias_lr": 0.1,
            "cos_lr": False, "patience": 30, "val_period": 1, "save_period": 10,
            "seed": 42, "deterministic": True, "workers": 0, "cache": False,
            "device": "auto", "amp": True, "freeze": "", "fliplr": 0.5,
            "degrees": 0.0, "translate": 0.1, "scale": 0.3, "hsv_h": 0.01,
            "hsv_s": 0.3, "hsv_v": 0.2, "mosaic": 0.0, "mixup": 0.0,
            "copy_paste": 0.0, "ignore_label": None,
        },
        "split": {"train": 0.8, "val": 0.1, "test": 0.1, "seed": 42,
                  "strategy": "collection_session", "holdout_towns": []},
        "inference": {"conf": 0.25, "overlay_opacity": 0.42, "provider": "auto", "imgsz": 512},
        "export": {"format": "onnx", "imgsz": 512, "opset": 17, "dynamic": False, "simplify": False},
    }


def _deep_merge(defaults: dict[str, Any], supplied: dict[str, Any]) -> dict[str, Any]:
    result = dict(defaults)
    for key, value in supplied.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def migrate_config(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a JSON object.")
    try:
        version = int(data.get("config_version", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("Configuration version must be an integer.") from exc
    if version > CONFIG_VERSION:
        raise ValueError(f"Configuration version {version} is newer than this application supports ({CONFIG_VERSION}).")
    if version < 0:
        raise ValueError("Configuration version cannot be negative.")

    project_root = Path(str(data.get("project_root") or Path.cwd()))
    migrated = _deep_merge(default_config(project_root), data)
    migrated["config_version"] = CONFIG_VERSION
    model = dict(migrated.get("model", {}))
    classes = model.get("classes", DEFAULT_CLASSES)
    if not isinstance(classes, dict):
        raise ValueError("Model classes must be a mapping of integer IDs to names.")
    try:
        normalized_classes = {int(k): str(v) for k, v in classes.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("Model class IDs must be integers.") from exc
    if not normalized_classes or sorted(normalized_classes) != list(range(len(normalized_classes))):
        raise ValueError("Model class IDs must be contiguous and start at 0.")

    # Configs created by the previous RoadForge schema contain the original
    # three semantic classes. Keep their values and append the new vehicle class
    # rather than forcing users to recreate settings/presets.
    if version < 3 and sorted(normalized_classes) == [0, 1, 2]:
        normalized_classes[3] = "vehicles"
    model["classes"] = normalized_classes
    model.setdefault("task", "semantic")
    migrated["model"] = model

    split = dict(migrated.get("split", {}))
    split.setdefault("holdout_towns", [])
    migrated["split"] = split
    migrated.setdefault("dataset", {"data_yaml": "", "prepared_dataset": ""})
    migrated["dataset"] = _deep_merge({"data_yaml": "", "prepared_dataset": ""}, migrated["dataset"])
    return migrated


def load_config(path: Path) -> dict[str, Any]:
    return migrate_config(json.loads(Path(path).read_text(encoding="utf-8-sig")))


def save_config(path: Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = migrate_config(data)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(normalized, stream, indent=2, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def snapshot_config(data: dict[str, Any]) -> dict[str, Any]:
    """Return a detached JSON-compatible run snapshot."""
    return json.loads(json.dumps(migrate_config(data), default=str))
