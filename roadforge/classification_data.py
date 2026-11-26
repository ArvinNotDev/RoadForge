from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CAPTURE_ROOT = Path.home() / "Documents" / "AVIS_RoadForge_Captures"
DEFAULT_TASKS = [
    {"name": "Obstacle ahead", "classes": ["no", "yes"]},
    {"name": "Return to right lane is safe", "classes": ["no", "yes"]},
    {
        "name": "Traffic sign",
        "classes": ["no_sign", "stop", "speed_limit", "no_entry", "turn_left", "give_way"],
    },
]


def _task_key(name: str, index: int) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return key or f"task_{index + 1}"


def validate_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= 3:
        raise ValueError("Configure between one and three classification tasks.")

    validated: list[dict[str, Any]] = []
    used_keys: set[str] = set()
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError(f"Task {index + 1} must be an object.")
        name = str(task.get("name", "")).strip()
        if not name:
            raise ValueError(f"Task {index + 1} needs a name.")
        raw_classes = task.get("classes", [])
        if isinstance(raw_classes, str):
            raw_classes = raw_classes.split(",")
        classes = [str(value).strip() for value in raw_classes if str(value).strip()]
        if len(classes) < 2:
            raise ValueError(f"Task '{name}' needs at least two classes.")
        lowered = [value.casefold() for value in classes]
        if len(set(lowered)) != len(lowered):
            raise ValueError(f"Task '{name}' contains duplicate class names.")
        key = _task_key(name, index)
        if key in used_keys:
            raise ValueError(f"Task names must be unique; '{name}' conflicts with another task.")
        used_keys.add(key)
        validated.append({"key": key, "name": name, "classes": classes})
    return validated


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path.name}:{line_number}: {exc}") from exc
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
    temporary.replace(path)


def load_dataset_config(dataset_root: Path | str) -> dict[str, Any]:
    path = Path(dataset_root) / "dataset.json"
    if not path.is_file():
        raise FileNotFoundError(f"No classification dataset configuration found at {path}.")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["tasks"] = validate_tasks(data.get("tasks", []))
    return data


def ensure_dataset(
    dataset_root: Path | str,
    tasks: list[dict[str, Any]],
    *,
    allow_task_update: bool = False,
) -> dict[str, Any]:
    root = Path(dataset_root).expanduser().resolve()
    normalized = validate_tasks(tasks)
    root.mkdir(parents=True, exist_ok=True)
    (root / "images").mkdir(exist_ok=True)
    config_path = root / "dataset.json"
    if config_path.is_file():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        old_tasks = validate_tasks(existing.get("tasks", []))
        if old_tasks != normalized:
            samples = _read_jsonl(root / "samples.jsonl")
            has_labels = any(bool(sample.get("labels")) for sample in samples)
            if has_labels or not allow_task_update:
                raise ValueError(
                    "This dataset already has a different task/class schema. "
                    "Create a new dataset or remove all annotations before changing its tasks."
                )
            existing["tasks"] = normalized
            existing["updated_at"] = datetime.now().astimezone().isoformat()
            config_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            return existing
        return existing

    config = {
        "format_version": 1,
        "name": root.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "image_size": 224,
        "tasks": normalized,
        "capture_source": "AVIS Engine / RoadForge",
    }
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_jsonl(root / "samples.jsonl", [])
    return config


def load_samples(dataset_root: Path | str) -> list[dict[str, Any]]:
    root = Path(dataset_root)
    rows = _read_jsonl(root / "samples.jsonl")
    return [
        row for row in rows
        if isinstance(row.get("sample_id"), str)
        and isinstance(row.get("image"), str)
        and (root / row["image"]).is_file()
    ]


def import_avis_captures(
    capture_root: Path | str,
    dataset_root: Path | str,
) -> dict[str, int]:
    source = Path(capture_root).expanduser().resolve()
    target = Path(dataset_root).expanduser().resolve()
    if not (target / "dataset.json").is_file():
        raise FileNotFoundError("Create or open a classification dataset before importing captures.")
    if not source.is_dir():
        raise FileNotFoundError(f"AVIS capture folder does not exist: {source}")

    manifest = _read_jsonl(source / "manifest.jsonl")
    records: list[dict[str, Any]] = []
    if manifest:
        for row in manifest:
            relative = str(row.get("image") or "").replace("\\", "/")
            candidate = source / relative if relative else source / "images" / str(row.get("filename") or "")
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
                records.append({"path": candidate, "metadata": row})
    else:
        for folder in (source / "images", source):
            if folder.is_dir():
                records.extend(
                    {"path": item, "metadata": {}}
                    for item in sorted(folder.iterdir())
                    if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
                )
                if records:
                    break

    samples = load_samples(target)
    known_hashes = {str(row.get("sha256")) for row in samples if row.get("sha256")}
    added = skipped = 0
    images_dir = target / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        source_path: Path = record["path"]
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if digest in known_hashes:
            skipped += 1
            continue
        metadata = record["metadata"]
        suffix = source_path.suffix.lower()
        filename = f"{digest[:16]}{suffix}"
        destination = images_dir / filename
        if not destination.exists():
            shutil.copy2(source_path, destination)
        session_id = str(metadata.get("session_id") or f"import-{source.name}")
        sample = {
            "sample_id": digest,
            "image": f"images/{filename}",
            "sha256": digest,
            "session_id": session_id,
            "captured_at": metadata.get("captured_at"),
            "source": str(source_path),
            "capture_metadata": metadata,
            "labels": {},
        }
        samples.append(sample)
        known_hashes.add(digest)
        added += 1

    _write_jsonl(target / "samples.jsonl", samples)
    return {"discovered": len(records), "added": added, "duplicates_skipped": skipped}


def save_sample_labels(
    dataset_root: Path | str,
    sample_id: str,
    labels: dict[str, str],
) -> None:
    root = Path(dataset_root)
    config = load_dataset_config(root)
    tasks = config["tasks"]
    sample_map = {task["key"]: task for task in tasks}
    if set(labels) != set(sample_map):
        raise ValueError("A class label is required for every configured task.")
    for key, label in labels.items():
        if label not in sample_map[key]["classes"]:
            raise ValueError(f"'{label}' is not a configured class for task '{sample_map[key]['name']}'.")

    samples = _read_jsonl(root / "samples.jsonl")
    for sample in samples:
        if sample.get("sample_id") == sample_id:
            sample["labels"] = dict(labels)
            sample["annotated_at"] = datetime.now().astimezone().isoformat()
            _write_jsonl(root / "samples.jsonl", samples)
            return
    raise KeyError(f"Sample not found: {sample_id}")


def annotated_samples(dataset_root: Path | str) -> list[dict[str, Any]]:
    config = load_dataset_config(dataset_root)
    tasks = config["tasks"]
    result = []
    for sample in load_samples(dataset_root):
        labels = sample.get("labels")
        if isinstance(labels, dict) and all(
            task["key"] in labels and labels[task["key"]] in task["classes"]
            for task in tasks
        ):
            result.append(sample)
    return result


def split_samples(
    samples: list[dict[str, Any]],
    seed: int = 42,
) -> tuple[dict[str, list[dict[str, Any]]], str | None]:
    if len(samples) < 3:
        raise ValueError("At least three fully annotated images are required to create train/validation/test splits.")
    rng = random.Random(int(seed))
    groups: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        group = str(sample.get("session_id") or "unknown-session")
        groups.setdefault(group, []).append(sample)

    warning = None
    if len(groups) >= 3:
        group_names = list(groups)
        rng.shuffle(group_names)
        test_count = max(1, round(len(group_names) * 0.1))
        val_count = max(1, round(len(group_names) * 0.1))
        while test_count + val_count >= len(group_names):
            if test_count > 1:
                test_count -= 1
            else:
                val_count -= 1
        test_groups = set(group_names[:test_count])
        val_groups = set(group_names[test_count:test_count + val_count])
        train_groups = set(group_names[test_count + val_count:])
        return (
            {
                "train": [sample for key in group_names if key in train_groups for sample in groups[key]],
                "val": [sample for key in group_names if key in val_groups for sample in groups[key]],
                "test": [sample for key in group_names if key in test_groups for sample in groups[key]],
            },
            None,
        )

    warning = (
        f"Only {len(groups)} AVIS capture session(s) are represented. A deterministic image-level split "
        "was used; neighbouring frames may therefore appear in different splits. Collect additional "
        "separate AVIS runs for session-isolated evaluation."
    )
    shuffled = list(samples)
    rng.shuffle(shuffled)
    test_count = max(1, round(len(shuffled) * 0.1))
    val_count = max(1, round(len(shuffled) * 0.1))
    train_count = len(shuffled) - test_count - val_count
    if train_count < 1:
        raise ValueError("Not enough annotated images for non-empty train, validation and test splits.")
    return (
        {
            "train": shuffled[:train_count],
            "val": shuffled[train_count:train_count + val_count],
            "test": shuffled[train_count + val_count:],
        },
        warning,
    )
