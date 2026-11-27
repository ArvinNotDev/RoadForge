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



def _capture_timestamp(sample: dict[str, Any]) -> float | None:
    metadata = sample.get("capture_metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    value = (
        sample.get("captured_at")
        or metadata.get("captured_at")
        or metadata.get("timestamp")
        or metadata.get("capture_timestamp")
    )
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        stamp = float(value)
        return stamp / 1000.0 if stamp > 100_000_000_000 else stamp
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        return parsed.timestamp()
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _label_tokens(sample: dict[str, Any]) -> set[str]:
    labels = sample.get("labels")
    if not isinstance(labels, dict):
        return set()
    return {
        f"{str(key)}\x1f{str(value)}"
        for key, value in labels.items()
        if value is not None and str(value).strip()
    }


def _temporal_split(
    samples: list[dict[str, Any]],
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    temporal_gap_seconds: float = 2.0,
) -> tuple[dict[str, list[dict[str, Any]]], int]:
    """Chronological holdout with an embargo around same-session boundaries."""
    indexed = list(enumerate(samples))
    timestamps = [_capture_timestamp(row) for row in samples]
    has_complete_timestamps = all(value is not None for value in timestamps)

    if has_complete_timestamps:
        indexed.sort(key=lambda item: (timestamps[item[0]], item[0]))
    else:
        # JSONL/manifest order is the best available temporal proxy when older
        # capture records do not contain timestamps.
        indexed.sort(key=lambda item: item[0])

    ordered = [row for _, row in indexed]
    count = len(ordered)
    train_end = max(1, min(count - 2, int(round(count * train_ratio))))
    val_end = max(train_end + 1, min(count - 1, int(round(count * (train_ratio + val_ratio)))))
    excluded_indices: set[int] = set()

    for boundary in (train_end, val_end):
        left = ordered[boundary - 1]
        right = ordered[boundary]
        left_session = str(left.get("session_id") or "unknown-session")
        right_session = str(right.get("session_id") or "unknown-session")
        if left_session != right_session:
            continue

        if has_complete_timestamps:
            left_time = _capture_timestamp(left)
            right_time = _capture_timestamp(right)
            if left_time is not None and right_time is not None and right_time > left_time:
                midpoint = (left_time + right_time) / 2.0
                radius = max(0.0, float(temporal_gap_seconds)) / 2.0
                for position, row in enumerate(ordered):
                    if str(row.get("session_id") or "unknown-session") != left_session:
                        continue
                    stamp = _capture_timestamp(row)
                    if stamp is not None and abs(stamp - midpoint) <= radius:
                        excluded_indices.add(position)
                if not any(position in excluded_indices for position in (boundary - 1, boundary)):
                    # A gap already wider than the embargo is safe; no row needs
                    # to be discarded at this boundary.
                    continue
            else:
                excluded_indices.update((boundary - 1, boundary))
        else:
            # With legacy captures, timestamps are unavailable. Drop the two
            # samples touching each same-session boundary instead of randomly
            # mixing adjacent frames into different partitions.
            excluded_indices.update((boundary - 1, boundary))

    partitions = {
        "train": [row for position, row in enumerate(ordered[:train_end]) if position not in excluded_indices],
        "val": [
            row for position, row in enumerate(ordered[train_end:val_end], start=train_end)
            if position not in excluded_indices
        ],
        "test": [
            row for position, row in enumerate(ordered[val_end:], start=val_end)
            if position not in excluded_indices
        ],
    }
    if any(not partitions[name] for name in ("train", "val", "test")):
        raise ValueError(
            "A leakage-aware temporal split cannot create non-empty train, validation and test sets "
            "after applying the temporal embargo. Collect more annotated frames across a longer "
            "capture period or add independent AVIS sessions; RoadForge will not silently fall back "
            "to a random frame split."
        )
    return partitions, len(excluded_indices)


def _build_classification_split(
    samples: list[dict[str, Any]],
    *,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], str | None, str, int]:
    if len(samples) < 3:
        raise ValueError(
            "At least three fully annotated images are required to create train/validation/test splits."
        )

    requested = ratios or {"train": 0.8, "val": 0.1, "test": 0.1}
    normalized = {name: float(requested.get(name, 0.0)) for name in ("train", "val", "test")}
    if any(value < 0 or value > 1 for value in normalized.values()):
        raise ValueError("Split ratios must be between 0 and 1.")
    if abs(sum(normalized.values()) - 1.0) > 1e-6 or any(value <= 0 for value in normalized.values()):
        raise ValueError("Train, validation and test ratios must each be non-zero and total 100%.")

    # First group by independent AVIS sessions. Exact byte-identical images
    # from legacy datasets are unioned so they cannot cross split boundaries.
    session_keys = [
        str(row.get("session_id") or row.get("group_id") or row.get("collection") or "unknown-session")
        for row in samples
    ]
    parents = {key: key for key in session_keys}

    def find(key: str) -> str:
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[left_root] = right_root

    digest_owner: dict[str, str] = {}
    for row, session_key in zip(samples, session_keys):
        digest = str(row.get("sha256") or row.get("content_hash") or "").strip().lower()
        if not digest:
            continue
        if digest in digest_owner:
            union(session_key, digest_owner[digest])
        else:
            digest_owner[digest] = session_key

    session_groups: dict[str, list[dict[str, Any]]] = {}
    for row, session_key in zip(samples, session_keys):
        session_groups.setdefault(find(session_key), []).append(row)

    if len(session_groups) >= 3:
        group_tokens = {
            group: set().union(*(_label_tokens(row) for row in values))
            for group, values in session_groups.items()
        }
        token_groups: dict[str, set[str]] = {}
        for group, tokens in group_tokens.items():
            for token in tokens:
                token_groups.setdefault(token, set()).add(group)

        # If a class occurs in only one independent session, that session must
        # remain in training. Otherwise a random session split could create a
        # checkpoint that never learned one of its configured classes.
        mandatory_train = {
            next(iter(groups))
            for groups in token_groups.values()
            if len(groups) == 1
        }
        remaining = [group for group in session_groups if group not in mandatory_train]
        if len(remaining) >= 2:
            rng = random.Random(int(seed))
            rng.shuffle(remaining)
            group_sizes = {key: len(value) for key, value in session_groups.items()}
            all_tokens = sorted(token_groups)
            token_totals = {
                token: sum(token in _label_tokens(row) for row in samples)
                for token in all_tokens
            }
            total_samples = len(samples)
            target_sizes = {
                part: total_samples * normalized[part]
                for part in ("train", "val", "test")
            }
            target_tokens = {
                part: {
                    token: token_totals[token] * normalized[part]
                    for token in all_tokens
                }
                for part in ("train", "val", "test")
            }
            assigned: dict[str, str] = {group: "train" for group in mandatory_train}
            current_sizes = Counter()
            current_tokens = {
                part: Counter()
                for part in ("train", "val", "test")
            }
            for group in mandatory_train:
                current_sizes["train"] += group_sizes[group]
                for token in group_tokens[group]:
                    current_tokens["train"][token] += sum(
                        token in _label_tokens(row) for row in session_groups[group]
                    )

            def assignment_cost(group: str, part: str) -> float:
                size_after = dict(current_sizes)
                size_after[part] = size_after.get(part, 0) + group_sizes[group]
                size_cost = sum(
                    ((size_after.get(name, 0) - target_sizes[name]) / max(target_sizes[name], 1.0)) ** 2
                    for name in ("train", "val", "test")
                )
                token_cost = 0.0
                for token in all_tokens:
                    for name in ("train", "val", "test"):
                        value = current_tokens[name].get(token, 0)
                        if name == part:
                            value += sum(token in _label_tokens(row) for row in session_groups[group])
                        target = target_tokens[name][token]
                        token_cost += ((value - target) / max(target, 1.0)) ** 2
                return size_cost + 0.35 * token_cost / max(1, len(all_tokens))

            # Reserve one independent group for validation and one for test.
            for part in ("val", "test"):
                selected = min(remaining, key=lambda group: assignment_cost(group, part))
                assigned[selected] = part
                current_sizes[part] += group_sizes[selected]
                for token in group_tokens[selected]:
                    current_tokens[part][token] += sum(
                        token in _label_tokens(row) for row in session_groups[selected]
                    )
                remaining.remove(selected)

            # Allocate all remaining complete sessions against both sample-count
            # targets and the per-class distribution targets.
            token_frequency = {
                token: max(1, len(groups))
                for token, groups in token_groups.items()
            }

            def rarity(group: str) -> float:
                return sum(1.0 / token_frequency[token] for token in group_tokens[group])

            remaining.sort(
                key=lambda group: (rarity(group), group_sizes[group]),
                reverse=True,
            )
            for group in remaining:
                part = min(
                    ("train", "val", "test"),
                    key=lambda candidate: assignment_cost(group, candidate),
                )
                assigned[group] = part
                current_sizes[part] += group_sizes[group]
                for token in group_tokens[group]:
                    current_tokens[part][token] += sum(
                        token in _label_tokens(row) for row in session_groups[group]
                    )

            split = {"train": [], "val": [], "test": []}
            for group, values in session_groups.items():
                split[assigned[group]].extend(values)
            if all(split[name] for name in split):
                return (
                    split,
                    None,
                    "session_isolated",
                    0,
                )

    # Fewer than three independent sessions (or class coverage that makes a
    # strict session split impossible): use chronological holdout with a purge
    # gap. This is deliberately not a random image-level fallback.
    temporal, excluded = _temporal_split(samples)
    warning = (
        f"Only {len(session_groups)} independent capture session group(s) are available, or a "
        "session-only split could not preserve training-class coverage. RoadForge used a "
        "chronological split with a temporal embargo; adjacent same-session frames near partition "
        "boundaries were excluded. This reduces frame leakage but does not provide a fully "
        "session-independent validation/test. Capture at least three separate AVIS runs for that."
    )
    return temporal, warning, "temporal_embargo", excluded


def split_samples(
    samples: list[dict[str, Any]],
    seed: int = 42,
) -> tuple[dict[str, list[dict[str, Any]]], str | None]:
    """Create deterministic, leakage-aware train/validation/test partitions."""
    split, warning, _strategy, _excluded = _build_classification_split(samples, seed=seed)
    return split, warning


def analyze_classification_split(
    samples: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    seed: int = 42,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Return the exact training split plus a class/session audit for the GUI and trainer."""
    split, split_warning, strategy, excluded = _build_classification_split(samples, seed=seed)
    counts = {part: len(values) for part, values in split.items()}
    total = max(1, sum(counts.values()))
    class_distribution: dict[str, dict[str, dict[str, int]]] = {}
    errors: list[str] = []
    warnings = [split_warning] if split_warning else []

    for part, values in split.items():
        class_distribution[part] = {}
        for task in tasks:
            class_distribution[part][task["key"]] = {
                class_name: sum(
                    row.get("labels", {}).get(task["key"]) == class_name
                    for row in values
                )
                for class_name in task["classes"]
            }

    if any(count == 0 for count in counts.values()):
        errors.append("Train, validation and test must all contain at least one sample.")

    for task in tasks:
        train_counts = class_distribution["train"][task["key"]]
        missing_train = [name for name, count in train_counts.items() if count == 0]
        if missing_train:
            errors.append(
                f"Task '{task['name']}' has no training examples for: {', '.join(missing_train)}. "
                "Add annotations for these classes or collect more independent sessions."
            )
        low_support = [name for name, count in train_counts.items() if 0 < count < 3]
        if low_support:
            warnings.append(
                f"Task '{task['name']}' has fewer than three training examples for: "
                f"{', '.join(low_support)}. These classes may overfit."
            )
        for part in ("val", "test"):
            missing = [
                name for name, count in class_distribution[part][task["key"]].items()
                if count == 0
            ]
            if missing:
                warnings.append(
                    f"{part} does not contain every class for task '{task['name']}'; "
                    f"missing: {', '.join(missing)}. Per-class evaluation will be incomplete."
                )

    session_distribution = {
        part: sorted({
            str(row.get("session_id") or row.get("group_id") or row.get("collection") or "unknown-session")
            for row in values
        })
        for part, values in split.items()
    }
    audit = {
        "strategy": strategy,
        "seed": int(seed),
        "requested_ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
        "counts": counts,
        "actual_ratios": {part: counts[part] / total for part in counts},
        "excluded_near_boundaries": int(excluded),
        "class_distribution": class_distribution,
        "session_distribution": session_distribution,
        "warnings": list(dict.fromkeys(warnings)),
        "errors": errors,
    }
    return split, audit
