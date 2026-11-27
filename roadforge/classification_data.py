from __future__ import annotations

import math
import hashlib
import json
import random
from collections import Counter
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
            if left_time is not None and right_time is not None and right_time >= left_time:
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
            # With legacy captures, timestamps are unavailable. Purge a small
            # sequence window on both sides of each same-session boundary.
            # The window grows with the dataset, with a minimum of three rows.
            embargo_samples = max(3, int(round(count * 0.02)))
            start = max(0, boundary - embargo_samples)
            stop = min(count, boundary + embargo_samples)
            for position in range(start, stop):
                if str(ordered[position].get("session_id") or "unknown-session") == left_session:
                    excluded_indices.add(position)

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
    # Legacy datasets may contain repeated image hashes even though the current
    # importer skips them. Keep only the first occurrence, in the earliest split,
    # so identical pixels can never leak into validation or test.
    seen_hashes: set[str] = set()
    duplicate_rows = 0
    for part in ("train", "val", "test"):
        unique_rows = []
        for row in partitions[part]:
            digest = str(row.get("sha256") or row.get("content_hash") or "").strip().lower()
            if digest and digest in seen_hashes:
                duplicate_rows += 1
                continue
            if digest:
                seen_hashes.add(digest)
            unique_rows.append(row)
        partitions[part] = unique_rows

    if any(not partitions[name] for name in ("train", "val", "test")):
        raise ValueError(
            "A leakage-aware temporal split cannot create non-empty train, validation and test sets "
            "after applying the temporal embargo and removing duplicate images. Collect more "
            "annotated frames across a longer capture period or add independent AVIS sessions; "
            "RoadForge will not silently fall back to a random frame split."
        )
    return partitions, len(excluded_indices) + duplicate_rows


def _build_classification_split(
    samples: list[dict[str, Any]],
    *,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], str | None, str, int]:
    """Build an in-domain, multi-task stratified image split.

    AVIS classification is commonly trained for one fixed road/camera setup.
    In that workflow, keeping an entire capture session in one partition can
    strand classes that were only annotated during one session. We therefore
    stratify individual images across the requested partitions while keeping
    exact SHA-256 duplicates together.
    """
    if len(samples) < 3:
        raise ValueError(
            "At least three fully annotated images are required to create train/validation/test splits."
        )

    requested = ratios or {"train": 0.8, "val": 0.1, "test": 0.1}
    normalized = {name: float(requested.get(name, 0.0)) for name in ("train", "val", "test")}
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in normalized.values()):
        raise ValueError("Split ratios must be finite values between 0 and 1.")
    if abs(sum(normalized.values()) - 1.0) > 1e-6 or any(value <= 0 for value in normalized.values()):
        raise ValueError("Train, validation and test ratios must each be non-zero and total 100%.")

    # Group byte-identical images before stratification. Distinct video frames
    # are intentionally allowed to land in different partitions for this
    # fixed-domain workflow.
    groups: dict[str, list[dict[str, Any]]] = {}
    sample_group_keys: list[str] = []
    for index, row in enumerate(samples):
        digest = str(row.get("sha256") or row.get("content_hash") or "").strip().lower()
        sample_id = str(row.get("sample_id") or index)
        key = f"sha256:{digest}" if digest else f"sample:{sample_id}:{index}"
        groups.setdefault(key, []).append(row)
        sample_group_keys.append(key)

    if len(groups) < 3:
        raise ValueError(
            "At least three distinct images are required after grouping exact duplicates."
        )

    group_token_counts: dict[str, Counter[str]] = {}
    token_totals: Counter[str] = Counter()
    token_group_support: Counter[str] = Counter()
    for group_key, values in groups.items():
        counts: Counter[str] = Counter()
        for row in values:
            labels = _label_tokens(row)
            counts.update(labels)
            token_totals.update(labels)
        group_token_counts[group_key] = counts
        token_group_support.update(counts.keys())

    tokens = sorted(token_totals)
    total_samples = len(samples)
    split_names = ("train", "val", "test")
    # Use integer quotas so ordinary one-image groups produce the requested
    # 80/10/10 counts exactly. Any rounding remainder is assigned to test.
    target_sizes = {
        part: int(total_samples * normalized[part])
        for part in split_names
    }
    remainder = total_samples - sum(target_sizes.values())
    for part in ("test", "val", "train"):
        if remainder <= 0:
            break
        target_sizes[part] += 1
        remainder -= 1

    target_tokens = {
        part: {
            token: token_totals[token] * normalized[part]
            for token in tokens
        }
        for part in split_names
    }

    best_assignment: dict[str, str] | None = None
    best_score = float("inf")
    # Multiple deterministic greedy passes make the assignment less sensitive
    # to tie-breaking while remaining reproducible for a given seed.
    attempts = min(192, max(64, len(groups) // 2))
    for attempt in range(attempts):
        rng = random.Random(int(seed) + attempt * 7919)
        ordered = list(groups.items())
        rng.shuffle(ordered)

        def rarity(item):
            group_key, _values = item
            return sum(
                1.0 / max(1, token_group_support[token])
                for token in group_token_counts[group_key]
            )

        ordered.sort(
            key=lambda item: (
                -rarity(item),
                -len(item[1]),
                rng.random(),
            )
        )

        assigned: dict[str, str] = {}
        sizes: Counter[str] = Counter()
        class_counts = {part: Counter() for part in split_names}

        for position, (group_key, values) in enumerate(ordered):
            group_size = len(values)
            additions = group_token_counts[group_key]
            remaining_groups = len(ordered) - position
            missing_parts = [part for part in split_names if sizes[part] == 0]
            eligible = [
                part for part in split_names
                if sizes[part] + group_size <= target_sizes[part]
            ]
            candidates = eligible or list(split_names)
            if remaining_groups <= len(missing_parts):
                reserved = [part for part in missing_parts if part in candidates]
                if reserved:
                    candidates = reserved

            def cost(part: str) -> tuple[float, float, str]:
                before_size = sizes[part]
                target_size = max(float(target_sizes[part]), 1.0)
                size_delta = (
                    ((before_size + group_size - target_size) / target_size) ** 2
                    - ((before_size - target_size) / target_size) ** 2
                )
                class_delta = 0.0
                coverage_bonus = 0.0
                for token, amount in additions.items():
                    before = class_counts[part][token]
                    target = max(target_tokens[part][token], 1.0)
                    class_delta += (
                        ((before + amount - target) / target) ** 2
                        - ((before - target) / target) ** 2
                    )
                    if token_group_support[token] >= 3 and before == 0:
                        coverage_bonus += 1.0
                overflow = max(0, before_size + group_size - target_sizes[part])
                score = (
                    size_delta
                    + 0.85 * class_delta / max(1, len(tokens))
                    - 0.75 * coverage_bonus
                    + 100.0 * overflow
                )
                return score, before_size / target_size, part

            selected = min(candidates, key=cost)
            assigned[group_key] = selected
            sizes[selected] += group_size
            class_counts[selected].update(additions)

        size_loss = sum(
            ((sizes[part] - target_sizes[part]) / max(target_sizes[part], 1.0)) ** 2
            for part in split_names
        )
        class_loss = 0.0
        missing_coverage = 0
        for token in tokens:
            for part in split_names:
                target = max(target_tokens[part][token], 1.0)
                actual = class_counts[part][token]
                class_loss += ((actual - target) / target) ** 2
                if token_group_support[token] >= 3 and actual == 0:
                    missing_coverage += 1
        score = size_loss + 0.85 * class_loss / max(1, len(tokens)) + 100.0 * missing_coverage
        if score < best_score:
            best_score = score
            best_assignment = assigned

    if best_assignment is None:
        raise RuntimeError("Could not construct a stratified image split.")

    # Refine the chosen split by swapping equally sized image groups. This
    # preserves exact partition sizes while improving multi-task class balance.
    class_counts = {part: Counter() for part in split_names}
    for group_key, part in best_assignment.items():
        class_counts[part].update(group_token_counts[group_key])
    equal_size_buckets: dict[int, list[str]] = {}
    for group_key, values in groups.items():
        equal_size_buckets.setdefault(len(values), []).append(group_key)
    swap_buckets = [values for values in equal_size_buckets.values() if len(values) >= 2]
    if swap_buckets and tokens:
        rng = random.Random(int(seed) + 104729)
        swap_steps = min(30000, max(4000, len(groups) * 60))
        for _ in range(swap_steps):
            bucket = rng.choice(swap_buckets)
            left_group, right_group = rng.sample(bucket, 2)
            left_part = best_assignment[left_group]
            right_part = best_assignment[right_group]
            if left_part == right_part:
                continue
            left_tokens = group_token_counts[left_group]
            right_tokens = group_token_counts[right_group]
            affected_tokens = set(left_tokens) | set(right_tokens)
            delta = 0.0
            for token in affected_tokens:
                left_before = class_counts[left_part][token]
                right_before = class_counts[right_part][token]
                left_after = left_before - left_tokens[token] + right_tokens[token]
                right_after = right_before - right_tokens[token] + left_tokens[token]
                left_target = max(target_tokens[left_part][token], 1.0)
                right_target = max(target_tokens[right_part][token], 1.0)
                delta += (
                    (left_after - target_tokens[left_part][token]) ** 2 / left_target ** 2
                    + (right_after - target_tokens[right_part][token]) ** 2 / right_target ** 2
                    - (left_before - target_tokens[left_part][token]) ** 2 / left_target ** 2
                    - (right_before - target_tokens[right_part][token]) ** 2 / right_target ** 2
                )
                if token_group_support[token] >= 3:
                    delta += 100.0 * (
                        int(left_after == 0) + int(right_after == 0)
                        - int(left_before == 0) - int(right_before == 0)
                    )
            if delta < -1e-12:
                best_assignment[left_group], best_assignment[right_group] = right_part, left_part
                class_counts[left_part].subtract(left_tokens)
                class_counts[left_part].update(right_tokens)
                class_counts[right_part].subtract(right_tokens)
                class_counts[right_part].update(left_tokens)

    split = {"train": [], "val": [], "test": []}
    for row, group_key in zip(samples, sample_group_keys):
        split[best_assignment[group_key]].append(row)
    if any(not split[part] for part in split_names):
        raise ValueError("The available distinct images cannot populate all three requested splits.")

    warning = (
        "Image-level stratification is enabled for in-domain training. Frames from the same "
        "AVIS capture session/route may appear in train, validation and test; this is intentional "
        "for the fixed deployment environment, but validation/test scores should not be interpreted "
        "as unseen-session generalization. Exact SHA-256 duplicate images remain in one partition."
    )
    return split, warning, "image_stratified", 0

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
    session_parts: dict[str, set[str]] = {}
    for part, values in split.items():
        for row in values:
            session = str(row.get("session_id") or row.get("group_id") or row.get("collection") or "unknown-session")
            session_parts.setdefault(session, set()).add(part)
    shared_sessions = {
        session: sorted(parts)
        for session, parts in sorted(session_parts.items())
        if len(parts) > 1
    }
    actual_ratios = {part: counts[part] / total for part in counts}
    for part in ("train", "val", "test"):
        if abs(actual_ratios[part] - 0.8 if part == "train" else actual_ratios[part] - 0.1) > 0.05:
            requested_ratio = 0.8 if part == "train" else 0.1
            warnings.append(
                f"The {part} split is {actual_ratios[part]:.1%}, notably different from the "
                f"requested {requested_ratio:.0%}; indivisible exact-duplicate groups can cause this."
            )
        if part in ("val", "test") and counts[part] < 5:
            warnings.append(
                f"The {part} split has only {counts[part]} sample(s); evaluation metrics may be unstable."
            )
    audit = {
        "strategy": strategy,
        "seed": int(seed),
        "requested_ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
        "counts": counts,
        "actual_ratios": actual_ratios,
        "excluded_near_boundaries": int(excluded),
        "class_distribution": class_distribution,
        "session_distribution": session_distribution,
        "shared_sessions_across_splits": shared_sessions,
        "warnings": list(dict.fromkeys(warnings)),
        "errors": errors,
    }
    return split, audit
