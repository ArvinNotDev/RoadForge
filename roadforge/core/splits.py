from __future__ import annotations

import hashlib
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .dataset import Sample


@dataclass
class SplitResult:
    train: list[Sample]
    val: list[Sample]
    test: list[Sample]
    group_assignment: dict[str, str]
    counts: dict[str, int]


def _validate_ratios(ratios: dict[str, float]) -> tuple[list[str], float]:
    names = [name for name in ("train", "val", "test") if float(ratios.get(name, 0)) > 0]
    values = {name: float(ratios.get(name, 0)) for name in ("train", "val", "test")}
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in values.values()):
        raise ValueError("Split percentages must be finite values between 0 and 1.")
    total = sum(values.values())
    if not names or abs(total - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test percentages must total 100%.")
    return names, total


def split_samples(samples: list[Sample], ratios: dict[str, float], seed: int = 42, holdout_towns=()) -> SplitResult:
    names, _ = _validate_ratios(ratios)
    if not samples:
        raise ValueError("No samples were supplied for splitting.")

    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[sample.group_id or sample.session_id or sample.collection].append(sample)

    # Union collection groups containing byte-identical images so copies cannot leak
    # across splits. Every copy remains in the prepared data and manifest.
    parent = {key: key for key in groups}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    by_hash: dict[str, str] = {}
    for sample in samples:
        if not sample.content_hash:
            continue
        key = sample.group_id or sample.session_id or sample.collection
        if sample.content_hash in by_hash:
            left, right = find(key), find(by_hash[sample.content_hash])
            if left != right:
                parent[left] = right
        else:
            by_hash[sample.content_hash] = key

    aliases = {key: find(key) for key in groups}
    merged_groups: dict[str, list[Sample]] = defaultdict(list)
    for key, values in groups.items():
        merged_groups[aliases[key]].extend(values)
    groups = merged_groups

    held_towns = {str(t) for t in holdout_towns}
    if held_towns and "test" not in names:
        raise ValueError("Town holdout requires a non-zero test split.")
    held_groups = {key for key, values in groups.items() if any(sample.town in held_towns for sample in values)}
    if held_towns and not held_groups:
        raise ValueError(f"None of the selected samples match held-out towns: {sorted(held_towns)}")

    assigned: dict[str, str] = {key: "test" for key in held_groups}
    groups = {key: value for key, value in groups.items() if key not in held_groups}
    active_names = [name for name in names if name != "test" or not held_towns]
    if held_towns:
        active_names = [name for name in ("train", "val") if float(ratios.get(name, 0)) > 0]
    if len(groups) < len(active_names):
        raise ValueError(
            f"Cannot make {len(active_names)} non-empty splits without leakage: "
            f"only {len(groups)} independent groups remain after town holdout."
        )

    rng = random.Random(seed)
    group_items = list(groups.items())
    rng.shuffle(group_items)
    ordered_names = sorted(active_names, key=lambda name: float(ratios[name]), reverse=True)
    target_total = sum(len(values) for values in groups.values())
    current = Counter()

    # Seed each non-empty requested split with a distinct whole group.
    for (group_id, group_samples), split_name in zip(group_items, ordered_names):
        assigned[group_id] = split_name
        current[split_name] += len(group_samples)

    ratio_total = sum(float(ratios[name]) for name in active_names)
    targets = {name: target_total * float(ratios[name]) / ratio_total for name in active_names}
    for group_id, group_samples in group_items[len(ordered_names):]:
        count = len(group_samples)

        def cost(candidate: str) -> tuple[float, float, str]:
            after = dict(current)
            after[candidate] = after.get(candidate, 0) + count
            squared = sum(
                (after.get(name, 0) - targets[name]) ** 2 / max(targets[name], 1.0)
                for name in active_names
            )
            return squared, current[candidate] / max(targets[candidate], 1.0), candidate

        selected = min(active_names, key=cost)
        assigned[group_id] = selected
        current[selected] += count

    split_values = {name: [] for name in ("train", "val", "test")}
    for sample in samples:
        original_group = sample.group_id or sample.session_id or sample.collection
        split_values[assigned[aliases[original_group]]].append(sample)

    expanded_assignment = {original: assigned[merged] for original, merged in aliases.items()}
    return SplitResult(
        split_values["train"], split_values["val"], split_values["test"],
        expanded_assignment, dict(Counter(assigned.values())),
    )


def _safe_component(value: str, fallback: str = "item") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    return cleaned[:80] or fallback


def _sample_output_stem(sample: Sample) -> str:
    image_path = Path(sample.image).expanduser().resolve()
    identity = hashlib.sha256(str(image_path).casefold().encode("utf-8", errors="replace")).hexdigest()[:10]
    return "__".join((
        _safe_component(Path(sample.collection).name, "collection"),
        _safe_component(sample.town, "town"),
        _safe_component(image_path.stem, "frame"),
        identity,
    ))


def prepare_dataset(
    samples: list[Sample], output_root: str | Path, ratios: dict[str, float], seed: int = 42,
    replace_existing: bool = False, ignore_label: int | None = None, holdout_towns=(),
) -> dict:
    """Copy valid image/mask pairs into Ultralytics' semantic PNG-mask layout.

    Source files are never moved or modified. The split manifest stores exact prepared
    paths so later evaluation does not have to reconstruct filenames heuristically.
    """
    import json
    import shutil
    import uuid

    import numpy as np
    import yaml
    from PIL import Image

    from .config import DEFAULT_CLASSES
    from .dataset import validate_samples

    report = validate_samples(samples, ignore_label=ignore_label)
    errors = [issue for issue in report.issues if issue.severity == "error"]
    if errors:
        raise ValueError(
            f"Dataset has {len(errors)} validation errors; inspect the report and explicitly "
            "select only valid image-mask pairs before preparation."
        )
    if not report.valid:
        raise ValueError("No valid image-mask pairs are available.")

    split = split_samples(report.valid, ratios, seed, holdout_towns)
    target = Path(output_root).expanduser().resolve()
    if target.exists() and not target.is_dir():
        raise NotADirectoryError(f"Prepared dataset destination is not a directory: {target}")
    if target.exists() and any(target.iterdir()):
        target = target.with_name(f"{target.name}_{uuid.uuid4().hex[:8]}")
    target.mkdir(parents=True, exist_ok=True)

    split_rows: dict[str, list[dict]] = {name: [] for name in ("train", "val", "test")}
    class_distribution = {}
    try:
        for part, subset in (("train", split.train), ("val", split.val), ("test", split.test)):
            image_dir = target / "images" / part
            mask_dir = target / "masks" / part
            image_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)
            pixel_counts: Counter[int] = Counter()
            image_counts: Counter[int] = Counter()
            ignored_pixel_count = 0

            stem_occurrences: Counter[str] = Counter()
            for sample in subset:
                image_source = Path(sample.image)
                mask_source = Path(sample.mask)
                base_stem = _sample_output_stem(sample)
                stem_occurrences[base_stem] += 1
                stem = base_stem if stem_occurrences[base_stem] == 1 else f"{base_stem}__duplicate_{stem_occurrences[base_stem]}"
                prepared_image = image_dir / f"{stem}{image_source.suffix.lower()}"
                prepared_mask = mask_dir / f"{stem}.png"
                if prepared_image.exists() or prepared_mask.exists():
                    raise FileExistsError(f"Prepared sample path collision: {prepared_image}")
                shutil.copy2(image_source, prepared_image)
                shutil.copy2(mask_source, prepared_mask)

                row = {
                    "sample_id": stem,
                    "image": prepared_image.name,
                    "mask": prepared_mask.name,
                    "prepared_image": prepared_image.relative_to(target).as_posix(),
                    "prepared_mask": prepared_mask.relative_to(target).as_posix(),
                    "source_image": str(image_source.resolve()),
                    "source_mask": str(mask_source.resolve()),
                    "collection": sample.collection,
                    "town": sample.town,
                    "session_id": sample.session_id,
                    "group_id": sample.group_id,
                    "route_id": sample.route_id,
                    "weather": sample.weather,
                    "traffic": sample.traffic,
                }
                split_rows[part].append(row)

                with Image.open(mask_source) as image_mask:
                    values, counts = np.unique(np.asarray(image_mask), return_counts=True)
                for class_id, pixel_count in zip(values, counts):
                    class_id = int(class_id)
                    if class_id in DEFAULT_CLASSES:
                        pixel_counts[class_id] += int(pixel_count)
                        image_counts[class_id] += 1
                    elif ignore_label is not None and class_id == ignore_label:
                        ignored_pixel_count += int(pixel_count)

            class_distribution[part] = {
                "pixel_counts": {i: pixel_counts.get(i, 0) for i in range(3)},
                "image_counts": {i: image_counts.get(i, 0) for i in range(3)},
                "ignored_pixel_count": ignored_pixel_count,
            }

        yaml_data = {
            "path": target.as_posix(),
            "train": "images/train",
            "val": "images/val",
            "masks_dir": "masks",
            "names": {int(key): value for key, value in DEFAULT_CLASSES.items()},
        }
        if split.test:
            yaml_data["test"] = "images/test"
        data_yaml = target / "data.yaml"
        data_yaml.write_text(yaml.safe_dump(yaml_data, sort_keys=False, allow_unicode=True), encoding="utf-8")

        counts = {"train": len(split.train), "val": len(split.val), "test": len(split.test)}
        manifest = {
            "format_version": 2,
            "seed": seed,
            "strategy": "collection_session",
            "holdout_towns": sorted(map(str, holdout_towns)),
            "ignore_label": ignore_label,
            "ratios": {key: float(value) for key, value in ratios.items()},
            "counts": counts,
            "class_distribution": class_distribution,
            "groups": split.group_assignment,
            "samples": split_rows,
        }
        manifest_path = target / "split_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        total_samples = max(1, sum(counts.values()))
        actual_ratios = {part: count / total_samples for part, count in counts.items()}
        warnings = []
        if holdout_towns:
            requested_test = float(ratios.get("test", 0))
            actual_test = actual_ratios["test"]
            if abs(actual_test - requested_test) > 0.01:
                warnings.append(
                    f"Held-out towns place complete collection sessions in test; actual test share "
                    f"is {actual_test:.1%} versus requested {requested_test:.1%}."
                )
        for part, values in class_distribution.items():
            absent = [DEFAULT_CLASSES[i] for i, count in values["pixel_counts"].items() if count == 0]
            if absent and counts[part]:
                warnings.append(f"{part} split contains no pixels for: {', '.join(absent)}")

        dataset_report = {
            "format_version": 2,
            "samples": len(report.valid),
            "counts": counts,
            "actual_ratios": actual_ratios,
            "class_distribution": class_distribution,
            "holdout_towns": sorted(map(str, holdout_towns)),
            "ignore_label": ignore_label,
            "group_count": len(split.group_assignment),
            "warnings": warnings,
            "source_validation": {
                "checked": report.checked,
                "issues": len(report.issues),
                "duplicate_images": len(report.duplicate_image_hashes),
            },
        }
        (target / "dataset_report.json").write_text(
            json.dumps(dataset_report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        # Do not leave a half-prepared dataset that could be mistaken for a valid one.
        shutil.rmtree(target, ignore_errors=True)
        raise

    return {
        "path": str(target),
        "data_yaml": str(data_yaml),
        "counts": counts,
        "groups": split.group_assignment,
        "manifest": manifest,
        "report": dataset_report,
    }
