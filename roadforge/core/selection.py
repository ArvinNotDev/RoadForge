from __future__ import annotations

import math
import random
from collections import Counter, defaultdict

import numpy as np
from PIL import Image, UnidentifiedImageError

from .config import DEFAULT_CLASSES
from .dataset import Sample

QUALITY_WEIGHTS = {
    "sharpness": 0.35,
    "contrast": 0.15,
    "exposure": 0.20,
    "mask_coverage": 0.15,
    "rare_class_support": 0.15,
}


def _percentile_ranks(values: list[float]) -> list[float]:
    """Return deterministic mid-ranks in [0, 1], preserving ties."""
    if len(values) <= 1:
        return [0.5] * len(values)
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = ((start + end - 1) / 2) / (len(order) - 1)
        for pos in range(start, end):
            result[order[pos]] = rank
        start = end
    return result


def _image_metrics(path: str) -> tuple[float, float, float]:
    """Compute sharpness, grayscale contrast and exposure without modifying source files."""
    try:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise ValueError(f"Could not read image for quality scoring: {path}: {exc}") from exc
    values = rgb.astype(np.float32, copy=False)
    gray = 0.299 * values[..., 0] + 0.587 * values[..., 1] + 0.114 * values[..., 2]
    contrast = float(gray.std())
    mean = float(gray.mean())
    clipped = float(np.count_nonzero((gray <= 5) | (gray >= 250)) / gray.size)
    padded = np.pad(gray, ((1, 1), (1, 1)), mode="edge")
    laplacian = (
        padded[1:-1, :-2] + padded[1:-1, 2:]
        + padded[:-2, 1:-1] + padded[2:, 1:-1] - 4.0 * gray
    )
    sharpness = float(laplacian.var())
    exposure = max(0.0, min(
        1.0,
        1.0 - 0.65 * abs(mean - 127.5) / 127.5 - 0.35 * min(1.0, clipped * 20.0),
    ))
    return sharpness, contrast, exposure


def _mask_metrics(path: str) -> tuple[float, set[int]]:
    try:
        with Image.open(path) as image:
            mask = np.asarray(image, dtype=np.uint8)
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise ValueError(f"Could not read semantic mask for quality scoring: {path}: {exc}") from exc
    counts = np.bincount(mask.reshape(-1), minlength=max(DEFAULT_CLASSES) + 1)
    foreground = float(counts[1:max(DEFAULT_CLASSES) + 1].sum() / max(mask.size, 1))
    coverage = min(1.0, foreground / 0.01) * min(1.0, max(0.0, (1.0 - foreground) / 0.02))
    present = {class_id for class_id in DEFAULT_CLASSES if counts[class_id] > 0}
    return coverage, present


def _allocate_quotas(total: int, capacities: dict[str, int], equal: bool, rng: random.Random | None = None) -> dict[str, int]:
    """Allocate an exact count without exceeding any town/session capacity."""
    keys = sorted(capacities)
    if total < 0 or total > sum(capacities.values()):
        raise ValueError("Requested sample allocation exceeds the available candidates.")
    quotas = {key: 0 for key in keys}
    if total >= len(keys):
        for key in keys:
            if capacities[key] > 0:
                quotas[key] = 1
        total -= sum(quotas.values())
    while total:
        active = [key for key in keys if quotas[key] < capacities[key]]
        if not active:
            break
        weights = {key: 1.0 if equal else float(capacities[key] - quotas[key]) for key in active}
        denominator = sum(weights.values())
        exact = {key: total * weights[key] / denominator for key in active}
        allocated = 0
        for key in active:
            amount = min(capacities[key] - quotas[key], math.floor(exact[key]))
            quotas[key] += amount
            allocated += amount
        total -= allocated
        if not total:
            break
        tie_order = list(active)
        if rng is not None:
            rng.shuffle(tie_order)
        tie_rank = {key: index for index, key in enumerate(tie_order)}
        remainder_order = sorted(
            (key for key in active if quotas[key] < capacities[key]),
            key=lambda key: (-(exact[key] - math.floor(exact[key])), tie_rank[key]),
        )
        if not remainder_order:
            break
        for key in remainder_order:
            if not total:
                break
            quotas[key] += 1
            total -= 1
    if total:
        raise ValueError("Could not allocate the requested count across towns and sessions.")
    return quotas


def select_quality_balanced_samples(
    samples: list[Sample],
    target_count: int,
    seed: int = 42,
    balance_towns: bool = True,
    progress=None,
) -> tuple[list[Sample], dict, dict[int, float]]:
    """Select a quality-ranked subset with town/session coverage and rare-class support.

    Image metrics are ranked within each town to avoid systematically favoring maps with
    different lighting or visual texture. The resulting score is a screening heuristic,
    not a claim that teacher-generated masks are ground truth.
    """
    if isinstance(target_count, bool) or not isinstance(target_count, int) or target_count < 1:
        raise ValueError("The target dataset size must be a positive integer.")
    if not samples:
        raise ValueError("No validated samples are available for quality-aware selection.")

    town_counts = Counter(str(sample.town or "Unknown") for sample in samples)
    towns = sorted(town_counts)
    if target_count >= len(samples):
        report = {
            "enabled": True, "applied": False,
            "method": "quality_stratified_town_session_v1",
            "reason": "The requested limit is not smaller than the valid dataset.",
            "target_samples": target_count, "available_samples": len(samples),
            "selected_samples": len(samples), "balance_towns": bool(balance_towns),
            "town_counts_before": dict(sorted(town_counts.items())),
            "town_counts_after": dict(sorted(town_counts.items())),
            "class_images_before": {}, "class_images_after": {},
            "quality_score": None, "quality_weights": QUALITY_WEIGHTS,
        }
        return list(samples), report, {}

    if target_count < len(towns):
        raise ValueError(
            f"The target ({target_count:,}) is smaller than the number of selected towns "
            f"({len(towns)}). Increase the target to at least {len(towns):,} so no town is lost."
        )

    raw: dict[int, dict] = {}
    by_town: dict[str, list[tuple[int, Sample]]] = defaultdict(list)
    class_before: Counter[int] = Counter()
    total = len(samples)
    for index, sample in enumerate(samples):
        if not sample.mask:
            raise ValueError(f"Validated sample has no semantic mask: {sample.image}")
        sharp, contrast, exposure = _image_metrics(sample.image)
        coverage, present = _mask_metrics(sample.mask)
        raw[index] = {
            "sharpness": sharp, "contrast": contrast, "exposure": exposure,
            "mask_coverage": coverage, "present_classes": present,
        }
        by_town[str(sample.town or "Unknown")].append((index, sample))
        class_before.update(present)
        if progress is not None and (index + 1 == total or (index + 1) % 100 == 0):
            progress("Scoring image and mask quality", index + 1, total)

    scores: dict[int, float] = {}
    for town_samples in by_town.values():
        rows = [raw[index] for index, _ in town_samples]
        sharp_ranks = _percentile_ranks([row["sharpness"] for row in rows])
        contrast_ranks = _percentile_ranks([row["contrast"] for row in rows])
        frequency = Counter(
            class_id for row in rows for class_id in row["present_classes"] if class_id != 0
        )
        rarity = [
            sum(1.0 / math.sqrt(max(1, frequency[class_id]))
                for class_id in row["present_classes"] if class_id != 0)
            for row in rows
        ]
        max_rarity = max(rarity, default=0.0) or 1.0
        for (index, _), row, sharp_rank, contrast_rank, rare in zip(
            town_samples, rows, sharp_ranks, contrast_ranks, rarity
        ):
            score = (
                QUALITY_WEIGHTS["sharpness"] * sharp_rank
                + QUALITY_WEIGHTS["contrast"] * contrast_rank
                + QUALITY_WEIGHTS["exposure"] * row["exposure"]
                + QUALITY_WEIGHTS["mask_coverage"] * row["mask_coverage"]
                + QUALITY_WEIGHTS["rare_class_support"] * rare / max_rarity
            )
            scores[index] = float(max(0.0, min(1.0, score)))

    rng = random.Random(seed)
    town_quotas = _allocate_quotas(
        target_count, {town: len(rows) for town, rows in by_town.items()},
        bool(balance_towns), rng=rng,
    )
    selected_indices: set[int] = set()
    for town in towns:
        sessions: dict[str, list[tuple[int, Sample]]] = defaultdict(list)
        for index, sample in by_town[town]:
            session = str(sample.group_id or sample.session_id or sample.collection)
            sessions[session].append((index, sample))
        session_quotas = _allocate_quotas(
            town_quotas[town],
            {session: len(rows) for session, rows in sessions.items()},
            bool(balance_towns), rng=rng,
        )
        for session in sorted(sessions):
            candidates = list(sessions[session])
            rng.shuffle(candidates)
            candidates.sort(key=lambda item: scores[item[0]], reverse=True)
            selected_indices.update(index for index, _ in candidates[:session_quotas[session]])

    selected = [sample for index, sample in enumerate(samples) if index in selected_indices]
    if len(selected) != target_count:
        raise RuntimeError(
            f"Quality selection produced {len(selected):,} samples instead of {target_count:,}."
        )
    selected_scores = {
        id(sample): scores[index]
        for index, sample in enumerate(samples) if index in selected_indices
    }
    town_after = Counter(str(sample.town or "Unknown") for sample in selected)
    class_after: Counter[int] = Counter(
        class_id for index in selected_indices for class_id in raw[index]["present_classes"]
    )
    selected_scores_list = [scores[index] for index in selected_indices]
    sorted_scores = sorted(selected_scores_list)
    report = {
        "enabled": True, "applied": True,
        "method": "quality_stratified_town_session_v1",
        "target_samples": target_count, "available_samples": len(samples),
        "selected_samples": len(selected),
        "removed_from_prepared_copy": len(samples) - len(selected),
        "balance_towns": bool(balance_towns),
        "town_counts_before": dict(sorted(town_counts.items())),
        "town_counts_after": dict(sorted(town_after.items())),
        "class_images_before": {
            str(class_id): int(class_before.get(class_id, 0)) for class_id in DEFAULT_CLASSES
        },
        "class_images_after": {
            str(class_id): int(class_after.get(class_id, 0)) for class_id in DEFAULT_CLASSES
        },
        "quality_score": {
            "minimum": min(selected_scores_list),
            "median": sorted_scores[len(sorted_scores) // 2],
            "mean": float(np.mean(selected_scores_list)),
            "maximum": max(selected_scores_list),
        },
        "quality_weights": QUALITY_WEIGHTS,
        "notes": [
            "Sharpness and contrast are ranked within each town.",
            "Rare foreground classes receive a sampling bonus; class IDs are never changed.",
            "The downstream splitter continues to keep collection sessions and exact duplicate images together.",
            "Scores are heuristic screening signals, not human verification of pseudo-label correctness.",
        ],
    }
    return selected, report, selected_scores
