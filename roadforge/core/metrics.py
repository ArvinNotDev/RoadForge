from __future__ import annotations

import numpy as np


def confusion_matrix(
    target: np.ndarray,
    prediction: np.ndarray,
    num_classes: int = 3,
    ignore_label: int | None = None,
) -> np.ndarray:
    if num_classes < 1:
        raise ValueError("num_classes must be positive.")
    target = np.asarray(target).astype(np.int64, copy=False).reshape(-1)
    prediction = np.asarray(prediction).astype(np.int64, copy=False).reshape(-1)
    if target.shape != prediction.shape:
        raise ValueError("Target and prediction must have the same number of pixels.")

    ignored = target == ignore_label if ignore_label is not None else np.zeros(target.shape, dtype=bool)
    invalid_target = ((target < 0) | (target >= num_classes)) & ~ignored
    if np.any(invalid_target):
        invalid = np.unique(target[invalid_target]).tolist()
        raise ValueError(f"Target contains class IDs outside [0, {num_classes - 1}]: {invalid}")
    valid = ~ignored
    invalid_prediction = (prediction[valid] < 0) | (prediction[valid] >= num_classes)
    if np.any(invalid_prediction):
        invalid = np.unique(prediction[valid][invalid_prediction]).tolist()
        raise ValueError(f"Prediction contains class IDs outside [0, {num_classes - 1}]: {invalid}")

    encoded = num_classes * target[valid] + prediction[valid]
    return np.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def _nanmean(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else None


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def metrics_from_confusion(matrix: np.ndarray, names: dict[int, str] | None = None) -> dict:
    cm = np.asarray(matrix, dtype=np.int64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1] or cm.shape[0] == 0:
        raise ValueError("Confusion matrix must be a non-empty square matrix.")
    if np.any(cm < 0):
        raise ValueError("Confusion matrix cannot contain negative counts.")

    tp = np.diag(cm).astype(np.float64)
    actual = cm.sum(axis=1).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    union = actual + predicted - tp
    iou = np.divide(tp, union, out=np.full_like(tp, np.nan), where=union != 0)
    precision = np.divide(tp, predicted, out=np.full_like(tp, np.nan), where=predicted != 0)
    recall = np.divide(tp, actual, out=np.full_like(tp, np.nan), where=actual != 0)
    dice = np.divide(2 * tp, actual + predicted, out=np.full_like(tp, np.nan), where=(actual + predicted) != 0)
    labels = names or {i: str(i) for i in range(len(tp))}

    return {
        "num_pixels": int(cm.sum()),
        "pixel_accuracy": float(tp.sum() / cm.sum()) if cm.sum() else None,
        "mean_pixel_accuracy": _nanmean(recall),
        "miou": _nanmean(iou),
        "foreground_miou": _nanmean(iou[1:]) if len(iou) > 1 else _nanmean(iou),
        "per_class": {
            str(i): {
                "name": labels.get(i, str(i)),
                "iou": _finite_or_none(iou[i]),
                "precision": _finite_or_none(precision[i]),
                "recall": _finite_or_none(recall[i]),
                "dice": _finite_or_none(dice[i]),
                "pixels": int(actual[i]),
                "predicted_pixels": int(predicted[i]),
            }
            for i in range(len(tp))
        },
        "confusion_matrix": cm.tolist(),
    }
