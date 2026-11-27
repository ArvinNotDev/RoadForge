from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .classification_data import (
    analyze_classification_split,
    annotated_samples,
    load_dataset_config,
)


IMAGE_SIZE = 224
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
DROPOUT_RATE = 0.30
LABEL_SMOOTHING = 0.02
WEIGHT_DECAY = 5e-4
EARLY_STOPPING_PATIENCE = 8
OVERFIT_GAP_THRESHOLD = 0.20



def _load_resnet18_initial_weights(base, checkpoint_path: Path | str) -> dict[str, Any]:
    """Load a local ResNet18 backbone from a weights-only PyTorch checkpoint."""
    import hashlib
    import torch

    source = Path(checkpoint_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"ResNet18 checkpoint does not exist: {source}")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(source, map_location="cpu")
    except Exception as exc:
        raise ValueError(
            "Could not safely read this .pt file as a weights checkpoint. "
            "Use a ResNet18 state_dict or a checkpoint containing state_dict/model_state_dict."
        ) from exc

    wrappers = ("state_dict", "model_state_dict", "model", "backbone", "weights", "net", "network", "ema")
    for _ in range(6):
        if not isinstance(payload, Mapping):
            break
        nested = next(
            (payload.get(key) for key in wrappers if isinstance(payload.get(key), Mapping)),
            None,
        )
        if nested is None:
            break
        payload = nested

    if not isinstance(payload, Mapping):
        raise ValueError("The selected .pt file does not contain a ResNet18 state dictionary.")

    expected = base.state_dict()
    prefixes = (
        "module.", "_orig_mod.", "model.", "network.", "net.",
        "backbone.", "encoder.", "resnet.", "resnet18.", "feature_extractor.",
    )
    compatible: dict[str, Any] = {}
    shape_mismatches: list[str] = []
    for raw_key, value in payload.items():
        if not torch.is_tensor(value):
            continue
        key = str(raw_key)
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
                    break
        if key.startswith("fc.") or key.startswith("classifier.") or key.startswith("heads."):
            continue
        if key not in expected:
            continue
        if tuple(value.shape) != tuple(expected[key].shape):
            shape_mismatches.append(
                f"{key}: checkpoint {tuple(value.shape)} != ResNet18 {tuple(expected[key].shape)}"
            )
            continue
        compatible[key] = value

    backbone_keys = [key for key in expected if not key.startswith("fc.")]
    required_keys = (
        "conv1.weight",
        "layer1.0.conv1.weight",
        "layer2.0.conv1.weight",
        "layer3.0.conv1.weight",
        "layer4.0.conv1.weight",
    )
    missing_required = [key for key in required_keys if key not in compatible]
    minimum_keys = (len(backbone_keys) * 4 + 4) // 5
    if missing_required or len(compatible) < minimum_keys:
        details = (
            f"Loaded {len(compatible)} of {len(backbone_keys)} ResNet18 backbone tensors; "
            f"at least {minimum_keys} are required."
        )
        if missing_required:
            details += " Missing core layers: " + ", ".join(missing_required) + "."
        if shape_mismatches:
            details += " Shape mismatches: " + "; ".join(shape_mismatches[:5]) + "."
        raise ValueError(
            "The selected checkpoint is not a compatible ResNet18 model. " + details
        )

    base.load_state_dict(compatible, strict=False)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "source": "local_resnet18_pt",
        "checkpoint_name": source.name,
        "checkpoint_path": str(source),
        "checkpoint_sha256": digest.hexdigest(),
        "loaded_backbone_tensors": len(compatible),
        "available_backbone_tensors": len(backbone_keys),
    }


def build_multitask_classifier(
    backbone: str,
    tasks: list[dict[str, Any]],
    pretrained: bool = True,
    initial_weights_path: Path | str | None = None,
):
    import torch.nn as nn
    from torchvision import models

    selected = str(backbone).strip().lower()
    initialization_info: dict[str, Any]
    if selected == "resnet18":
        if initial_weights_path and pretrained:
            raise ValueError("Choose either local ResNet18 .pt weights or ImageNet initialization, not both.")
        weights = models.ResNet18_Weights.DEFAULT if pretrained and not initial_weights_path else None
        base = models.resnet18(weights=weights)
        if initial_weights_path:
            initialization_info = _load_resnet18_initial_weights(base, initial_weights_path)
        else:
            initialization_info = {"source": "imagenet" if pretrained else "random"}
        feature_dim = int(base.fc.in_features)
        base.fc = nn.Identity()
    elif selected == "efficientnet_b0":
        if initial_weights_path:
            raise ValueError("Local .pt fine-tuning is currently supported only for ResNet18.")
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base = models.efficientnet_b0(weights=weights)
        initialization_info = {"source": "imagenet" if pretrained else "random"}
        feature_dim = int(base.classifier[-1].in_features)
        base.classifier = nn.Identity()
    else:
        raise ValueError("Backbone must be either resnet18 or efficientnet_b0.")

    class MultiTaskClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone_name = selected
            self.feature_extractor = base
            # Dropout regularizes the shared representation before task-specific heads.
            self.dropout = nn.Dropout(p=DROPOUT_RATE)
            self.heads = nn.ModuleList(
                [nn.Linear(feature_dim, len(task["classes"])) for task in tasks]
            )

        def forward(self, images):
            features = self.dropout(self.feature_extractor(images))
            return [head(features) for head in self.heads]

    model = MultiTaskClassifier()
    model.initialization_info = initialization_info
    return model


class _ClassificationImageDataset:
    def __new__(cls, rows, tasks, transform):
        import torch
        from PIL import Image
        from torch.utils.data import Dataset

        class DatasetImpl(Dataset):
            def __init__(self):
                self.rows = rows
                self.tasks = tasks
                self.transform = transform
                self.maps = [
                    {name: index for index, name in enumerate(task["classes"])}
                    for task in tasks
                ]

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, index):
                row = self.rows[index]
                with Image.open(row["_absolute_image"]) as image:
                    image = image.convert("RGB")
                    tensor = self.transform(image)
                labels = tuple(
                    self.maps[task_index][row["labels"][task["key"]]]
                    for task_index, task in enumerate(self.tasks)
                )
                return tensor, tuple(torch.tensor(value, dtype=torch.long) for value in labels)

        return DatasetImpl()


def _make_transforms():
    from torchvision import transforms

    train = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        # Horizontal flips are deliberately excluded: they can change road-sign
        # meaning and reverse left/right manoeuvre labels.
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12, hue=0.025)],
            p=0.7,
        ),
        transforms.RandomApply(
            [transforms.RandomAffine(degrees=2.0, translate=(0.015, 0.015), scale=(0.97, 1.03))],
            p=0.35,
        ),
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
    ])
    evaluation = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
    ])
    return train, evaluation


def _metrics_from_confusions(confusions, tasks):
    import torch

    metrics: dict[str, dict[str, Any]] = {}
    for task, raw_matrix in zip(tasks, confusions):
        matrix = raw_matrix.to(dtype=torch.float64)
        true_support = matrix.sum(dim=1)
        predicted_support = matrix.sum(dim=0)
        true_positive = matrix.diag()
        precision = true_positive / predicted_support.clamp_min(1.0)
        recall = true_positive / true_support.clamp_min(1.0)
        denominator = precision + recall
        f1 = torch.where(denominator > 0, 2.0 * precision * recall / denominator, 0.0)
        active = true_support > 0
        macro_f1 = float(f1[active].mean().item()) if bool(active.any()) else 0.0
        balanced_accuracy = float(recall[active].mean().item()) if bool(active.any()) else 0.0
        total = float(matrix.sum().item())
        accuracy = float(true_positive.sum().item() / total) if total else 0.0
        metrics[task["key"]] = {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "macro_f1": macro_f1,
            "per_class": {
                class_name: {
                    "precision": float(precision[index].item()),
                    "recall": float(recall[index].item()),
                    "f1": float(f1[index].item()),
                    "support": int(true_support[index].item()),
                }
                for index, class_name in enumerate(task["classes"])
            },
            "confusion_matrix": matrix.to(dtype=torch.int64).tolist(),
            "unsupported_classes": [
                class_name
                for index, class_name in enumerate(task["classes"])
                if int(true_support[index].item()) == 0
            ],
        }
    return metrics


def _update_confusions(confusions, outputs, targets):
    import torch

    for index, output in enumerate(outputs):
        classes = int(output.shape[1])
        predictions = output.detach().argmax(dim=1).to(device="cpu", dtype=torch.long)
        expected = targets[index].detach().to(device="cpu", dtype=torch.long)
        encoded = expected * classes + predictions
        confusions[index] += torch.bincount(
            encoded, minlength=classes * classes
        ).reshape(classes, classes)


def _evaluate(model, loader, tasks, device):
    import torch

    model.eval()
    confusions = [
        torch.zeros((len(task["classes"]), len(task["classes"])), dtype=torch.long)
        for task in tasks
    ]
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = [item.to(device, non_blocking=True) for item in targets]
            outputs = model(images)
            losses = [
                torch.nn.functional.cross_entropy(outputs[index], targets[index])
                for index in range(len(tasks))
            ]
            batch_count = int(images.shape[0])
            total_loss += float(torch.stack(losses).mean().item()) * batch_count
            total_samples += batch_count
            _update_confusions(confusions, outputs, targets)
    return (total_loss / max(1, total_samples)), _metrics_from_confusions(confusions, tasks)


def train_multitask_classifier(
    dataset_root: Path | str,
    workspace: Path | str,
    backbone: str,
    epochs: int = 30,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    pretrained: bool = True,
    seed: int = 42,
    progress: Callable[[dict[str, Any]], None] | None = None,
    initial_weights_path: Path | str | None = None,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    dataset_root = Path(dataset_root).expanduser().resolve()
    workspace = Path(workspace).expanduser().resolve()
    config = load_dataset_config(dataset_root)
    tasks = config["tasks"]
    rows = annotated_samples(dataset_root)
    if len(rows) < 3:
        raise ValueError(
            f"Only {len(rows)} fully annotated images are available. Annotate more images; "
            "at least three are required even for a basic split."
        )

    split, split_audit = analyze_classification_split(rows, tasks, seed=seed)
    if split_audit["errors"]:
        details = "\n".join(f"• {item}" for item in split_audit["errors"])
        raise ValueError("The classification dataset is not ready for reliable training:\n" + details)

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    for row in rows:
        row["_absolute_image"] = str((dataset_root / row["image"]).resolve())

    train_transform, eval_transform = _make_transforms()
    train_dataset = _ClassificationImageDataset(split["train"], tasks, train_transform)
    val_dataset = _ClassificationImageDataset(split["val"], tasks, eval_transform)
    test_dataset = _ClassificationImageDataset(split["test"], tasks, eval_transform)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    effective_batch_size = max(1, min(256, int(batch_size)))
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=use_cuda,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda,
    )

    selected_backbone = str(backbone).strip().lower()
    model = build_multitask_classifier(
        selected_backbone,
        tasks,
        pretrained=pretrained,
        initial_weights_path=initial_weights_path,
    ).to(torch.device("cuda" if use_cuda else "cpu"))
    device = next(model.parameters()).device
    initialization_info = dict(getattr(model, "initialization_info", {"source": "unknown"}))

    task_class_weights = []
    low_support_warnings = []
    for task in tasks:
        counts = Counter(row["labels"][task["key"]] for row in split["train"])
        total = len(split["train"])
        weights = []
        for class_name in task["classes"]:
            count = counts.get(class_name, 0)
            # Missing training classes are rejected by the split audit. Inverse
            # frequency weights reduce majority-class dominance without resampling.
            weights.append(total / (len(task["classes"]) * count) if count else 0.0)
            if 0 < count < 3:
                low_support_warnings.append(
                    f"{task['name']} / {class_name} has only {count} training sample(s)."
                )
        task_class_weights.append(torch.tensor(weights, dtype=torch.float32, device=device))

    optimizer = torch.optim.AdamW(
        [
            {"params": model.feature_extractor.parameters(), "lr": float(learning_rate)},
            {"params": model.heads.parameters(), "lr": float(learning_rate) * 5.0},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        threshold=1e-3,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = workspace / "classification_runs" / f"{dataset_root.name}_{selected_backbone}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    effective_epochs = max(1, min(500, int(epochs)))
    best_score = -1.0
    best_epoch = 0
    best_record: dict[str, Any] | None = None
    epochs_without_improvement = 0
    high_gap_epochs = 0
    overfit_suspected = False
    history: list[dict[str, Any]] = []
    warnings = list(split_audit["warnings"]) + low_support_warnings
    warning_text = " ".join(dict.fromkeys(warnings)) if warnings else None

    def checkpoint_payload(epoch: int, score: float, validation_metrics) -> dict[str, Any]:
        return {
            "format_version": 1,
            "backbone": selected_backbone,
            "tasks": tasks,
            "input_size": IMAGE_SIZE,
            "mean": IMAGE_MEAN,
            "std": IMAGE_STD,
            "initialization": initialization_info,
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "selection_metric": "validation_macro_f1",
            "validation_score": score,
            "validation_metrics": validation_metrics,
            "split_strategy": split_audit["strategy"],
        }

    if progress:
        progress({
            "state": "started",
            "message": (
                f"Training {selected_backbone} on {device}; "
                f"{len(split['train'])} train / {len(split['val'])} validation / "
                f"{len(split['test'])} test; split={split_audit['strategy']}; "
                f"initialization={initialization_info.get('source', 'unknown')}."
            ),
            "warning": warning_text,
            "split_audit": split_audit,
        })

    for epoch in range(1, effective_epochs + 1):
        model.train()
        running_loss = 0.0
        batches = 0
        train_confusions = [
            torch.zeros((len(task["classes"]), len(task["classes"])), dtype=torch.long)
            for task in tasks
        ]

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = [item.to(device, non_blocking=True) for item in targets]
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            task_losses = [
                torch.nn.functional.cross_entropy(
                    outputs[index],
                    targets[index],
                    weight=task_class_weights[index],
                    label_smoothing=LABEL_SMOOTHING,
                )
                for index in range(len(tasks))
            ]
            loss = torch.stack(task_losses).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Training produced a non-finite loss. Check labels and image files.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            running_loss += float(loss.detach().item())
            batches += 1
            _update_confusions(train_confusions, outputs, targets)

        train_metrics = _metrics_from_confusions(train_confusions, tasks)
        val_loss, val_metrics = _evaluate(model, val_loader, tasks, device)
        train_macro_f1 = sum(item["macro_f1"] for item in train_metrics.values()) / len(tasks)
        val_macro_f1 = sum(item["macro_f1"] for item in val_metrics.values()) / len(tasks)
        train_accuracy = {
            key: item["accuracy"] for key, item in train_metrics.items()
        }
        val_accuracy = {
            key: item["accuracy"] for key, item in val_metrics.items()
        }

        generalization_gap = train_macro_f1 - val_macro_f1
        if generalization_gap >= OVERFIT_GAP_THRESHOLD:
            high_gap_epochs += 1
        else:
            high_gap_epochs = 0
        if high_gap_epochs >= 3:
            overfit_suspected = True

        scheduler.step(val_macro_f1)
        record = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, batches),
            "validation_loss": val_loss,
            "train_accuracy": train_accuracy,
            "validation_accuracy": val_accuracy,
            "train_macro_f1": train_macro_f1,
            "validation_macro_f1": {
                key: item["macro_f1"] for key, item in val_metrics.items()
            },
            "validation_mean_macro_f1": val_macro_f1,
            "generalization_gap_macro_f1": generalization_gap,
            "train_metrics": train_metrics,
            "validation_metrics": val_metrics,
            "learning_rate_backbone": optimizer.param_groups[0]["lr"],
            "learning_rate_heads": optimizer.param_groups[1]["lr"],
        }
        history.append(record)
        if progress:
            progress({"state": "epoch", **record})

        if val_macro_f1 > best_score + 1e-4:
            best_score = val_macro_f1
            best_epoch = epoch
            best_record = record
            epochs_without_improvement = 0
            torch.save(
                checkpoint_payload(epoch, val_macro_f1, val_metrics),
                run_dir / "best.pt",
            )
        else:
            epochs_without_improvement += 1

        torch.save(
            checkpoint_payload(epoch, val_macro_f1, val_metrics),
            run_dir / "last.pt",
        )
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            break

    if best_record is None:
        raise RuntimeError("Training ended without producing a validation-selected checkpoint.")

    best_checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["state_dict"])
    test_loss, test_metrics = _evaluate(model, test_loader, tasks, device)
    test_accuracy = {key: item["accuracy"] for key, item in test_metrics.items()}
    test_macro_f1 = {key: item["macro_f1"] for key, item in test_metrics.items()}

    if overfit_suspected:
        warnings.append(
            "A persistent train/validation macro-F1 gap above 0.20 was observed for at least "
            "three epochs. The best validation checkpoint was retained; collect more diverse "
            "sessions and review class balance."
        )
    warnings = list(dict.fromkeys(warnings))

    report = {
        "status": "complete",
        "backbone": selected_backbone,
        "device": str(device),
        "pretrained": bool(pretrained),
        "initialization": initialization_info,
        "dataset": str(dataset_root),
        "run_dir": str(run_dir),
        "best_checkpoint": str(run_dir / "best.pt"),
        "last_checkpoint": str(run_dir / "last.pt"),
        "annotated_samples": len(rows),
        "split_counts": split_audit["counts"],
        "split_strategy": split_audit["strategy"],
        "split_audit": split_audit,
        "split_warning": " ".join(split_audit["warnings"]) or None,
        "excluded_near_boundaries": split_audit["excluded_near_boundaries"],
        "class_distribution": split_audit["class_distribution"],
        "best_epoch": best_epoch,
        "selection_metric": "validation_macro_f1",
        "best_validation_macro_f1": best_score,
        "best_validation_mean_accuracy": sum(best_record["validation_accuracy"].values()) / len(tasks),
        "best_validation_metrics": best_record["validation_metrics"],
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
        "test_metrics": test_metrics,
        "overfit_suspected": overfit_suspected,
        "warnings": warnings,
        "regularization": {
            "dropout": DROPOUT_RATE,
            "weight_decay": WEIGHT_DECAY,
            "label_smoothing": LABEL_SMOOTHING,
            "gradient_clip_norm": 2.0,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "selection_metric": "mean_validation_macro_f1",
        },
        "tasks": tasks,
        "history": history,
    }
    (run_dir / "training_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (run_dir / "training_config.json").write_text(
        json.dumps(
            {
                "backbone": selected_backbone,
                "epochs_requested": effective_epochs,
                "batch_size": effective_batch_size,
                "learning_rate_backbone": float(learning_rate),
                "learning_rate_heads": float(learning_rate) * 5.0,
                "pretrained": bool(pretrained),
                "initialization": initialization_info,
                "seed": int(seed),
                "input_size": IMAGE_SIZE,
                "tasks": tasks,
                "split_audit": split_audit,
                "regularization": report["regularization"],
                "augmentation": {
                    "color_jitter_probability": 0.7,
                    "small_affine_probability": 0.35,
                    "horizontal_flip": False,
                },
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    if progress:
        progress({"state": "complete", "report": report})
    return report


def load_classifier_checkpoint(checkpoint_path: Path | str, map_location: str = "cpu"):
    import torch

    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Checkpoint does not contain a valid task/class schema.")
    model = build_multitask_classifier(payload.get("backbone", ""), tasks, pretrained=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload
