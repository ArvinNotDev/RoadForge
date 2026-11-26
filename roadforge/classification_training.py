from __future__ import annotations

import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .classification_data import annotated_samples, load_dataset_config, split_samples


IMAGE_SIZE = 224
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def build_multitask_classifier(backbone: str, tasks: list[dict[str, Any]], pretrained: bool = True):
    import torch.nn as nn
    from torchvision import models

    selected = str(backbone).strip().lower()
    if selected == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base = models.resnet18(weights=weights)
        feature_dim = int(base.fc.in_features)
        base.fc = nn.Identity()
    elif selected == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base = models.efficientnet_b0(weights=weights)
        feature_dim = int(base.classifier[-1].in_features)
        base.classifier = nn.Identity()
    else:
        raise ValueError("Backbone must be either resnet18 or efficientnet_b0.")

    class MultiTaskClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone_name = selected
            self.feature_extractor = base
            self.heads = nn.ModuleList(
                [nn.Linear(feature_dim, len(task["classes"])) for task in tasks]
            )

        def forward(self, images):
            features = self.feature_extractor(images)
            return [head(features) for head in self.heads]

    return MultiTaskClassifier()


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
        # Avoid horizontal flips: they can reverse the meaning of road signs and lane manoeuvres.
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12, hue=0.025)],
            p=0.7,
        ),
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
    ])
    validation = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
    ])
    return train, validation


def _evaluate(model, loader, tasks, device):
    import torch

    model.eval()
    correct = [0 for _ in tasks]
    totals = [0 for _ in tasks]
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = [item.to(device, non_blocking=True) for item in targets]
            outputs = model(images)
            losses = [
                torch.nn.functional.cross_entropy(outputs[i], targets[i])
                for i in range(len(tasks))
            ]
            total_loss += float(torch.stack(losses).mean().item())
            batches += 1
            for index, output in enumerate(outputs):
                correct[index] += int((output.argmax(dim=1) == targets[index]).sum().item())
                totals[index] += int(targets[index].numel())
    accuracies = {
        task["key"]: (correct[index] / totals[index] if totals[index] else 0.0)
        for index, task in enumerate(tasks)
    }
    return (total_loss / max(1, batches)), accuracies


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
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    dataset_root = Path(dataset_root).expanduser().resolve()
    workspace = Path(workspace).expanduser().resolve()
    config = load_dataset_config(dataset_root)
    tasks = config["tasks"]
    rows = annotated_samples(dataset_root)
    if len(rows) < 3:
        raise ValueError(
            f"Only {len(rows)} fully annotated images are available. Annotate at least three images; "
            "more images and multiple capture sessions are strongly recommended."
        )

    split, split_warning = split_samples(rows, seed=seed)
    if not split["train"] or not split["val"] or not split["test"]:
        raise ValueError("The dataset split produced an empty train, validation or test partition.")

    for row in rows:
        row["_absolute_image"] = str((dataset_root / row["image"]).resolve())

    train_transform, eval_transform = _make_transforms()
    train_dataset = _ClassificationImageDataset(split["train"], tasks, train_transform)
    val_dataset = _ClassificationImageDataset(split["val"], tasks, eval_transform)
    test_dataset = _ClassificationImageDataset(split["test"], tasks, eval_transform)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    train_loader = DataLoader(
        train_dataset, batch_size=max(1, int(batch_size)), shuffle=True,
        num_workers=0, pin_memory=use_cuda, generator=generator,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=max(1, int(batch_size)), shuffle=False,
        num_workers=0, pin_memory=use_cuda,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=max(1, int(batch_size)), shuffle=False,
        num_workers=0, pin_memory=use_cuda,
    )

    random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)

    selected_backbone = str(backbone).strip().lower()
    model = build_multitask_classifier(selected_backbone, tasks, pretrained=pretrained).to(device)
    task_class_weights = []
    missing_training_classes: dict[str, list[str]] = {}
    for task in tasks:
        counts = Counter(row["labels"][task["key"]] for row in split["train"])
        total = len(split["train"])
        weights = []
        missing = []
        for class_name in task["classes"]:
            count = counts.get(class_name, 0)
            if count == 0:
                missing.append(class_name)
                weights.append(0.0)
            else:
                weights.append(total / (len(task["classes"]) * count))
        if missing:
            missing_training_classes[task["key"]] = missing
        task_class_weights.append(torch.tensor(weights, dtype=torch.float32, device=device))

    optimizer = torch.optim.AdamW(
        [
            {"params": model.feature_extractor.parameters(), "lr": float(learning_rate)},
            {"params": model.heads.parameters(), "lr": float(learning_rate) * 5.0},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = workspace / "classification_runs" / f"{dataset_root.name}_{selected_backbone}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    effective_epochs = max(1, min(500, int(epochs)))
    patience = 7
    best_score = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    def checkpoint_payload(epoch: int, score: float) -> dict[str, Any]:
        return {
            "format_version": 1,
            "backbone": selected_backbone,
            "tasks": tasks,
            "input_size": IMAGE_SIZE,
            "mean": IMAGE_MEAN,
            "std": IMAGE_STD,
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "validation_score": score,
        }

    warnings = [item for item in (split_warning,) if item]
    if missing_training_classes:
        missing_text = "; ".join(
            f"{key}: {', '.join(values)}" for key, values in missing_training_classes.items()
        )
        warnings.append("These classes have no training examples and cannot be learned yet: " + missing_text)
    warning_text = " ".join(warnings) if warnings else None
    if progress:
        progress({
            "state": "started",
            "message": f"Training {selected_backbone} on {device}; {len(split['train'])} train / "
                       f"{len(split['val'])} validation / {len(split['test'])} test images.",
            "warning": warning_text,
        })

    for epoch in range(1, effective_epochs + 1):
        model.train()
        running_loss = 0.0
        batches = 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = [item.to(device, non_blocking=True) for item in targets]
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            task_losses = [
                torch.nn.functional.cross_entropy(
                    outputs[index], targets[index], weight=task_class_weights[index]
                )
                for index in range(len(tasks))
            ]
            loss = torch.stack(task_losses).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Training produced a non-finite loss. Check labels and image files.")
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().item())
            batches += 1

        val_loss, val_accuracies = _evaluate(model, val_loader, tasks, device)
        val_score = sum(val_accuracies.values()) / len(tasks)
        scheduler.step(val_score)
        record = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, batches),
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracies,
            "validation_mean_accuracy": val_score,
            "learning_rate_backbone": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        if progress:
            progress({"state": "epoch", **record})

        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(checkpoint_payload(epoch, val_score), run_dir / "best.pt")
        else:
            epochs_without_improvement += 1
        torch.save(checkpoint_payload(epoch, val_score), run_dir / "last.pt")
        if epochs_without_improvement >= patience:
            break

    best_checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["state_dict"])
    test_loss, test_accuracies = _evaluate(model, test_loader, tasks, device)

    report = {
        "status": "complete",
        "backbone": selected_backbone,
        "device": str(device),
        "pretrained": bool(pretrained),
        "dataset": str(dataset_root),
        "run_dir": str(run_dir),
        "best_checkpoint": str(run_dir / "best.pt"),
        "last_checkpoint": str(run_dir / "last.pt"),
        "annotated_samples": len(rows),
        "split_counts": {key: len(value) for key, value in split.items()},
        "split_warning": split_warning,
        "missing_training_classes": missing_training_classes,
        "best_epoch": best_epoch,
        "best_validation_mean_accuracy": best_score,
        "test_loss": test_loss,
        "test_accuracy": test_accuracies,
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
                "batch_size": int(batch_size),
                "learning_rate": float(learning_rate),
                "pretrained": bool(pretrained),
                "seed": int(seed),
                "input_size": IMAGE_SIZE,
                "tasks": tasks,
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
