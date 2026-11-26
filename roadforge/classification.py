from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
)

from .classification_data import (
    DEFAULT_CAPTURE_ROOT,
    DEFAULT_TASKS,
    annotated_samples,
    ensure_dataset,
    import_avis_captures,
    load_dataset_config,
    load_samples,
    save_sample_labels,
    validate_tasks,
)
from .classification_training import train_multitask_classifier


class ClassificationTrainingWorker(QThread):
    progress = Signal(dict)
    completed = Signal(dict)
    failed = Signal(str)

    def __init__(self, dataset_root, workspace, backbone, epochs, batch_size, learning_rate, pretrained):
        super().__init__()
        self.args = (
            dataset_root, workspace, backbone, epochs, batch_size,
            learning_rate, pretrained,
        )

    def run(self):
        try:
            report = train_multitask_classifier(
                *self.args,
                progress=self.progress.emit,
            )
            self.completed.emit(report)
        except Exception as exc:
            import traceback
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


class ClassificationPage(QWidget):
    """AVIS-specific image capture import, multi-task annotation and training UI."""

    def __init__(self, workspace: Path | str, parent=None):
        super().__init__(parent)
        self.workspace = Path(workspace).expanduser()
        self.capture_root = DEFAULT_CAPTURE_ROOT
        self.dataset_root = self.workspace / "classification_datasets" / "avis_engine"
        self.tasks: list[dict[str, Any]] = []
        self.samples: list[dict[str, Any]] = []
        self.current_index = 0
        self.worker: ClassificationTrainingWorker | None = None
        self._build_ui()
        self._load_default_dataset()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(26, 22, 26, 26)
        layout.setSpacing(12)

        title = QLabel("AVIS Classification Dataset")
        title.setObjectName("PageTitle")
        subtitle = QLabel(
            "Capture frames in AVIS, define one to three classification tasks, annotate each image, "
            "then train ResNet18 or EfficientNet-B0."
        )
        subtitle.setObjectName("Muted")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        io_box = QGroupBox("AVIS captures and dataset")
        io_layout = QGridLayout(io_box)
        self.capture_edit = QLineEdit(str(self.capture_root))
        self.dataset_edit = QLineEdit(str(self.dataset_root))
        io_layout.addWidget(QLabel("AVIS capture folder"), 0, 0)
        io_layout.addWidget(self.capture_edit, 0, 1)
        io_layout.addWidget(self._button("Browse…", self._browse_capture), 0, 2)
        io_layout.addWidget(QLabel("Classification dataset"), 1, 0)
        io_layout.addWidget(self.dataset_edit, 1, 1)
        io_layout.addWidget(self._button("Browse…", self._browse_dataset), 1, 2)
        io_actions = QHBoxLayout()
        io_actions.addWidget(self._button("Create / Open Dataset", self.create_or_open_dataset, primary=True))
        io_actions.addWidget(self._button("Save Task Setup", self.save_task_setup))
        io_actions.addWidget(self._button("Import AVIS Captures", self.import_captures))
        io_actions.addStretch(1)
        io_layout.addLayout(io_actions, 2, 0, 1, 3)
        self.dataset_status = QLabel("Create or open a classification dataset to begin.")
        self.dataset_status.setObjectName("Muted")
        self.dataset_status.setWordWrap(True)
        io_layout.addWidget(self.dataset_status, 3, 0, 1, 3)
        layout.addWidget(io_box)

        task_box = QGroupBox("Task and class configuration")
        task_layout = QVBoxLayout(task_box)
        count_row = QHBoxLayout()
        count_row.addWidget(QLabel("Number of tasks"))
        self.task_count = QSpinBox()
        self.task_count.setRange(1, 3)
        self.task_count.setValue(2)
        self.task_count.valueChanged.connect(self._update_task_visibility)
        count_row.addWidget(self.task_count)
        count_row.addStretch(1)
        count_row.addWidget(QLabel("Each task is a separate classification head."))
        task_layout.addLayout(count_row)
        task_grid = QGridLayout()
        self.task_name_edits: list[QLineEdit] = []
        self.task_class_edits: list[QLineEdit] = []
        defaults = DEFAULT_TASKS
        self.task_boxes: list[QGroupBox] = []
        for index in range(3):
            group = QGroupBox(f"Task {index + 1}")
            form = QFormLayout(group)
            name = QLineEdit(defaults[index]["name"])
            classes = QLineEdit(", ".join(defaults[index]["classes"]))
            classes.setPlaceholderText("class_a, class_b, class_c")
            form.addRow("Task name", name)
            form.addRow("Classes", classes)
            self.task_name_edits.append(name)
            self.task_class_edits.append(classes)
            self.task_boxes.append(group)
            task_grid.addWidget(group, index // 2, index % 2)
        task_layout.addLayout(task_grid)
        hint = QLabel(
            "Example: obstacle = no, yes · return to right lane = no, yes · traffic sign = "
            "no_sign, stop, speed_limit, no_entry, turn_left, give_way. "
            "Use a No Sign class when the image contains no traffic sign."
        )
        hint.setObjectName("Muted")
        hint.setWordWrap(True)
        task_layout.addWidget(hint)
        layout.addWidget(task_box)
        self._update_task_visibility()

        annotation_box = QGroupBox("Sequential image annotation")
        annotation_layout = QVBoxLayout(annotation_box)
        annotation_top = QHBoxLayout()
        self.annotation_counter = QLabel("No images loaded")
        self.annotation_counter.setStyleSheet("font-weight: 700; color: #84edc7;")
        self.sample_details = QLabel("Import AVIS captures to populate the annotation queue.")
        self.sample_details.setObjectName("Muted")
        self.sample_details.setWordWrap(True)
        annotation_top.addWidget(self.annotation_counter)
        annotation_top.addStretch(1)
        annotation_layout.addLayout(annotation_top)
        annotation_layout.addWidget(self.sample_details)
        annotation_body = QHBoxLayout()
        self.image_preview = QLabel("No image selected")
        self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview.setMinimumSize(520, 310)
        self.image_preview.setStyleSheet(
            "background:#0b1118;border:1px solid #29394b;border-radius:8px;color:#9aaabc;padding:10px;"
        )
        annotation_body.addWidget(self.image_preview, 3)
        self.label_panel = QGroupBox("Labels for this image")
        self.label_layout = QVBoxLayout(self.label_panel)
        self.annotation_combos: list[tuple[dict[str, Any], QComboBox]] = []
        annotation_body.addWidget(self.label_panel, 2)
        annotation_layout.addLayout(annotation_body)
        annotation_actions = QHBoxLayout()
        annotation_actions.addWidget(self._button("Previous", self.previous_image))
        annotation_actions.addWidget(self._button("Skip for now", self.next_image))
        annotation_actions.addStretch(1)
        annotation_actions.addWidget(self._button("Save Labels & Next", self.save_and_next, primary=True))
        annotation_layout.addLayout(annotation_actions)
        layout.addWidget(annotation_box)

        training_box = QGroupBox("Multi-task model training")
        training_layout = QGridLayout(training_box)
        self.backbone_combo = QComboBox()
        self.backbone_combo.addItem("ResNet18", "resnet18")
        self.backbone_combo.addItem("EfficientNet-B0", "efficientnet_b0")
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 500)
        self.epochs_spin.setValue(30)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 256)
        self.batch_spin.setValue(16)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.000001, 0.01)
        self.lr_spin.setDecimals(6)
        self.lr_spin.setSingleStep(0.00005)
        self.lr_spin.setValue(0.0001)
        self.pretrained_check = QCheckBox("Initialize from ImageNet pretrained weights")
        self.pretrained_check.setChecked(True)
        training_layout.addWidget(QLabel("Backbone"), 0, 0)
        training_layout.addWidget(self.backbone_combo, 0, 1)
        training_layout.addWidget(QLabel("Epochs"), 0, 2)
        training_layout.addWidget(self.epochs_spin, 0, 3)
        training_layout.addWidget(QLabel("Batch size"), 1, 0)
        training_layout.addWidget(self.batch_spin, 1, 1)
        training_layout.addWidget(QLabel("Backbone learning rate"), 1, 2)
        training_layout.addWidget(self.lr_spin, 1, 3)
        training_layout.addWidget(self.pretrained_check, 2, 0, 1, 3)
        self.train_button = self._button("Train selected backbone", self.start_training, primary=True)
        training_layout.addWidget(self.train_button, 2, 3)
        self.training_progress = QProgressBar()
        self.training_progress.setRange(0, 100)
        self.training_progress.setValue(0)
        training_layout.addWidget(self.training_progress, 3, 0, 1, 4)
        self.training_status = QLabel("Training has not started.")
        self.training_status.setObjectName("Muted")
        self.training_status.setWordWrap(True)
        training_layout.addWidget(self.training_status, 4, 0, 1, 4)
        layout.addWidget(training_box)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(3000)
        self.log.setPlaceholderText("Import, annotation and training events will appear here.")
        log_box = QGroupBox("Classification log")
        log_layout = QVBoxLayout(log_box)
        log_layout.addWidget(self.log)
        layout.addWidget(log_box)

        scroll.setWidget(content)
        outer.addWidget(scroll)

    @staticmethod
    def _button(text: str, callback, primary: bool = False) -> QPushButton:
        item = QPushButton(text)
        if primary:
            item.setObjectName("Primary")
        item.clicked.connect(callback)
        return item

    def _update_task_visibility(self, *_):
        count = self.task_count.value()
        for index, group in enumerate(self.task_boxes):
            group.setVisible(index < count)

    def _read_task_edits(self) -> list[dict[str, Any]]:
        count = self.task_count.value()
        specs = []
        for index in range(count):
            specs.append({
                "name": self.task_name_edits[index].text().strip(),
                "classes": self.task_class_edits[index].text().split(","),
            })
        return validate_tasks(specs)

    def _set_task_edits(self, tasks: list[dict[str, Any]]):
        normalized = validate_tasks(tasks)
        self.task_count.setValue(len(normalized))
        for index in range(3):
            if index < len(normalized):
                self.task_name_edits[index].setText(normalized[index]["name"])
                self.task_class_edits[index].setText(", ".join(normalized[index]["classes"]))
            elif index >= len(DEFAULT_TASKS):
                self.task_name_edits[index].clear()
                self.task_class_edits[index].clear()
        self._update_task_visibility()

    def _browse_capture(self):
        folder = QFileDialog.getExistingDirectory(self, "Select AVIS capture folder", self.capture_edit.text())
        if folder:
            self.capture_edit.setText(folder)

    def _browse_dataset(self):
        folder = QFileDialog.getExistingDirectory(self, "Select classification dataset", self.dataset_edit.text())
        if folder:
            self.dataset_edit.setText(folder)

    def _load_default_dataset(self):
        self.dataset_root = Path(self.dataset_edit.text())
        if (self.dataset_root / "dataset.json").is_file():
            try:
                self._open_dataset(self.dataset_root)
            except Exception as exc:
                self._append_log(f"Could not open the previous dataset: {exc}")

    def _open_dataset(self, root: Path):
        config = load_dataset_config(root)
        self.dataset_root = root.resolve()
        self.tasks = config["tasks"]
        self._set_task_edits(self.tasks)
        self.samples = load_samples(self.dataset_root)
        self.current_index = self._first_pending_index()
        self._rebuild_annotation_controls()
        self._refresh_dataset_status()
        self._show_current_sample()

    def create_or_open_dataset(self):
        root = Path(self.dataset_edit.text()).expanduser()
        if (root / "dataset.json").is_file():
            try:
                self._open_dataset(root)
                self._append_log(f"Opened classification dataset: {self.dataset_root}")
                return
            except Exception as exc:
                QMessageBox.warning(self, "Could not open dataset", str(exc))
                return
        try:
            tasks = self._read_task_edits()
            ensure_dataset(root, tasks)
            self._open_dataset(root)
            self._append_log(f"Created classification dataset: {self.dataset_root}")
        except Exception as exc:
            QMessageBox.warning(self, "Dataset setup failed", str(exc))

    def save_task_setup(self):
        try:
            root = Path(self.dataset_edit.text()).expanduser()
            tasks = self._read_task_edits()
            ensure_dataset(root, tasks, allow_task_update=True)
            self._open_dataset(root)
            self._append_log("Task/class schema saved.")
        except Exception as exc:
            QMessageBox.warning(self, "Could not save task setup", str(exc))

    def import_captures(self):
        try:
            root = Path(self.dataset_edit.text()).expanduser()
            if not (root / "dataset.json").is_file():
                self.create_or_open_dataset()
            if not (root / "dataset.json").is_file():
                return
            result = import_avis_captures(self.capture_edit.text(), root)
            self._open_dataset(root)
            message = (
                f"Discovered {result['discovered']} · imported {result['added']} · "
                f"exact duplicates skipped {result['duplicates_skipped']}."
            )
            self.dataset_status.setText(message)
            self._append_log(message)
        except Exception as exc:
            QMessageBox.warning(self, "Capture import failed", str(exc))

    def _first_pending_index(self) -> int:
        if not self.samples or not self.tasks:
            return 0
        for index, sample in enumerate(self.samples):
            labels = sample.get("labels", {})
            if not all(labels.get(task["key"]) in task["classes"] for task in self.tasks):
                return index
        return max(0, len(self.samples) - 1)

    def _rebuild_annotation_controls(self):
        while self.label_layout.count():
            item = self.label_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.annotation_combos = []
        for task in self.tasks:
            row = QFormLayout()
            combo = QComboBox()
            combo.addItem("— Select class —", "")
            for class_name in task["classes"]:
                combo.addItem(class_name, class_name)
            row.addRow(task["name"], combo)
            self.label_layout.addLayout(row)
            self.annotation_combos.append((task, combo))
        self.label_layout.addStretch(1)

    def _sample_is_annotated(self, sample: dict[str, Any]) -> bool:
        labels = sample.get("labels", {})
        return bool(self.tasks) and all(
            labels.get(task["key"]) in task["classes"] for task in self.tasks
        )

    def _refresh_dataset_status(self):
        total = len(self.samples)
        complete = sum(1 for row in self.samples if self._sample_is_annotated(row))
        pending = total - complete
        self.dataset_status.setText(
            f"Dataset: {self.dataset_root} · {total} images · {complete} fully annotated · {pending} pending"
        )
        self.annotation_counter.setText(
            f"{complete}/{total} images annotated · {pending} remaining"
        )

    def _show_current_sample(self):
        if not self.samples or not self.tasks:
            self.image_preview.setPixmap(QPixmap())
            self.image_preview.setText("No images in the annotation queue")
            self.sample_details.setText("Capture frames in AVIS, then import the capture folder.")
            self.annotation_counter.setText("0 images loaded")
            return
        self.current_index = max(0, min(self.current_index, len(self.samples) - 1))
        sample = self.samples[self.current_index]
        image_path = self.dataset_root / sample["image"]
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self.image_preview.setText(f"Could not decode image:\n{image_path}")
        else:
            self.image_preview.setPixmap(
                pixmap.scaled(
                    1050, 560,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        labels = sample.get("labels", {})
        for task, combo in self.annotation_combos:
            selected = labels.get(task["key"], "")
            index = combo.findData(selected)
            combo.setCurrentIndex(index if index >= 0 else 0)
        metadata = sample.get("capture_metadata", {})
        captured = sample.get("captured_at") or metadata.get("captured_at") or "time unavailable"
        session = sample.get("session_id") or "unknown session"
        self.sample_details.setText(
            f"Image {self.current_index + 1}/{len(self.samples)} · {image_path.name} · "
            f"session: {session} · captured: {captured}"
        )
        self.annotation_counter.setText(
            f"Image {self.current_index + 1}/{len(self.samples)} · "
            f"{sum(1 for row in self.samples if self._sample_is_annotated(row))}/{len(self.samples)} complete"
        )

    def _append_log(self, message: str):
        self.log.appendPlainText(str(message))

    def _reload_samples(self):
        self.samples = load_samples(self.dataset_root)
        self._refresh_dataset_status()
        self._show_current_sample()

    def previous_image(self):
        if not self.samples:
            return
        self.current_index = max(0, self.current_index - 1)
        self._show_current_sample()

    def next_image(self):
        if not self.samples:
            return
        self.current_index = min(len(self.samples) - 1, self.current_index + 1)
        self._show_current_sample()

    def save_and_next(self):
        if not self.samples or not self.tasks:
            return
        labels = {
            task["key"]: str(combo.currentData() or "")
            for task, combo in self.annotation_combos
        }
        if any(not value for value in labels.values()):
            QMessageBox.warning(
                self,
                "Missing class label",
                "Select exactly one class for every configured task before saving this image.",
            )
            return
        sample = self.samples[self.current_index]
        try:
            save_sample_labels(self.dataset_root, sample["sample_id"], labels)
            self._reload_samples()
            self._append_log(
                f"Annotated {Path(sample['image']).name}: "
                + ", ".join(f"{key}={value}" for key, value in labels.items())
            )
            if self.current_index < len(self.samples) - 1:
                self.current_index += 1
            else:
                self.current_index = self._first_pending_index()
            self._show_current_sample()
        except Exception as exc:
            QMessageBox.warning(self, "Could not save annotation", str(exc))

    def start_training(self):
        if self.worker is not None and self.worker.isRunning():
            return
        root = Path(self.dataset_edit.text()).expanduser()
        if not (root / "dataset.json").is_file():
            QMessageBox.warning(self, "Dataset required", "Create or open a classification dataset first.")
            return
        try:
            count = len(annotated_samples(root))
            if count < 3:
                raise ValueError(f"Only {count} images have labels for every task. Annotate at least three.")
        except Exception as exc:
            QMessageBox.warning(self, "Dataset is not ready", str(exc))
            return

        self.dataset_root = root.resolve()
        self.train_button.setEnabled(False)
        self.training_progress.setValue(0)
        self.training_status.setText("Initializing model and loading pretrained weights…")
        self.worker = ClassificationTrainingWorker(
            self.dataset_root,
            self.workspace,
            self.backbone_combo.currentData(),
            self.epochs_spin.value(),
            self.batch_spin.value(),
            self.lr_spin.value(),
            self.pretrained_check.isChecked(),
        )
        self.worker.progress.connect(self._training_progress)
        self.worker.completed.connect(self._training_completed)
        self.worker.failed.connect(self._training_failed)
        self.worker.start()
        self._append_log(f"Started {self.backbone_combo.currentText()} multi-task training.")

    def _training_progress(self, payload: dict[str, Any]):
        state = payload.get("state")
        if state == "started":
            self.training_status.setText(str(payload.get("message", "Training started.")))
            if payload.get("warning"):
                self._append_log("SPLIT WARNING · " + str(payload["warning"]))
        elif state == "epoch":
            epochs = max(1, self.epochs_spin.value())
            epoch = int(payload.get("epoch", 0))
            self.training_progress.setValue(min(100, int(epoch * 100 / epochs)))
            accuracies = payload.get("validation_accuracy", {})
            self.training_status.setText(
                f"Epoch {epoch}/{epochs} · train loss {payload.get('train_loss', 0):.4f} · "
                f"validation accuracy " + ", ".join(
                    f"{key}: {value:.1%}" for key, value in accuracies.items()
                )
            )
        elif state == "complete":
            self.training_progress.setValue(100)

    def _training_completed(self, report: dict[str, Any]):
        self.train_button.setEnabled(True)
        self.training_progress.setValue(100)
        self.training_status.setText(
            f"Training complete · best epoch {report['best_epoch']} · "
            f"mean validation accuracy {report['best_validation_mean_accuracy']:.1%} · "
            f"test: {report['test_accuracy']} · checkpoint: {report['best_checkpoint']}"
        )
        self._append_log(json.dumps(report, indent=2, ensure_ascii=False))
        self._append_log(f"Best checkpoint: {report['best_checkpoint']}")

    def _training_failed(self, details: str):
        self.train_button.setEnabled(True)
        self.training_status.setText("Training failed. See the classification log for details.")
        self._append_log("TRAINING ERROR\n" + details)

    def is_training(self) -> bool:
        return self.worker is not None and self.worker.isRunning()
