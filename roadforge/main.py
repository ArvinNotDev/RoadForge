from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtGui import QAction, QDesktopServices, QImage, QPixmap
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QFrame, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QScrollArea, QSpinBox, QSplitter, QStackedWidget,
    QStatusBar, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QInputDialog,
)

from .core.config import default_config, load_config, save_config
from .core.dataset import CollectionSummary, Sample, ValidationReport, discover_collections, export_validation_report
from .core.registry import list_models, register_model, remove_model
from .core.runtime import (
    benchmark_model, compare_models, create_deployment_package, download_official_checkpoint,
    environment_report, evaluate_dataset, export_onnx, run_folder_inference,
    run_image_inference, run_video_inference,
)
from .core.splits import prepare_dataset
from .core.workers import DatasetScanWorker, OperationWorker, TrainingWorker, ValidationWorker


APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
MODEL_SIZES = ("n", "s", "m", "l", "x")


STYLE = """
QMainWindow, QWidget { background: #111720; color: #e6edf5; font-family: 'Segoe UI'; font-size: 10pt; }
QWidget#Sidebar { background: #0b1118; border-right: 1px solid #253242; }
QLabel#Brand { color: #eff6ff; font-size: 17pt; font-weight: 700; padding: 14px 10px 3px; }
QLabel#SubBrand { color: #70d7b0; font-size: 9pt; padding: 0 10px 16px; }
QLabel#PageTitle { color: #f4f7fb; font-size: 20pt; font-weight: 650; }
QLabel#Muted { color: #9aaabc; }
QFrame#Card, QGroupBox { background: #17212d; border: 1px solid #29394b; border-radius: 10px; }
QGroupBox { margin-top: 12px; padding: 12px 10px 10px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #92a7bc; }
QPushButton { background: #243449; color: #edf5ff; border: 1px solid #36506b; border-radius: 7px; padding: 8px 12px; }
QPushButton:hover { background: #2e4963; } QPushButton:disabled { color: #687789; background: #19222d; }
QPushButton#Primary { background: #17a879; border-color: #24c28d; color: #061610; font-weight: 700; }
QPushButton#Danger { background: #8d3541; border-color: #b94b59; }
QPushButton#Nav { text-align: left; padding: 10px; border: 0; background: transparent; color: #b3c0cf; }
QPushButton#Nav:checked { background: #183d3a; color: #84edc7; border-left: 3px solid #45d5a1; font-weight: 650; }
QLineEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QListWidget { background: #0e151d; color: #e6edf5; border: 1px solid #2d4054; border-radius: 6px; padding: 6px; selection-background-color: #226c5b; }
QComboBox QAbstractItemView { background: #111a24; selection-background-color: #226c5b; }
QCheckBox { spacing: 7px; } QCheckBox::indicator { width: 16px; height: 16px; }
QTableWidget { background: #101720; alternate-background-color: #151f2a; gridline-color: #273747; border: 1px solid #29394b; border-radius: 7px; selection-background-color: #205748; }
QHeaderView::section { background: #1b2836; color: #a9bacb; padding: 7px; border: 0; border-bottom: 1px solid #33475b; font-weight: 600; }
QProgressBar { background: #0c131b; border: 1px solid #304257; border-radius: 6px; text-align: center; color: #ecf6ff; min-height: 18px; }
QProgressBar::chunk { background: #18a879; border-radius: 5px; }
QScrollArea { border: 0; } QStatusBar { background: #0b1118; border-top: 1px solid #263443; color: #9aaabc; }
"""


def button(text: str, callback=None, primary=False, danger=False) -> QPushButton:
    item = QPushButton(text)
    if primary:
        item.setObjectName("Primary")
    if danger:
        item.setObjectName("Danger")
    if callback:
        item.clicked.connect(callback)
    return item


def table_widget(columns: list[str]) -> QTableWidget:
    table = QTableWidget(0, len(columns))
    table.setHorizontalHeaderLabels(columns)
    table.setAlternatingRowColors(True)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    table.setSortingEnabled(False)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    table.horizontalHeader().setStretchLastSection(True)
    return table


def display_image(path: str | Path, label: QLabel, max_size: QSize = QSize(850, 500)) -> None:
    pixmap = QPixmap(str(path))
    if not pixmap.isNull():
        label.setPixmap(pixmap.scaled(max_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        label.setText("")


class MetricCard(QFrame):
    def __init__(self, title: str, value: str = "—", parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        layout = QVBoxLayout(self)
        title_label = QLabel(title)
        title_label.setObjectName("Muted")
        self.value_label = QLabel(value)
        self.value_label.setStyleSheet("font-size: 21pt; font-weight: 700; color: #e9f5ff;")
        layout.addWidget(title_label)
        layout.addWidget(self.value_label)


class MainWindow(QMainWindow):
    PAGES = ["Dashboard", "Dataset Manager", "Dataset Quality & Splits", "Model & Training", "Experiments", "Evaluation", "Model Testing", "ONNX Export", "Settings", "Logs & Diagnostics"]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RoadForge")
        self.resize(1500, 960)
        self.setMinimumSize(1120, 720)
        self.setStyleSheet(STYLE)
        self.config = default_config(PROJECT_ROOT)
        self.settings_path = APP_DIR / "workspace" / "settings.json"
        self.config_path: Path | None = None
        if self.settings_path.is_file():
            try:
                settings = load_config(self.settings_path)
                self.config.update(settings)
            except Exception:
                settings = None  # A damaged optional preferences file must not prevent startup.
            if settings:
                default_path = str(settings.get("default_training_config", "")).strip()
                if default_path and Path(default_path).is_file():
                    try:
                        preset = load_config(Path(default_path))
                        # A preset restores experiment settings, while machine-specific
                        # workspace and source roots remain controlled by application settings.
                        for key in ("project_root", "workspace", "dataset_roots", "default_training_config"):
                            if key in settings:
                                preset[key] = settings[key]
                        self.config.update(preset)
                        self.config_path = Path(default_path)
                    except Exception:
                        # Keep valid app preferences even if the optional preset is damaged.
                        self.config_path = None
        self.collections: list[CollectionSummary] = []
        self.validation_report: ValidationReport | None = None
        self.data_yaml = str(self.config.get("dataset", {}).get("data_yaml", ""))
        self.prepared_dataset = str(self.config.get("dataset", {}).get("prepared_dataset", ""))
        self.worker = None
        self.training_worker: TrainingWorker | None = None
        self.webcam_worker = None
        self.active_model = ""
        self.latest_evaluation: dict | None = None
        self.latest_benchmark: dict | None = None
        self._log_lines: list[str] = []
        self._build_ui()
        self._sync_config_controls()
        self._refresh_configs()
        self._refresh_experiments()
        self._update_dashboard_environment()
        QTimer.singleShot(400, self.rescan)

    def _build_ui(self):
        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(root)
        self.sidebar = QWidget()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(240)
        side_layout = QVBoxLayout(self.sidebar)
        side_layout.setContentsMargins(10, 12, 10, 12)
        brand = QLabel("ROADFORGE")
        brand.setObjectName("Brand")
        subtitle = QLabel("ROAD-SCENE DATA & MODEL WORKBENCH")
        subtitle.setObjectName("SubBrand")
        side_layout.addWidget(brand)
        side_layout.addWidget(subtitle)
        self.nav_buttons = []
        self.stack = QStackedWidget()
        for index, title in enumerate(self.PAGES):
            nav = QPushButton(title)
            nav.setObjectName("Nav")
            nav.setCheckable(True)
            nav.clicked.connect(lambda checked=False, i=index: self.stack.setCurrentIndex(i))
            self.nav_buttons.append(nav)
            side_layout.addWidget(nav)
            self.stack.addWidget(self._page(index))
        self.nav_buttons[0].setChecked(True)
        side_layout.addStretch(1)
        self.busy_text = QLabel("Ready")
        self.busy_text.setObjectName("Muted")
        side_layout.addWidget(self.busy_text)
        root_layout.addWidget(self.sidebar)
        root_layout.addWidget(self.stack, 1)
        self.stack.currentChanged.connect(self._nav_changed)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready · existing project data is opened read only")
        exit_action = QAction("Quit", self)
        exit_action.triggered.connect(self.close)
        self.addAction(exit_action)

    def _page(self, index: int) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(26, 22, 26, 22)
        title = QLabel(self.PAGES[index])
        title.setObjectName("PageTitle")
        outer.addWidget(title)
        if index == 0:
            self._build_dashboard(outer)
        elif index == 1:
            self._build_dataset_page(outer)
        elif index == 2:
            self._build_quality_page(outer)
        elif index == 3:
            self._build_training_page(outer)
        elif index == 4:
            self._build_experiments_page(outer)
        elif index == 5:
            self._build_evaluation_page(outer)
        elif index == 6:
            self._build_testing_page(outer)
        elif index == 7:
            self._build_onnx_page(outer)
        elif index == 8:
            self._build_settings_page(outer)
        else:
            self._build_logs_page(outer)
        return page

    def _nav_changed(self, index: int):
        for i, item in enumerate(self.nav_buttons):
            item.setChecked(i == index)

    def _build_dashboard(self, layout):
        cards = QGridLayout()
        self.dashboard_cards = {}
        for index, (key, name) in enumerate((("collections", "Collections"), ("images", "Raw images"), ("annotated", "Annotated pairs"), ("gpu", "Compute device"), ("model", "Selected model"), ("latest", "Latest run"), ("best_miou", "Best validation mIoU"), ("test_miou", "Latest test mIoU"))):
            card = MetricCard(name)
            self.dashboard_cards[key] = card
            cards.addWidget(card, index // 4, index % 4)
        layout.addLayout(cards)
        self.dashboard_info = QLabel("Scanning the selected CARLA data roots…")
        self.dashboard_info.setWordWrap(True)
        self.dashboard_info.setObjectName("Muted")
        layout.addWidget(self.dashboard_info)
        row = QHBoxLayout()
        self.dashboard_history = table_widget(["Experiment", "Created", "Best checkpoint", "Status"])
        self.dashboard_history.setMaximumHeight(280)
        row.addWidget(self.dashboard_history, 2)
        self.dashboard_runtime = QPlainTextEdit()
        self.dashboard_runtime.setReadOnly(True)
        self.dashboard_runtime.setMaximumWidth(430)
        row.addWidget(self.dashboard_runtime, 1)
        layout.addLayout(row)
        layout.addStretch(1)

    def _build_dataset_page(self, layout):
        root_group = QGroupBox("Read-only source roots")
        root_row = QHBoxLayout(root_group)
        self.roots_edit = QLineEdit()
        self.roots_edit.setPlaceholderText("One dataset collection root per line")
        self.roots_edit.setToolTip("Source image and annotation folders are only read. New collections appear after Rescan.")
        root_row.addWidget(self.roots_edit, 1)
        root_row.addWidget(button("Browse root…", self._browse_roots))
        self.rescan_button = button("Rescan", self.rescan, primary=True)
        root_row.addWidget(self.rescan_button)
        layout.addWidget(root_group)
        selectors = QHBoxLayout()
        town_box = QGroupBox("Towns · multi-select")
        town_layout = QVBoxLayout(town_box)
        self.town_filter = QListWidget()
        town_layout.addWidget(self.town_filter)
        town_buttons = QHBoxLayout()
        town_buttons.addWidget(button("All", lambda: self._check_list(self.town_filter, True)))
        town_buttons.addWidget(button("None", lambda: self._check_list(self.town_filter, False)))
        town_layout.addLayout(town_buttons)
        weather_box = QGroupBox("Weather · multi-select")
        weather_layout = QVBoxLayout(weather_box)
        self.weather_filter = QListWidget()
        weather_layout.addWidget(self.weather_filter)
        weather_buttons = QHBoxLayout()
        weather_buttons.addWidget(button("All", lambda: self._check_list(self.weather_filter, True)))
        weather_buttons.addWidget(button("None", lambda: self._check_list(self.weather_filter, False)))
        weather_layout.addLayout(weather_buttons)
        traffic_box = QGroupBox("Traffic · multi-select")
        traffic_layout = QVBoxLayout(traffic_box)
        self.traffic_filter = QListWidget()
        traffic_layout.addWidget(self.traffic_filter)
        traffic_buttons = QHBoxLayout()
        traffic_buttons.addWidget(button("All", lambda: self._check_list(self.traffic_filter, True)))
        traffic_buttons.addWidget(button("None", lambda: self._check_list(self.traffic_filter, False)))
        traffic_layout.addLayout(traffic_buttons)
        selectors.addWidget(town_box, 1)
        selectors.addWidget(weather_box, 1)
        selectors.addWidget(traffic_box, 1)
        layout.addLayout(selectors)
        actions = QHBoxLayout()
        actions.addWidget(button("Select All Collections", lambda: self._set_collection_checks(True)))
        actions.addWidget(button("Select Annotated", lambda: self._set_collection_checks(True, annotated_only=True)))
        actions.addWidget(button("Deselect All", lambda: self._set_collection_checks(False)))
        self.dataset_summary = QLabel("No collections scanned.")
        self.dataset_summary.setObjectName("Muted")
        actions.addWidget(self.dataset_summary, 1)
        layout.addLayout(actions)
        self.collection_table = table_widget(["Use", "Collection", "Town(s)", "Images", "Annotated", "Missing masks", "Weather", "Traffic", "Status", "Dimensions"])
        self.collection_table.setColumnWidth(0, 42)
        layout.addWidget(self.collection_table, 1)

    def _build_quality_page(self, layout):
        toolbar = QHBoxLayout()
        self.ignore_checkbox = QCheckBox("Allow mask value 255 for validation/evaluation")
        self.ignore_checkbox.setToolTip("Enable only if 255 intentionally marks ignored pixels. Evaluation excludes these pixels. Training is safely blocked when they are present because this installed Ultralytics semantic trainer has no verified ignore-index option in this Studio build.")
        toolbar.addWidget(self.ignore_checkbox)
        toolbar.addWidget(button("Validate selected samples", self.validate_selected, primary=True))
        self.export_validation_button = button("Export full report…", self.export_validation)
        self.export_validation_button.setEnabled(False)
        toolbar.addWidget(self.export_validation_button)
        self.quality_status = QLabel("Run validation before preparing splits.")
        self.quality_status.setObjectName("Muted")
        toolbar.addWidget(self.quality_status, 1)
        layout.addLayout(toolbar)
        self.quality_stats = QLabel("Class pixel distribution: —")
        self.quality_stats.setWordWrap(True)
        self.quality_stats.setObjectName("Muted")
        layout.addWidget(self.quality_stats)
        split_box = QGroupBox("Group-aware train / validation / test split")
        split_form = QHBoxLayout(split_box)
        self.train_ratio = QDoubleSpinBox(); self.train_ratio.setRange(0, 100); self.train_ratio.setDecimals(1); self.train_ratio.setValue(80); self.train_ratio.setSuffix(" % train")
        self.val_ratio = QDoubleSpinBox(); self.val_ratio.setRange(0, 100); self.val_ratio.setDecimals(1); self.val_ratio.setValue(10); self.val_ratio.setSuffix(" % validation")
        self.test_ratio = QDoubleSpinBox(); self.test_ratio.setRange(0, 100); self.test_ratio.setDecimals(1); self.test_ratio.setValue(10); self.test_ratio.setSuffix(" % test")
        self.split_seed = QSpinBox(); self.split_seed.setRange(0, 2147483647); self.split_seed.setValue(42); self.split_seed.setPrefix("Seed ")
        self.dataset_name = QLineEdit("carla_semantic")
        for widget in (self.train_ratio, self.val_ratio, self.test_ratio, self.split_seed, self.dataset_name):
            split_form.addWidget(widget)
        holdout_box = QGroupBox("Optional test town holdout")
        holdout_layout = QVBoxLayout(holdout_box)
        self.holdout_towns = QListWidget(); self.holdout_towns.setMaximumHeight(90); self.holdout_towns.setMinimumWidth(140)
        holdout_layout.addWidget(self.holdout_towns)
        holdout_hint = QLabel("Selecting a town assigns its entire collection session to test."); holdout_hint.setObjectName("Muted"); holdout_hint.setWordWrap(True)
        holdout_layout.addWidget(holdout_hint)
        split_form.addWidget(holdout_box, 1)
        self.prepare_button = button("Prepare dataset copy…", self.prepare_selected, primary=True)
        split_form.addWidget(self.prepare_button)
        layout.addWidget(split_box)
        self.prepared_status = QLabel("Prepared dataset output is written only under the Studio workspace.")
        self.prepared_status.setObjectName("Muted")
        layout.addWidget(self.prepared_status)
        self.issue_table = table_widget(["Severity", "Code", "Message", "Image", "Mask"])
        layout.addWidget(self.issue_table, 1)

    def _build_training_page(self, layout):
        head = QHBoxLayout()
        self.model_size = QComboBox(); self.model_size.addItems(MODEL_SIZES); self.model_size.setCurrentText("l")
        self.model_size.currentTextChanged.connect(self._update_model_name)
        self.model_name = QLabel("YOLO26l-sem · semantic")
        self.checkpoint_edit = QLineEdit()
        self.checkpoint_edit.setPlaceholderText("Select a local yolo26*-sem.pt checkpoint; blank starts from YAML weights")
        self.checkpoint_edit.setToolTip("Missing checkpoints are never downloaded automatically. Use the explicit official checkpoint button to fetch one.")
        checkpoint_button = button("Browse…", lambda: self._choose_file(self.checkpoint_edit, "Model checkpoints (*.pt *.yaml *.yml)"))
        self.download_button = button("Get official checkpoint…", self.download_checkpoint)
        head.addWidget(QLabel("YOLO26 semantic size")); head.addWidget(self.model_size)
        head.addWidget(self.model_name)
        head.addWidget(self.checkpoint_edit, 1); head.addWidget(checkpoint_button); head.addWidget(self.download_button)
        layout.addLayout(head)
        class_row = QHBoxLayout()
        self.class_names = []
        for index, default in enumerate(("background", "ego_drivable_area", "lane_lines", "vehicles")):
            edit = QLineEdit(default); edit.setReadOnly(True); edit.setMaximumWidth(230); edit.setToolTip(f"Fixed semantic class ID {index}; meanings match the source combined masks")
            self.class_names.append(edit)
            class_row.addWidget(QLabel(f"{index}:")); class_row.addWidget(edit)
        class_row.addWidget(QLabel("Task: semantic segmentation · PNG class IDs"))
        class_row.addStretch(1)
        layout.addLayout(class_row)
        dataset_row = QHBoxLayout()
        self.data_yaml_edit = QLineEdit()
        self.data_yaml_edit.setPlaceholderText("Prepared four-class data.yaml (created from validated combined_4class masks)")
        dataset_row.addWidget(self.data_yaml_edit, 1)
        dataset_row.addWidget(button("Browse data.yaml…", lambda: self._choose_data_yaml()))
        self.use_prepared_button = button("Use latest prepared dataset", self.use_prepared_dataset)
        dataset_row.addWidget(self.use_prepared_button)
        layout.addLayout(dataset_row)
        settings_scroll = QScrollArea(); settings_scroll.setWidgetResizable(True)
        content = QWidget(); settings_layout = QVBoxLayout(content)
        core_box = QGroupBox("Training configuration")
        form = QGridLayout(core_box)
        self.training_widgets = {}
        specs = [
            ("epochs", "Epochs", "spin", (1, 10000, 100)), ("imgsz", "Image size", "spin", (128, 2048, 512)),
            ("batch", "Batch size (-1 = framework AutoBatch)", "spin", (-1, 128, 2)), ("optimizer", "Optimizer", "combo", ("auto", "SGD", "Adam", "AdamW", "RMSProp")),
            ("lr0", "Initial learning rate", "double", (0, 1, 0.01)), ("lrf", "Final LR factor", "double", (0, 1, 0.01)),
            ("momentum", "Momentum", "double", (0, 1, 0.937)), ("weight_decay", "Weight decay", "double", (0, 1, 0.0005)),
            ("warmup_epochs", "Warmup epochs", "double", (0, 100, 3)), ("patience", "Early stop patience", "spin", (0, 1000, 30)),
            ("save_period", "Checkpoint interval (epochs)", "spin", (-1, 10000, 10)), ("seed", "Random seed", "spin", (0, 2147483647, 42)),
            ("workers", "Data loader workers", "spin", (0, 64, 0)), ("device", "Device", "combo", ("auto", "0", "cpu")),
            ("cache", "Cache mode", "combo", ("False", "ram", "disk")), ("freeze", "Freeze layer indices (comma separated)", "text", ""),
            ("warmup_momentum", "Warmup momentum", "double", (0, 1, 0.8)), ("warmup_bias_lr", "Warmup bias LR", "double", (0, 1, 0.1)),
            ("val_period", "Validation frequency (epochs)", "spin", (1, 1000, 1)),
        ]
        from ultralytics.cfg import DEFAULT_CFG
        supported_specs = [spec for spec in specs if spec[0] in vars(DEFAULT_CFG) or spec[0] in {"cache", "device", "freeze"}]
        for index, (key, title, kind, spec) in enumerate(supported_specs):
            if kind == "spin":
                widget = QSpinBox(); widget.setRange(*spec[:2]); widget.setValue(spec[2])
            elif kind == "double":
                widget = QDoubleSpinBox(); widget.setRange(spec[0], spec[1]); widget.setDecimals(5); widget.setSingleStep(0.001); widget.setValue(spec[2])
            elif kind == "combo":
                widget = QComboBox(); widget.addItems(spec)
            else:
                widget = QLineEdit(str(spec))
            self.training_widgets[key] = widget
            form.addWidget(QLabel(title), index // 4, (index % 4) * 2)
            form.addWidget(widget, index // 4, (index % 4) * 2 + 1)
        settings_layout.addWidget(core_box)
        aug_box = QGroupBox("Conservative synchronized augmentation")
        aug_form = QGridLayout(aug_box)
        self.augmentation_widgets = {}
        aug_specs = [("fliplr", "Horizontal flip", 0.0, 1.0, 0.5), ("degrees", "Rotation ±degrees", 0.0, 180.0, 0.0),
                     ("translate", "Translate fraction", 0.0, 1.0, 0.1), ("scale", "Scale fraction", 0.0, 1.0, 0.3),
                     ("hsv_h", "Hue", 0.0, 1.0, 0.01), ("hsv_s", "Saturation", 0.0, 1.0, 0.3),
                     ("hsv_v", "Brightness", 0.0, 1.0, 0.2), ("mosaic", "Mosaic", 0.0, 1.0, 0.0),
                     ("mixup", "MixUp", 0.0, 1.0, 0.0), ("copy_paste", "Copy/paste", 0.0, 1.0, 0.0)]
        for index, (key, title, low, high, value) in enumerate(aug_specs):
            widget = QDoubleSpinBox(); widget.setRange(low, high); widget.setDecimals(3); widget.setSingleStep(0.01); widget.setValue(value)
            widget.setToolTip("Ultralytics applies geometry consistently to image and semantic mask; photometric settings affect RGB only.")
            self.augmentation_widgets[key] = widget
            aug_form.addWidget(QLabel(title), index // 5, (index % 5) * 2)
            aug_form.addWidget(widget, index // 5, (index % 5) * 2 + 1)
        self.amp_checkbox = QCheckBox("AMP mixed precision"); self.amp_checkbox.setChecked(True)
        self.deterministic_checkbox = QCheckBox("Deterministic training"); self.deterministic_checkbox.setChecked(True)
        self.cosine_checkbox = QCheckBox("Cosine learning-rate schedule")
        aug_form.addWidget(self.amp_checkbox, 2, 0, 1, 2); aug_form.addWidget(self.deterministic_checkbox, 2, 2, 1, 2); aug_form.addWidget(self.cosine_checkbox, 2, 4, 1, 2)
        settings_layout.addWidget(aug_box)
        settings_scroll.setWidget(content)
        layout.addWidget(settings_scroll, 1)
        actions = QHBoxLayout()
        self.effective_summary = QLineEdit(); self.effective_summary.setReadOnly(True); self.effective_summary.setPlaceholderText("Effective training settings will be shown before start")
        self.effective_summary.setToolTip("The exact model, dataset, device and hyperparameters are displayed and confirmed before a training run starts.")
        actions.addWidget(self.effective_summary, 1)
        self.start_training_button = button("Start fine-tuning…", self.start_training, primary=True)
        actions.addWidget(self.start_training_button)
        self.pause_button = button("Pause at epoch", self.pause_training); self.pause_button.setEnabled(False); actions.addWidget(self.pause_button)
        self.resume_button = button("Resume pause", self.resume_training); self.resume_button.setEnabled(False); actions.addWidget(self.resume_button)
        self.stop_button = button("Safe stop", self.stop_training, danger=True); self.stop_button.setEnabled(False); actions.addWidget(self.stop_button)
        layout.addLayout(actions)
        self.training_progress = QProgressBar(); self.training_progress.setValue(0); layout.addWidget(self.training_progress)
        self.training_status = QLabel("No active run."); self.training_status.setObjectName("Muted"); layout.addWidget(self.training_status)

    def _build_experiments_page(self, layout):
        actions = QHBoxLayout()
        actions.addWidget(button("Refresh", self._refresh_experiments))
        actions.addWidget(button("Open selected run", self.open_selected_experiment))
        actions.addWidget(button("Use best.pt as new fine-tune start", self.use_experiment_best))
        actions.addWidget(button("Resume selected last.pt", self.resume_experiment))
        actions.addStretch(1)
        layout.addLayout(actions)
        self.experiment_table = table_widget(["Run", "Created", "Status", "Best checkpoint", "Last checkpoint", "Summary"])
        layout.addWidget(self.experiment_table, 1)
        config_box = QGroupBox("Training configurations")
        config_row = QHBoxLayout(config_box)
        self.config_combo = QComboBox()
        config_row.addWidget(self.config_combo, 1)
        config_row.addWidget(button("Load", self.load_selected_config))
        config_row.addWidget(button("Save", self.save_current_config))
        config_row.addWidget(button("Save as…", self.save_config_as))
        config_row.addWidget(button("Rename", self.rename_config))
        config_row.addWidget(button("Delete", self.delete_config, danger=True))
        config_row.addWidget(button("Set default", self.set_default_config))
        config_row.addWidget(button("Import…", self.import_config))
        config_row.addWidget(button("Export…", self.export_config))
        layout.addWidget(config_box)
        library_box = QGroupBox("Model library · registered semantic artifacts")
        library_layout = QVBoxLayout(library_box)
        library_actions = QHBoxLayout()
        library_actions.addWidget(button("Refresh library", self._refresh_model_registry))
        library_actions.addWidget(button("Register model…", self.register_model_file))
        library_actions.addWidget(button("Use selected model", self.use_registered_model))
        library_actions.addWidget(button("Remove entry…", self.remove_registered_model))
        library_actions.addStretch(1)
        library_layout.addLayout(library_actions)
        self.model_registry_table = table_widget(["Model", "Format", "Task", "State", "Dataset / run", "Metrics", "Path"])
        self.model_registry_table.setMaximumHeight(190)
        library_layout.addWidget(self.model_registry_table)
        library_hint = QLabel("Training checkpoints and ONNX exports are added automatically. Registering or removing an entry never moves or deletes the model file.")
        library_hint.setWordWrap(True); library_hint.setObjectName("Muted")
        library_layout.addWidget(library_hint)
        layout.addWidget(library_box)

    def _build_evaluation_page(self, layout):
        model_line = QHBoxLayout()
        self.eval_model = QLineEdit(); self.eval_model.setPlaceholderText("YOLO26 semantic .pt or .onnx model")
        self.eval_dataset = QLineEdit(); self.eval_dataset.setPlaceholderText("Prepared dataset folder")
        self.eval_split = QComboBox(); self.eval_split.addItems(["test", "val"])
        model_line.addWidget(self.eval_model, 2); model_line.addWidget(button("Model…", lambda: self._choose_file(self.eval_model, "Models (*.pt *.onnx)")))
        model_line.addWidget(self.eval_dataset, 2); model_line.addWidget(button("Dataset…", self.choose_eval_dataset))
        model_line.addWidget(self.eval_split)
        model_line.addWidget(button("Evaluate split", self.start_evaluation, primary=True))
        layout.addLayout(model_line)
        note = QLabel("Test metrics are independent final measurements; use validation metrics for checkpoint selection. Per-sample RGB, masks, overlays, and error maps are saved with the report.")
        note.setWordWrap(True); note.setObjectName("Muted"); layout.addWidget(note)
        self.eval_status = QLabel("No evaluation run."); self.eval_status.setObjectName("Muted"); layout.addWidget(self.eval_status)
        self.eval_metrics = QPlainTextEdit(); self.eval_metrics.setReadOnly(True); self.eval_metrics.setMaximumHeight(190); layout.addWidget(self.eval_metrics)
        self.eval_examples = table_widget(["Image", "Town", "Weather", "Traffic", "Prediction"])
        self.eval_examples.cellDoubleClicked.connect(self.open_eval_example)
        layout.addWidget(self.eval_examples, 1)

    def _build_testing_page(self, layout):
        model_row = QHBoxLayout()
        self.test_model = QLineEdit(); self.test_model.setPlaceholderText("Semantic checkpoint (.pt) or ONNX (.onnx)")
        model_row.addWidget(self.test_model, 1); model_row.addWidget(button("Choose model…", lambda: self._choose_file(self.test_model, "Models (*.pt *.onnx)")))
        self.test_imgsz = QSpinBox(); self.test_imgsz.setRange(128, 2048); self.test_imgsz.setValue(512); self.test_imgsz.setPrefix("Image size ")
        model_row.addWidget(self.test_imgsz); layout.addLayout(model_row)
        sources = QHBoxLayout()
        sources.addWidget(button("Single image…", self.choose_test_image, primary=True))
        sources.addWidget(button("Image folder…", self.choose_test_folder))
        sources.addWidget(button("Video…", self.choose_test_video))
        sources.addWidget(button("Webcam", self.toggle_webcam))
        self.opacity = QDoubleSpinBox(); self.opacity.setRange(0, 1); self.opacity.setSingleStep(0.05); self.opacity.setValue(0.42); self.opacity.setPrefix("Overlay ")
        sources.addWidget(self.opacity); sources.addStretch(1)
        layout.addLayout(sources)
        self.test_status = QLabel("Choose a semantic model and input source."); self.test_status.setObjectName("Muted"); layout.addWidget(self.test_status)
        preview_grid = QGridLayout()
        self.test_previews = {}
        for index, title in enumerate(("Original RGB", "Semantic class map", "Prediction overlay", "Drivable area only", "Lane lines only", "Vehicles only")):
            tile = QWidget(); tile_layout = QVBoxLayout(tile); tile_layout.setContentsMargins(2, 2, 2, 2)
            caption = QLabel(title); caption.setObjectName("Muted"); tile_layout.addWidget(caption)
            preview = QLabel(title); preview.setAlignment(Qt.AlignmentFlag.AlignCenter); preview.setMinimumSize(260, 190)
            preview.setStyleSheet("background:#0b1118;border:1px solid #29394b;border-radius:8px;color:#74869a;")
            tile_layout.addWidget(preview, 1)
            self.test_previews[title] = preview
            preview_grid.addWidget(tile, index // 3, index % 3)
        self.test_preview = self.test_previews["Prediction overlay"]
        layout.addLayout(preview_grid, 1)

    def _build_onnx_page(self, layout):
        top = QHBoxLayout()
        self.onnx_source = QLineEdit(); self.onnx_source.setPlaceholderText("Trained semantic PyTorch checkpoint")
        top.addWidget(self.onnx_source, 1); top.addWidget(button("Checkpoint…", lambda: self._choose_file(self.onnx_source, "PyTorch checkpoint (*.pt)")))
        self.onnx_imgsz = QSpinBox(); self.onnx_imgsz.setRange(128, 2048); self.onnx_imgsz.setValue(512); self.onnx_imgsz.setPrefix("Image size "); top.addWidget(self.onnx_imgsz)
        self.onnx_opset = QSpinBox(); self.onnx_opset.setRange(12, 21); self.onnx_opset.setValue(17); self.onnx_opset.setPrefix("Opset "); top.addWidget(self.onnx_opset)
        self.onnx_dynamic = QCheckBox("Dynamic shapes"); top.addWidget(self.onnx_dynamic)
        self.onnx_simplify = QCheckBox("Simplify graph"); top.addWidget(self.onnx_simplify)
        top.addWidget(button("Export + validate", self.start_onnx_export, primary=True))
        layout.addLayout(top)
        comparison = QHBoxLayout()
        self.onnx_path = QLineEdit(); self.onnx_path.setPlaceholderText("Exported ONNX file")
        comparison.addWidget(self.onnx_path, 1); comparison.addWidget(button("Compare…", self.choose_onnx_for_compare)); comparison.addWidget(button("PyTorch / ONNX parity…", self.start_parity)); comparison.addWidget(button("Benchmark…", self.start_benchmark)); comparison.addWidget(button("Deployment package…", self.start_deployment))
        layout.addLayout(comparison)
        self.onnx_status = QLabel("ONNX export, graph validation, runtime metadata, parity and benchmarking results will appear here.")
        self.onnx_status.setObjectName("Muted"); self.onnx_status.setWordWrap(True); layout.addWidget(self.onnx_status)
        self.onnx_log = QPlainTextEdit(); self.onnx_log.setReadOnly(True); layout.addWidget(self.onnx_log, 1)

    def _build_settings_page(self, layout):
        workspace_box = QGroupBox("Application workspace")
        form = QFormLayout(workspace_box)
        self.settings_workspace = QLineEdit()
        workspace_row = QHBoxLayout(); workspace_row.addWidget(self.settings_workspace, 1); workspace_row.addWidget(button("Browse…", self.choose_workspace))
        workspace_widget = QWidget(); workspace_widget.setLayout(workspace_row); form.addRow("Studio output directory", workspace_widget)
        self.settings_roots = QPlainTextEdit(); self.settings_roots.setPlaceholderText("One existing collection root per line")
        form.addRow("Read-only dataset roots", self.settings_roots)
        layout.addWidget(workspace_box)
        env_box = QGroupBox("Detected environment")
        env_layout = QVBoxLayout(env_box)
        self.environment_text = QPlainTextEdit(); self.environment_text.setReadOnly(True); env_layout.addWidget(self.environment_text)
        layout.addWidget(env_box, 1)
        actions = QHBoxLayout()
        actions.addWidget(button("Refresh environment", self._update_environment_text))
        actions.addWidget(button("Save preferences", self.save_preferences, primary=True))
        actions.addStretch(1); layout.addLayout(actions)
        layout.addStretch(1)

    def _build_logs_page(self, layout):
        actions = QHBoxLayout()
        self.log_level = QComboBox(); self.log_level.addItems(["All", "Info", "Warning", "Error", "Debug"])
        actions.addWidget(QLabel("Level")); actions.addWidget(self.log_level)
        actions.addWidget(button("Export logs…", self.export_logs))
        actions.addWidget(button("Clear view", lambda: self.logs_view.clear()))
        layout.addLayout(actions)
        self.logs_view = QPlainTextEdit(); self.logs_view.setReadOnly(True); layout.addWidget(self.logs_view, 1)

    def _sync_config_controls(self):
        roots = self.config.get("dataset_roots", [])
        root_text = "\n".join(map(str, roots))
        if hasattr(self, "roots_edit"):
            self.roots_edit.setText(root_text)
            self.settings_roots.setPlainText(root_text)
            self.settings_workspace.setText(self.config.get("workspace", ""))
        model = self.config.get("model", {})
        if hasattr(self, "model_size"):
            self.model_size.setCurrentText(str(model.get("size", "l")))
            self.checkpoint_edit.setText(str(model.get("checkpoint", "")))
            names = model.get("classes", {})
            for index, edit in enumerate(self.class_names):
                edit.setText(str(names.get(index, names.get(str(index), edit.text()))))
        if hasattr(self, "training_widgets"):
            for key, widget in self.training_widgets.items():
                value = self.config.get("training", {}).get(key)
                if value is None:
                    continue
                self._set_widget(widget, value)
            for key, widget in self.augmentation_widgets.items():
                self._set_widget(widget, self.config.get("training", {}).get(key, widget.value()))
            self.amp_checkbox.setChecked(bool(self.config.get("training", {}).get("amp", True)))
            self.deterministic_checkbox.setChecked(bool(self.config.get("training", {}).get("deterministic", True)))
            self.cosine_checkbox.setChecked(bool(self.config.get("training", {}).get("cos_lr", False)))
        dataset = self.config.get("dataset", {})
        if hasattr(self, "data_yaml_edit"):
            self.data_yaml_edit.setText(str(dataset.get("data_yaml", "")))
            self.eval_dataset.setText(str(dataset.get("prepared_dataset", "")))
        split = self.config.get("split", {})
        if hasattr(self, "holdout_towns"):
            selected_towns = {str(value) for value in split.get("holdout_towns", [])}
            for index in range(self.holdout_towns.count()):
                item = self.holdout_towns.item(index)
                item.setCheckState(Qt.CheckState.Checked if item.text() in selected_towns else Qt.CheckState.Unchecked)
        if hasattr(self, "train_ratio"):
            self.train_ratio.setValue(float(split.get("train", 0.8)) * 100)
            self.val_ratio.setValue(float(split.get("val", 0.1)) * 100)
            self.test_ratio.setValue(float(split.get("test", 0.1)) * 100)
            self.split_seed.setValue(int(split.get("seed", 42)))
        inference = self.config.get("inference", {})
        if hasattr(self, "test_imgsz"):
            self.test_imgsz.setValue(int(inference.get("imgsz", 512)))
            self.opacity.setValue(float(inference.get("overlay_opacity", 0.42)))
        export = self.config.get("export", {})
        if hasattr(self, "onnx_imgsz"):
            self.onnx_imgsz.setValue(int(export.get("imgsz", 512)))
            self.onnx_opset.setValue(int(export.get("opset", 17)))
            self.onnx_dynamic.setChecked(bool(export.get("dynamic", False)))
            self.onnx_simplify.setChecked(bool(export.get("simplify", False)))

    @staticmethod
    def _set_widget(widget, value):
        if isinstance(widget, QComboBox):
            text = str(value)
            if widget.findText(text) < 0 and text in ("True", "False"):
                pass
            else:
                widget.setCurrentText(text)
        elif isinstance(widget, QCheckBox):
            widget.setChecked(bool(value))
        elif isinstance(widget, QLineEdit):
            widget.setText(str(value))
        elif hasattr(widget, "setValue"):
            try:
                widget.setValue(float(value) if isinstance(widget, QDoubleSpinBox) else int(value))
            except (ValueError, TypeError):
                pass

    def _check_list(self, widget: QListWidget, checked: bool):
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for i in range(widget.count()):
            widget.item(i).setCheckState(state)

    def _browse_roots(self):
        root = QFileDialog.getExistingDirectory(self, "Select a dataset root", str(PROJECT_ROOT))
        if root:
            values = [v for v in self.roots_edit.text().splitlines() if v.strip()]
            if root not in values:
                values.append(root)
            self.roots_edit.setText("\n".join(values))
            self.settings_roots.setPlainText("\n".join(values))

    def _choose_file(self, widget: QLineEdit, filter_text: str):
        path, _ = QFileDialog.getOpenFileName(self, "Select file", str(PROJECT_ROOT), filter_text)
        if path:
            widget.setText(path)

    def _start_busy(self, label: str):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "Operation running", "Wait for the active operation to finish before starting another one.")
            return False
        self.busy_text.setText(label)
        self.statusBar().showMessage(label)
        return True

    def _start_operation(self, label, function, *args, on_done=None, **kwargs):
        if not self._start_busy(label):
            return
        worker = OperationWorker(function, *args, **kwargs)
        self.worker = worker
        worker.progress.connect(self._operation_progress)
        worker.log.connect(self._append_log)
        worker.failed.connect(lambda error, op=label: self._operation_failed(op, error))
        worker.done.connect(lambda value, cb=on_done, op=label: self._operation_done(op, value, cb))
        worker.start()

    def _operation_progress(self, event):
        values = event.get("args", ())
        if values:
            self.busy_text.setText(f"{self.busy_text.text()} · {values}")

    def _operation_done(self, label, value, callback):
        self.busy_text.setText("Ready")
        self.statusBar().showMessage(f"{label} completed", 7000)
        self._append_log(f"{label} completed")
        if callback:
            try:
                callback(value)
            except Exception:
                self._show_error(f"Could not display {label} result", traceback.format_exc())

    def _operation_failed(self, label, error):
        self.busy_text.setText("Ready")
        self.statusBar().showMessage(f"{label} failed", 10000)
        self._append_log(f"ERROR · {label}\n{error}")
        self._show_error(f"{label} failed", error)

    def _show_error(self, title, error):
        box = QMessageBox(self); box.setIcon(QMessageBox.Icon.Critical); box.setWindowTitle(title); box.setText(title); box.setDetailedText(str(error)); box.exec()

    def _append_log(self, message: str):
        line = f"{datetime.now().astimezone().strftime('%H:%M:%S')} · {message}"
        self._log_lines.append(line)
        if hasattr(self, "logs_view"):
            level = self.log_level.currentText().lower() if hasattr(self, "log_level") else "all"
            if level == "all" or level in message.lower():
                self.logs_view.appendPlainText(line)
        if hasattr(self, "dashboard_info") and "ERROR" in message:
            self.dashboard_info.setText(message[:1000])

    def _dataset_roots(self):
        return [Path(v.strip()).expanduser() for v in self.roots_edit.text().splitlines() if v.strip()]

    def rescan(self):
        if not self._start_busy("Scanning CARLA collections…"):
            return
        roots = [str(p) for p in self._dataset_roots()]
        worker = DatasetScanWorker(roots)
        self.worker = worker
        worker.progress.connect(self.busy_text.setText)
        worker.failed.connect(lambda error: self._operation_failed("Dataset scan", error))
        worker.done.connect(self._scan_done)
        worker.start()

    def _scan_done(self, collections):
        self.collections = collections
        self.validation_report = None
        selected_before = {self.collection_table.item(r, 1).text() for r in range(self.collection_table.rowCount()) if self.collection_table.item(r, 0).checkState() == Qt.CheckState.Checked}
        self.collection_table.setRowCount(0)
        for collection in self.collections:
            row = self.collection_table.rowCount(); self.collection_table.insertRow(row)
            choose = QTableWidgetItem(); choose.setFlags(choose.flags() | Qt.ItemFlag.ItemIsUserCheckable); choose.setCheckState(Qt.CheckState.Checked if collection.name in selected_before or (not selected_before and collection.annotated_count > 0) else Qt.CheckState.Unchecked)
            self.collection_table.setItem(row, 0, choose)
            state = "Ready" if collection.annotated_count == collection.image_count and collection.image_count else ("Partial" if collection.annotated_count else "No labels")
            values = [collection.name, ", ".join(collection.towns) or "—", str(collection.image_count), str(collection.annotated_count), str(collection.missing_mask_count), ", ".join(collection.weather) or "Unknown", ", ".join(collection.traffic) or "Unknown", state, ", ".join(collection.dimensions) or "Unknown"]
            for col, value in enumerate(values, 1):
                self.collection_table.setItem(row, col, QTableWidgetItem(value))
        towns = sorted({s.town for c in self.collections for s in c.samples})
        weathers = sorted({s.weather for c in self.collections for s in c.samples})
        traffic = sorted({s.traffic for c in self.collections for s in c.samples})
        self.town_filter.clear(); self.weather_filter.clear(); self.traffic_filter.clear()
        for target, values in ((self.town_filter, towns), (self.weather_filter, weathers), (self.traffic_filter, traffic)):
            for value in values:
                item = QListWidgetItem(value); item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable); item.setCheckState(Qt.CheckState.Checked); target.addItem(item)
        held_out = {str(value) for value in self.config.get("split", {}).get("holdout_towns", [])}
        self.holdout_towns.clear()
        for town in towns:
            item = QListWidgetItem(town); item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if town in held_out else Qt.CheckState.Unchecked)
            self.holdout_towns.addItem(item)
        total_images = sum(c.image_count for c in collections); total_annotated = sum(c.annotated_count for c in collections)
        self.dataset_summary.setText(f"{len(collections)} collections · {total_images:,} images · {total_annotated:,} manifest-declared combined masks")
        self.dashboard_cards["collections"].value_label.setText(f"{len(collections):,}")
        self.dashboard_cards["images"].value_label.setText(f"{total_images:,}")
        self.dashboard_cards["annotated"].value_label.setText(f"{total_annotated:,}")
        self.dashboard_info.setText("Collections are indexed from the selected roots. Source images and annotations remain read only; prepared copies are written under the Studio workspace.")
        self.busy_text.setText("Ready"); self.statusBar().showMessage("Collection scan complete", 7000)
        self._append_log(f"Scanned {len(collections)} collections, {total_images} images, {total_annotated} annotated pairs")

    def _set_collection_checks(self, checked: bool, annotated_only=False):
        for row in range(self.collection_table.rowCount()):
            is_annotated = int(self.collection_table.item(row, 4).text()) > 0
            self.collection_table.item(row, 0).setCheckState(Qt.CheckState.Checked if checked and (not annotated_only or is_annotated) else Qt.CheckState.Unchecked)

    @staticmethod
    def _checked_values(widget: QListWidget) -> set[str]:
        return {widget.item(i).text() for i in range(widget.count()) if widget.item(i).checkState() == Qt.CheckState.Checked}

    def _selected_samples(self) -> list[Sample]:
        selected = {self.collection_table.item(r, 1).text() for r in range(self.collection_table.rowCount()) if self.collection_table.item(r, 0).checkState() == Qt.CheckState.Checked}
        towns, weathers, traffic = self._checked_values(self.town_filter), self._checked_values(self.weather_filter), self._checked_values(self.traffic_filter)
        return [sample for collection in self.collections if collection.name in selected for sample in collection.samples
                if (not towns or sample.town in towns) and (not weathers or sample.weather in weathers) and (not traffic or sample.traffic in traffic)]

    def validate_selected(self):
        samples = self._selected_samples()
        if not samples:
            QMessageBox.information(self, "No samples", "Select one or more collections and town/weather filters first.")
            return
        ignore = 255 if self.ignore_checkbox.isChecked() else None
        if not self._start_busy("Validating selected dataset…"):
            return
        worker = ValidationWorker(samples, ignore)
        self.worker = worker
        worker.progress.connect(self.quality_status.setText)
        worker.failed.connect(lambda err: self._operation_failed("Dataset validation", err))
        worker.done.connect(self._validation_done)
        worker.start()
        self.stack.setCurrentIndex(2)

    def _validation_done(self, report: ValidationReport):
        self.validation_report = report
        self.export_validation_button.setEnabled(True)
        counts = {}
        for issue in report.issues:
            counts[issue.severity] = counts.get(issue.severity, 0) + 1
        self.quality_status.setText(f"Checked {report.checked:,} · valid pairs {len(report.valid):,} · invalid samples {report.invalid_count:,} · errors {counts.get('error',0):,} · warnings {counts.get('warning',0):,}")
        total = max(1, sum(report.class_pixels.values()))
        dist = " · ".join(
            f"{k} {name}: {report.class_pixels.get(k,0):,} px "
            f"({100*report.class_pixels.get(k,0)/total:.2f}%) / "
            f"{report.class_images.get(k,0):,} images"
            for k, name in (
                (0, "background"), (1, "drivable"),
                (2, "lane"), (3, "vehicles"),
            )
        )
        self.quality_stats.setText(f"Valid image-mask pairs: {len(report.valid):,} / {report.checked:,}. Dimensions: {report.dimensions}. Pixel distribution: {dist}")
        self.issue_table.setRowCount(0)
        display_issues = report.issues[:3000]
        for issue in display_issues:
            row = self.issue_table.rowCount(); self.issue_table.insertRow(row)
            values = (issue.severity.upper(), issue.code, issue.message, issue.image, issue.mask)
            for col, value in enumerate(values):
                self.issue_table.setItem(row, col, QTableWidgetItem(str(value)))
        if len(report.issues) > len(display_issues):
            self.quality_status.setText(self.quality_status.text() + f" · showing first {len(display_issues):,} of {len(report.issues):,} issues")
        self.busy_text.setText("Ready")
        self._append_log(f"Validation complete: {len(report.valid)} valid pairs and {len(report.issues)} report entries")

    def export_validation(self):
        if self.validation_report is None:
            QMessageBox.information(self, "No validation report", "Validate selected samples before exporting a report.")
            return
        default = Path(self.config.get("workspace", APP_DIR / "workspace")) / "reports" / "dataset_validation.json"
        path, _ = QFileDialog.getSaveFileName(self, "Export complete validation report", str(default), "JSON report (*.json)")
        if not path:
            return
        try:
            result = export_validation_report(self.validation_report, path)
            self._append_log(f"Exported full dataset validation report: {result}")
            QMessageBox.information(self, "Report exported", f"Complete validation report saved to:\n{result}")
        except Exception as exc:
            self._show_error("Could not export validation report", str(exc))

    def prepare_selected(self):
        if self.validation_report is None:
            QMessageBox.warning(self, "Validation required", "Validate the selected source samples before preparing a dataset.")
            return
        ratios = {"train": self.train_ratio.value()/100, "val": self.val_ratio.value()/100, "test": self.test_ratio.value()/100}
        if abs(sum(ratios.values()) - 1) > 0.001:
            QMessageBox.warning(self, "Split total", "Train, validation, and test percentages must sum to 100%.")
            return
        if not self.validation_report.valid:
            QMessageBox.warning(self, "No valid pairs", "The validation report contains no valid image-mask pairs.")
            return
        try:
            dataset_name = self._safe_config_name(self.dataset_name.text())
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid dataset name", str(exc)); return
        output = Path(self.config.get("workspace", "")) / "prepared_datasets" / dataset_name
        holdout = self._checked_values(self.holdout_towns)
        summary = f"Pairs: {len(self.validation_report.valid):,}\nTrain / validation / test: {ratios}\nGrouping: whole collection session, with exact duplicate images kept together\nHeld-out test towns: {sorted(holdout) or 'none'}\nOutput: {output}\n\nContinue with a new prepared copy?"
        if QMessageBox.question(self, "Prepare dataset", summary) != QMessageBox.StandardButton.Yes:
            return
        self.prepare_button.setEnabled(False)
        self._start_operation("Dataset preparation", prepare_dataset, self.validation_report.valid, output, ratios, self.split_seed.value(), False, 255 if self.ignore_checkbox.isChecked() else None, holdout, on_done=self._prepared_done)

    def _prepared_done(self, result):
        self.prepared_dataset = result["path"]
        self.data_yaml = result["data_yaml"]
        self.data_yaml_edit.setText(self.data_yaml)
        self.eval_dataset.setText(self.prepared_dataset)
        self.config.setdefault("dataset", {}).update({"data_yaml": self.data_yaml, "prepared_dataset": self.prepared_dataset})
        self.prepared_status.setText(f"Prepared copy: {self.prepared_dataset} · {result['counts']} · warnings: {result['report']['warnings'] or 'none'}")
        self.prepare_button.setEnabled(True)
        self._append_log(f"Prepared dataset with split counts {result['counts']}")

    def _choose_data_yaml(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select semantic dataset YAML", str(Path(self.config.get("workspace", APP_DIR))), "YAML (*.yaml *.yml)")
        if path:
            self.data_yaml_edit.setText(path); self.data_yaml = path; self.prepared_dataset = str(Path(path).parent); self.eval_dataset.setText(self.prepared_dataset)
            self.config.setdefault("dataset", {}).update({"data_yaml": self.data_yaml, "prepared_dataset": self.prepared_dataset})

    def use_prepared_dataset(self):
        workspace = Path(self.config.get("workspace", "")) / "prepared_datasets"
        if not workspace.is_dir():
            QMessageBox.information(self, "No prepared datasets", "Prepare a dataset copy in Dataset Quality & Splits first.")
            return
        folders = sorted((p for p in workspace.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
        if not folders:
            return
        selected, ok = QInputDialog.getItem(self, "Select prepared dataset", "Dataset", [str(x) for x in folders], 0, False)
        if ok:
            self.prepared_dataset = selected; self.data_yaml = str(Path(selected) / "data.yaml")
            self.data_yaml_edit.setText(self.data_yaml); self.eval_dataset.setText(selected)
            self.config.setdefault("dataset", {}).update({"data_yaml": self.data_yaml, "prepared_dataset": self.prepared_dataset})

    def _update_model_name(self):
        if hasattr(self, "model_name"):
            self.model_name.setText(f"YOLO26{self.model_size.currentText()}-sem · semantic")
        if hasattr(self, "dashboard_cards"):
            self.dashboard_cards["model"].value_label.setText(f"YOLO26{self.model_size.currentText()}-sem")

    def download_checkpoint(self):
        size = self.model_size.currentText()
        destination = Path(self.config.get("workspace", "")) / "checkpoints" / f"yolo26{size}-sem.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if QMessageBox.question(self, "Checkpoint exists", f"Use the already downloaded checkpoint?\n{destination}") == QMessageBox.StandardButton.Yes:
                self.checkpoint_edit.setText(str(destination)); self.config["model"]["checkpoint"] = str(destination); return
            return
        if QMessageBox.question(self, "Download official checkpoint", f"Fetch official Ultralytics semantic weights for YOLO26{size}-sem?\n\nThis explicit action may use the network and saves a new copy at:\n{destination}") != QMessageBox.StandardButton.Yes:
            return
        self._start_operation("Official checkpoint download", download_official_checkpoint, size, destination, on_done=lambda path: (self.checkpoint_edit.setText(path), self._append_log(f"Downloaded explicit official checkpoint: {path}")))

    def _training_config(self) -> dict:
        config = json.loads(json.dumps(self.config))
        config.setdefault("model", {})
        config["model"].update({"family": "yolo26", "size": self.model_size.currentText(), "checkpoint": self.checkpoint_edit.text().strip(), "task": "semantic", "classes": {i: edit.text().strip() for i, edit in enumerate(self.class_names)}})
        config["training"] = {}
        for key, widget in self.training_widgets.items():
            if isinstance(widget, QComboBox):
                value = widget.currentText()
                if key == "cache":
                    value = False if value == "False" else value
            elif isinstance(widget, QLineEdit):
                value = widget.text().strip()
            elif isinstance(widget, QDoubleSpinBox):
                value = widget.value()
            else:
                value = widget.value()
            config["training"][key] = value
        config["training"].update({key: widget.value() for key, widget in self.augmentation_widgets.items()})
        config["training"].update({"amp": self.amp_checkbox.isChecked(), "deterministic": self.deterministic_checkbox.isChecked(), "cos_lr": self.cosine_checkbox.isChecked(), "ignore_label": 255 if self.ignore_checkbox.isChecked() else None})
        config["workspace"] = str(self.settings_workspace.text().strip() or APP_DIR / "workspace")
        config["dataset_roots"] = [v.strip() for v in self.settings_roots.toPlainText().splitlines() if v.strip()]
        data_yaml = self.data_yaml_edit.text().strip()
        prepared = self.eval_dataset.text().strip() or (str(Path(data_yaml).parent) if data_yaml else "")
        config["dataset"] = {"data_yaml": data_yaml, "prepared_dataset": prepared}
        config["split"] = {
            **config.get("split", {}),
            "train": self.train_ratio.value() / 100,
            "val": self.val_ratio.value() / 100,
            "test": self.test_ratio.value() / 100,
            "seed": self.split_seed.value(),
            "strategy": "collection_session",
            "holdout_towns": sorted(self._checked_values(self.holdout_towns)) if self.holdout_towns.count() else list(config.get("split", {}).get("holdout_towns", [])),
        }
        config["inference"] = {
            **config.get("inference", {}),
            "imgsz": self.test_imgsz.value(),
            "overlay_opacity": self.opacity.value(),
        }
        config["export"] = {
            **config.get("export", {}),
            "format": "onnx", "imgsz": self.onnx_imgsz.value(),
            "opset": self.onnx_opset.value(), "dynamic": self.onnx_dynamic.isChecked(),
            "simplify": self.onnx_simplify.isChecked(),
        }
        return config

    def start_training(self):
        self.data_yaml = self.data_yaml_edit.text().strip()
        if not self.data_yaml or not Path(self.data_yaml).is_file():
            QMessageBox.warning(self, "Dataset not selected", "Choose or prepare a dataset data.yaml before training.")
            return
        config = self._training_config()
        checkpoint = config["model"].get("checkpoint", "")
        if checkpoint and not Path(checkpoint).is_file():
            QMessageBox.warning(self, "Checkpoint missing", f"The selected checkpoint does not exist:\n{checkpoint}\nUse the explicit official checkpoint download button or relink a local file.")
            return
        config["model"]["classes"] = {int(k): v for k, v in config["model"]["classes"].items()}
        summary = {"task": "semantic", "architecture": f"YOLO26{config['model']['size']}-sem", "checkpoint": checkpoint or "architecture YAML only (random initialization)", "dataset": self.data_yaml, "device": config["training"].get("device"), "epochs": config["training"].get("epochs"), "image_size": config["training"].get("imgsz"), "batch": config["training"].get("batch"), "optimizer": config["training"].get("optimizer"), "seed": config["training"].get("seed")}
        self.effective_summary.setText(json.dumps(summary, ensure_ascii=False))
        if QMessageBox.question(self, "Confirm effective training configuration", json.dumps(summary, indent=2, ensure_ascii=False) + "\n\nStart this training run?") != QMessageBox.StandardButton.Yes:
            return
        self._launch_training(config)

    def _launch_training(self, config, resume=None):
        worker = TrainingWorker(config, self.data_yaml_edit.text().strip(), config["workspace"], resume)
        self.training_worker = worker; self.worker = worker
        worker.progress.connect(self._training_progress)
        worker.log.connect(self._append_log)
        worker.failed.connect(self._training_failed)
        worker.done.connect(self._training_done)
        for control in (self.start_training_button, self.pause_button, self.resume_button, self.stop_button):
            control.setEnabled(False)
        self.start_training_button.setEnabled(False); self.stop_button.setEnabled(True); self.pause_button.setEnabled(True)
        self.training_status.setText("Training worker started…")
        self.training_progress.setValue(0)
        self._append_log("Training started. Stop and pause take effect at the next safe epoch boundary.")
        worker.start()

    def _training_progress(self, event):
        state = event.get("state", "")
        if state in {"running", "epoch_complete"}:
            epoch, total = int(event.get("epoch", 0)), int(event.get("epochs", 0))
            if total:
                self.training_progress.setValue(round(100 * epoch / total))
            batch = event.get("batch"); batches = event.get("batches")
            batch_text = f" · batch {batch}/{batches}" if batch and batches else ""
            self.training_status.setText(f"{state.replace('_',' ').title()} · epoch {epoch}/{total}{batch_text} · {event.get('metrics', {})}")
            self.pause_button.setEnabled(state == "running"); self.resume_button.setEnabled(False)
        elif state == "paused":
            self.training_status.setText(event.get("message", "Paused at epoch boundary")); self.resume_button.setEnabled(True); self.pause_button.setEnabled(False)
        else:
            self.training_status.setText(event.get("message", state))

    def pause_training(self):
        if self.training_worker:
            self.training_worker.request_pause(); self.pause_button.setEnabled(False)

    def resume_training(self):
        if self.training_worker:
            self.training_worker.request_resume(); self.resume_button.setEnabled(False)

    def stop_training(self):
        if self.training_worker and QMessageBox.question(self, "Safe stop", "Stop after the current epoch checkpoint is written? The last resumable checkpoint will be preserved.") == QMessageBox.StandardButton.Yes:
            self.training_worker.request_stop(); self.stop_button.setEnabled(False)

    def _training_failed(self, error):
        self._set_training_idle()
        self.training_status.setText("Training failed. Any saved checkpoint remains available for resume.")
        self._append_log("ERROR · training failed\n" + error)
        self._refresh_experiments()
        self._show_error("Training failed", error)

    def _training_done(self, result):
        self._set_training_idle()
        self.training_status.setText(f"Run finished · {result['run_id']} · best={result['best']}")
        self.active_model = result["best"] if Path(result["best"]).is_file() else result["last"]
        self.test_model.setText(self.active_model); self.eval_model.setText(self.active_model); self.onnx_source.setText(self.active_model)
        details = {"run_id": result["run_id"], "dataset": self.data_yaml_edit.text().strip(), "metrics": result.get("metrics", {}), "architecture": f"YOLO26{self.model_size.currentText()}-sem"}
        for candidate in (result.get("best"), result.get("last")):
            if candidate and Path(candidate).is_file():
                try:
                    self._register_model_artifact(candidate, details)
                except Exception as exc:
                    self._append_log(f"ERROR · model library registration failed: {exc}")
        self._append_log(f"Training finished: {json.dumps(result, default=str)}")
        self._refresh_experiments(); self._refresh_dashboard_runs()
        QMessageBox.information(self, "Training finished", f"Run: {result['run_id']}\nBest: {result['best']}\nLast: {result['last']}")

    def _set_training_idle(self):
        for control in (self.start_training_button, self.pause_button, self.resume_button, self.stop_button):
            control.setEnabled(control is self.start_training_button)

    def _refresh_configs(self):
        if not hasattr(self, "config_combo"):
            return
        folder = Path(self.config.get("workspace", APP_DIR / "workspace")) / "configs"
        folder.mkdir(parents=True, exist_ok=True)
        current = self.config_combo.currentText()
        self.config_combo.clear()
        for path in sorted(folder.glob("*.json")):
            self.config_combo.addItem(path.stem, str(path))
        preferred = self.config_combo.findData(str(self.config_path)) if self.config_path else -1
        index = preferred if preferred >= 0 else self.config_combo.findText(current)
        if index >= 0:
            self.config_combo.setCurrentIndex(index)

    @staticmethod
    def _safe_config_name(name: str) -> str:
        value = name.strip()
        if not value or value in {".", ".."} or len(value) > 80:
            raise ValueError("Configuration name must contain 1–80 characters.")
        windows_device = value.split(".", 1)[0].upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
        if any(char in value for char in '/\\:<>"|?*') or value.endswith((".", " ")) or windows_device:
            raise ValueError("Name cannot contain path separators, reserved filename characters, or Windows device names.")
        return value

    def save_current_config(self):
        if self.config_path is None:
            return self.save_config_as()
        save_config(self.config_path, self._training_config())
        self._refresh_configs(); self._append_log(f"Saved configuration: {self.config_path}")

    def save_config_as(self):
        name, ok = QInputDialog.getText(self, "Save configuration", "Configuration name")
        if not ok or not name.strip():
            return
        try:
            safe_name = self._safe_config_name(name)
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid configuration name", str(exc)); return
        path = Path(self.config.get("workspace", APP_DIR / "workspace")) / "configs" / f"{safe_name}.json"
        if path.exists() and QMessageBox.question(self, "Replace configuration", f"Replace {path.name}?") != QMessageBox.StandardButton.Yes:
            return
        save_config(path, self._training_config()); self.config_path = path; self._refresh_configs(); self._append_log(f"Saved configuration: {path}")

    def load_selected_config(self):
        path_text = self.config_combo.currentData()
        if not path_text:
            return
        try:
            self.config = load_config(path_text)
            self.config_path = Path(path_text)
            self._sync_config_controls()
            self._append_log(f"Loaded configuration: {path_text}")
        except Exception as exc:
            self._show_error("Could not load configuration", str(exc))

    def rename_config(self):
        old = self.config_combo.currentData()
        if not old:
            return
        new, ok = QInputDialog.getText(self, "Rename configuration", "New name", text=Path(old).stem)
        if ok and new.strip():
            try:
                safe_name = self._safe_config_name(new)
            except ValueError as exc:
                QMessageBox.warning(self, "Invalid configuration name", str(exc)); return
            target = Path(old).with_name(safe_name + ".json")
            if target.exists():
                QMessageBox.warning(self, "Name exists", "A configuration with that name already exists."); return
            Path(old).rename(target)
            if self.config_path and self.config_path == Path(old): self.config_path = target
            self._refresh_configs()

    def delete_config(self):
        old = self.config_combo.currentData()
        if old and QMessageBox.question(self, "Delete configuration", f"Delete {Path(old).name}?") == QMessageBox.StandardButton.Yes:
            Path(old).unlink(missing_ok=True); self._refresh_configs()
            if self.config_path == Path(old): self.config_path = None

    def set_default_config(self):
        source = self.config_combo.currentData()
        if source:
            self.config["default_training_config"] = source
            self.save_preferences()
            self._append_log(f"Default training configuration set to {source}")

    def import_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import configuration", str(PROJECT_ROOT), "JSON (*.json)")
        if not path: return
        name, ok = QInputDialog.getText(self, "Import configuration", "Save as", text=Path(path).stem)
        if ok and name.strip():
            try:
                safe_name = self._safe_config_name(name)
            except ValueError as exc:
                QMessageBox.warning(self, "Invalid configuration name", str(exc)); return
            target = Path(self.config.get("workspace", APP_DIR / "workspace")) / "configs" / f"{safe_name}.json"
            if target.exists():
                QMessageBox.warning(self, "Name exists", "Choose a new name to preserve existing configurations."); return
            save_config(target, load_config(path)); self._refresh_configs()

    def export_config(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export training configuration", str(PROJECT_ROOT / "training_config.json"), "JSON (*.json)")
        if path: save_config(Path(path), self._training_config())

    def _refresh_experiments(self):
        if not hasattr(self, "experiment_table"):
            return
        runs = Path(self.config.get("workspace", APP_DIR / "workspace")) / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        entries = sorted((p for p in runs.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
        for table in (self.experiment_table, self.dashboard_history):
            table.setRowCount(0)
            for run in entries:
                metadata_path = run / "run_metadata.json"
                try: meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                except Exception: meta = {}
                best = meta.get("best", str(run / "weights" / "best.pt"))
                last = meta.get("last", str(run / "weights" / "last.pt"))
                status = meta.get("status") or ("complete" if Path(best).is_file() else ("in progress" if not metadata_path.exists() else "stopped / failed"))
                summary = str(meta.get("metrics", {}))[:160]
                row = table.rowCount(); table.insertRow(row)
                if table is self.experiment_table:
                    values = [run.name, datetime.fromtimestamp(run.stat().st_mtime).strftime("%Y-%m-%d %H:%M"), status, best, last, summary]
                else:
                    values = [run.name, datetime.fromtimestamp(run.stat().st_mtime).strftime("%Y-%m-%d %H:%M"), best, status]
                for col, value in enumerate(values):
                    item = QTableWidgetItem(str(value)); item.setData(Qt.ItemDataRole.UserRole, str(run)); table.setItem(row, col, item)
        self._refresh_dashboard_metrics(entries)
        self._refresh_model_registry()

    def _model_library_workspace(self) -> Path:
        return Path(self.config.get("workspace", APP_DIR / "workspace")).expanduser()

    def _refresh_model_registry(self):
        if not hasattr(self, "model_registry_table"):
            return
        table = self.model_registry_table
        table.setRowCount(0)
        try:
            entries = list_models(self._model_library_workspace())
        except Exception as exc:
            self._append_log(f"ERROR · model registry could not be read: {exc}")
            return
        for entry in sorted(entries, key=lambda item: item.get("registered_at", ""), reverse=True):
            path = str(entry.get("path", ""))
            details = entry.get("details", {}) or {}
            status = "Available" if Path(path).is_file() else "Missing file"
            dataset_run = details.get("run_id") or details.get("dataset") or "—"
            metrics = json.dumps(details.get("metrics", {}), ensure_ascii=False, separators=(",", ":"))[:180]
            values = [entry.get("name", Path(path).stem), entry.get("format", ""), entry.get("task", "semantic"), status, dataset_run, metrics or "—", path]
            row = table.rowCount(); table.insertRow(row)
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setData(Qt.ItemDataRole.UserRole, path)
                table.setItem(row, col, item)

    def _selected_registered_model(self) -> str | None:
        row = self.model_registry_table.currentRow()
        item = self.model_registry_table.item(row, 0) if row >= 0 else None
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _register_model_artifact(self, model_path: str | Path, details: dict | None = None):
        entry = register_model(self._model_library_workspace(), model_path, details)
        self._refresh_model_registry()
        self._append_log(f"Registered semantic model: {entry['path']}")
        return entry

    def register_model_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Register semantic model", str(PROJECT_ROOT), "Models (*.pt *.onnx)")
        if not path:
            return
        try:
            self._register_model_artifact(path, {"source": "manual"})
        except Exception as exc:
            self._show_error("Could not register model", str(exc))

    def use_registered_model(self):
        path = self._selected_registered_model()
        if not path:
            QMessageBox.information(self, "Select a model", "Select an entry in the model library first.")
            return
        if not Path(path).is_file():
            QMessageBox.warning(self, "Model file missing", f"The registered file is unavailable:\n{path}")
            return
        self.active_model = path
        self.test_model.setText(path)
        self.eval_model.setText(path)
        if Path(path).suffix.lower() == ".onnx":
            self.onnx_path.setText(path)
        else:
            self.onnx_source.setText(path)
            self.checkpoint_edit.setText(path)
        self.stack.setCurrentIndex(6)

    def remove_registered_model(self):
        path = self._selected_registered_model()
        if not path:
            QMessageBox.information(self, "Select a model", "Select an entry in the model library first.")
            return
        if QMessageBox.question(self, "Remove model entry", f"Remove this entry from the library? The model file will be preserved.\n\n{path}") != QMessageBox.StandardButton.Yes:
            return
        remove_model(self._model_library_workspace(), path)
        self._refresh_model_registry()

    def _refresh_dashboard_runs(self):
        self._refresh_experiments()

    def _refresh_dashboard_metrics(self, runs):
        best_miou = "—"; test_miou = "—"
        for run in runs:
            for candidate in (run / "metrics.json", run / "run_metadata.json"):
                try:
                    value = json.loads(candidate.read_text(encoding="utf-8"))
                    metrics = value.get("results", value.get("metrics", {}))
                    for key, metric in metrics.items():
                        if "miou" in str(key).lower() and isinstance(metric, (int, float)):
                            best_miou = f"{metric:.3f}"; break
                except Exception:
                    pass
                if best_miou != "—": break
            if best_miou != "—": break
        self.dashboard_cards["best_miou"].value_label.setText(best_miou)
        self.dashboard_cards["test_miou"].value_label.setText(test_miou)
        if runs:
            self.dashboard_cards["latest"].value_label.setText(runs[0].name[-13:])
        else:
            self.dashboard_cards["latest"].value_label.setText("—")

    def _selected_experiment_dir(self) -> Path | None:
        row = self.experiment_table.currentRow()
        if row < 0:
            return None
        item = self.experiment_table.item(row, 0)
        return Path(item.data(Qt.ItemDataRole.UserRole)) if item else None

    def open_selected_experiment(self):
        path = self._selected_experiment_dir()
        if path: QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def use_experiment_best(self):
        path = self._selected_experiment_dir()
        if path:
            checkpoint = path / "weights" / "best.pt"
            if not checkpoint.is_file(): QMessageBox.information(self, "No best checkpoint", f"No best.pt found in {path}"); return
            self.checkpoint_edit.setText(str(checkpoint)); self._append_log(f"Selected new fine-tune initialization: {checkpoint}")
            self.stack.setCurrentIndex(3)

    def resume_experiment(self):
        path = self._selected_experiment_dir()
        if path:
            checkpoint = path / "weights" / "last.pt"
            if not checkpoint.is_file(): QMessageBox.information(self, "No last checkpoint", f"No resumable last.pt found in {path}"); return
            try:
                saved = load_config(path / "config.json")
                cfg = saved.get("config", self._training_config())
                saved_data_yaml = saved.get("data_yaml", "")
                if saved_data_yaml:
                    self.data_yaml_edit.setText(saved_data_yaml)
            except Exception:
                cfg = self._training_config()
            self.data_yaml = self.data_yaml_edit.text().strip()
            if not self.data_yaml or not Path(self.data_yaml).is_file():
                QMessageBox.warning(self, "Dataset unavailable", "The original prepared dataset YAML is missing. Restore the original path to resume, or select best.pt and start a new fine-tuning run.")
                return
            self.stack.setCurrentIndex(3)
            self._launch_training(cfg, str(checkpoint))

    def choose_eval_dataset(self):
        folder = QFileDialog.getExistingDirectory(self, "Prepared dataset", str(Path(self.config.get("workspace", APP_DIR)) / "prepared_datasets"))
        if folder:
            self.eval_dataset.setText(folder)

    def start_evaluation(self):
        model, dataset, split = self.eval_model.text().strip(), self.eval_dataset.text().strip(), self.eval_split.currentText()
        if not model or not dataset:
            QMessageBox.warning(self, "Evaluation inputs", "Select a model and a prepared dataset."); return
        if split == "test" and QMessageBox.question(self, "Independent test evaluation", "Run the final independent test evaluation? Avoid using the test split repeatedly to select checkpoints.") != QMessageBox.StandardButton.Yes:
            return
        ignore = 255 if self.ignore_checkbox.isChecked() else None
        self.eval_status.setText("Evaluating samples…")
        self._start_operation(f"{split.title()} evaluation", evaluate_dataset, model, dataset, split, self.test_imgsz.value(), "auto", ignore, on_done=self._evaluation_done)

    def _evaluation_done(self, result):
        self.latest_evaluation = result
        self.eval_status.setText(f"Evaluated {result['samples']} samples · split={result['split']} · report saved to {result['dataset']}\\evaluations")
        self.eval_metrics.setPlainText(json.dumps(result["metrics"], indent=2))
        self.eval_examples.setRowCount(0)
        for example in result["examples"]:
            row = self.eval_examples.rowCount(); self.eval_examples.insertRow(row)
            for col, value in enumerate((Path(example["image"]).name, example.get("town"), example.get("weather"), example.get("traffic"), example["prediction"])):
                self.eval_examples.setItem(row, col, QTableWidgetItem(str(value)))
        if result["split"] == "test":
            self.dashboard_cards["test_miou"].value_label.setText(f"{result['metrics']['miou']:.3f}")
        self._append_log(f"Evaluation metrics: {json.dumps(result['metrics'])}")

    def open_eval_example(self, row, column):
        item = self.eval_examples.item(row, 4)
        if item: QDesktopServices.openUrl(QUrl.fromLocalFile(item.text()))

    def choose_test_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select image", str(PROJECT_ROOT), "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)")
        if not path: return
        self._start_operation("Image inference", run_image_inference, self.test_model.text().strip(), path, str(Path(self.config.get("workspace", APP_DIR)) / "inference"), self.test_imgsz.value(), "auto", self.opacity.value(), on_done=self._image_inference_done)

    def _image_inference_done(self, result):
        self.test_status.setText(f"Classes: {result['classes_present']} · mask: {result['mask']} · overlay: {result['overlay']}")
        display_image(result["image"], self.test_previews["Original RGB"], QSize(420, 240))
        display_image(result["mask"], self.test_previews["Semantic class map"], QSize(420, 240))
        display_image(result["overlay"], self.test_preview, QSize(420, 240))
        display_image(result["drivable"], self.test_previews["Drivable area only"], QSize(420, 240))
        display_image(result["lane_lines"], self.test_previews["Lane lines only"], QSize(420, 240))
        display_image(result["vehicles"], self.test_previews["Vehicles only"], QSize(420, 240))

    def choose_test_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select image directory", str(PROJECT_ROOT))
        if folder:
            output = Path(self.config.get("workspace", APP_DIR)) / "inference" / f"folder_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            self._start_operation("Folder inference", run_folder_inference, self.test_model.text().strip(), folder, str(output), self.test_imgsz.value(), "auto", self.opacity.value(), on_done=lambda result: self.test_status.setText(f"Processed {result['processed']} images · {result['output_dir']}"))

    def choose_test_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select video", str(PROJECT_ROOT), "Video (*.mp4 *.avi *.mov *.mkv *.wmv)")
        if not path: return
        target = Path(self.config.get("workspace", APP_DIR)) / "inference" / f"{Path(path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_overlay.mp4"
        self._start_operation("Video inference", run_video_inference, self.test_model.text().strip(), path, str(target), self.test_imgsz.value(), "auto", self.opacity.value(), True, on_done=lambda r: self.test_status.setText(f"{r['frames']} frames · {r['processing_fps']:.2f} FPS · {r['mean_frame_ms']:.1f} ms/frame · {r['output']}"))

    def toggle_webcam(self):
        if self.webcam_worker and self.webcam_worker.isRunning():
            self.webcam_worker.request_stop()
            self.test_status.setText("Stopping webcam after the current frame…")
            return
        model = self.test_model.text().strip()
        if not model:
            QMessageBox.warning(self, "Model required", "Select a semantic .pt or .onnx model before opening the webcam.")
            return
        from .core.workers import WebcamWorker
        self.webcam_worker = WebcamWorker(model, self.test_imgsz.value(), "auto", self.opacity.value())
        self.webcam_worker.frame.connect(self._webcam_frame)
        self.webcam_worker.status.connect(self.test_status.setText)
        self.webcam_worker.failed.connect(lambda error: self._show_error("Webcam inference failed", error))
        self.webcam_worker.done.connect(lambda: self.test_status.setText("Webcam stopped."))
        self.webcam_worker.start()

    def _webcam_frame(self, rgb):
        height, width = rgb.shape[:2]
        qimage = QImage(rgb.data, width, height, int(rgb.strides[0]), QImage.Format.Format_RGB888).copy()
        self.test_preview.setPixmap(QPixmap.fromImage(qimage).scaled(self.test_preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def start_onnx_export(self):
        source = self.onnx_source.text().strip()
        if not source:
            QMessageBox.warning(self, "Checkpoint required", "Select a trained .pt semantic checkpoint first."); return
        if QMessageBox.question(self, "Export ONNX", f"Export and validate ONNX from:\n{source}\n\nImage size: {self.onnx_imgsz.value()} · Opset: {self.onnx_opset.value()}") != QMessageBox.StandardButton.Yes:
            return
        output_root = Path(self.config.get("workspace", APP_DIR)) / "exports"
        self._start_operation("ONNX export", export_onnx, source, self.onnx_imgsz.value(), self.onnx_opset.value(), self.onnx_dynamic.isChecked(), self.onnx_simplify.isChecked(), output_root, on_done=self._onnx_export_done)

    def _onnx_export_done(self, result):
        self.onnx_path.setText(result["path"])
        self.test_model.setText(result["path"])
        self.onnx_status.setText(f"Export validated · {result['onnx_checker']} · {result['size_bytes']:,} bytes · providers={result['providers']}")
        self.onnx_log.setPlainText(json.dumps(result, indent=2))
        try:
            self._register_model_artifact(result["path"], {"source": "onnx_export", "from_checkpoint": self.onnx_source.text().strip(), "imgsz": self.onnx_imgsz.value()})
        except Exception as exc:
            self._append_log(f"ERROR · model library registration failed: {exc}")

    def choose_onnx_for_compare(self):
        self._choose_file(self.onnx_path, "ONNX (*.onnx)")

    def start_parity(self):
        pt, onnx = self.onnx_source.text().strip(), self.onnx_path.text().strip()
        if not pt or not onnx:
            QMessageBox.warning(self, "Parity inputs", "Choose both the PyTorch checkpoint and exported ONNX model."); return
        image, _ = QFileDialog.getOpenFileName(self, "Select parity sample image", str(PROJECT_ROOT), "Images (*.png *.jpg *.jpeg)")
        if image:
            self._start_operation("PyTorch / ONNX comparison", compare_models, pt, onnx, image, self.onnx_imgsz.value(), "auto", on_done=lambda result: self.onnx_log.setPlainText(json.dumps(result, indent=2)))

    def start_benchmark(self):
        model = self.onnx_path.text().strip() or self.onnx_source.text().strip()
        if not model:
            QMessageBox.warning(self, "Model required", "Choose a PyTorch or ONNX model."); return
        image, _ = QFileDialog.getOpenFileName(self, "Select benchmark image", str(PROJECT_ROOT), "Images (*.png *.jpg *.jpeg)")
        if image:
            self._start_operation("Model benchmark", benchmark_model, model, image, self.onnx_imgsz.value(), "auto", on_done=self._benchmark_done)

    def _benchmark_done(self, result):
        self.latest_benchmark = result
        self.onnx_log.setPlainText(json.dumps(result, indent=2))
        workspace = Path(self.config.get("workspace", APP_DIR))
        from .core.runtime import atomic_json
        path = workspace / "benchmarks" / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        atomic_json(path, result)
        self._append_log(f"Benchmark report saved: {path}")

    def start_deployment(self):
        model = self.onnx_path.text().strip()
        if not model:
            QMessageBox.warning(self, "ONNX model required", "Export and validate a semantic ONNX model before creating a deployment package."); return
        output = QFileDialog.getExistingDirectory(self, "Select parent folder for new deployment package", str(Path(self.config.get("workspace", APP_DIR))))
        if not output: return
        target = Path(output) / f"deployment_{Path(model).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        evaluation = self.latest_evaluation if self.latest_evaluation and Path(self.latest_evaluation.get("model", "")).resolve() == Path(model).resolve() else None
        benchmark = self.latest_benchmark if self.latest_benchmark and Path(self.latest_benchmark.get("model", "")).resolve() == Path(model).resolve() else None
        self._start_operation("Deployment package", create_deployment_package, model, str(target), self._training_config(), evaluation, benchmark, on_done=lambda result: self.onnx_log.setPlainText(json.dumps(result, indent=2)))

    def choose_workspace(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Studio workspace", str(APP_DIR))
        if folder: self.settings_workspace.setText(folder)

    def _update_dashboard_environment(self):
        report = environment_report()
        self.dashboard_cards["gpu"].value_label.setText(report.get("gpu", "CPU"))
        self.dashboard_cards["model"].value_label.setText(f"YOLO26{self.model_size.currentText()}-sem")
        self.dashboard_runtime.setPlainText(json.dumps(report, indent=2))
        self.environment_text.setPlainText(json.dumps(report, indent=2))
        self._refresh_dashboard_runs()

    def _update_environment_text(self):
        self._update_dashboard_environment()

    def save_preferences(self):
        self.config = self._training_config()
        roots = [v for v in self.settings_roots.toPlainText().splitlines() if v.strip()]
        self.config["dataset_roots"] = roots
        self.config["workspace"] = self.settings_workspace.text().strip()
        save_config(self.settings_path, self.config)
        self.roots_edit.setText("\n".join(roots))
        self._append_log("Application preferences saved")

    def export_logs(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export logs", str(Path(self.config.get("workspace", APP_DIR)) / "studio.log"), "Text (*.txt)")
        if path: Path(path).write_text("\n".join(self._log_lines), encoding="utf-8")

    def closeEvent(self, event):
        if self.webcam_worker and self.webcam_worker.isRunning():
            self.webcam_worker.request_stop()
            self.webcam_worker.wait(3000)
            if self.webcam_worker.isRunning():
                QMessageBox.warning(self, "Webcam still active", "Wait for the webcam worker to release the camera before closing the Studio.")
                event.ignore(); return
        if self.training_worker and self.training_worker.isRunning():
            QMessageBox.warning(self, "Training active", "Request a safe stop and wait for the checkpoint boundary before closing the Studio.")
            event.ignore(); return
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Operation active", "Wait for the background operation to finish before closing the Studio.")
            event.ignore(); return
        try:
            self.save_preferences()
        except Exception:
            pass
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("RoadForge")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
