"""PySide6 desktop interface for collecting raw CARLA RGB frames."""
import sys
import cv2
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QPlainTextEdit, QScrollArea, QSpinBox, QVBoxLayout, QWidget, QTabWidget,
    QTreeWidget, QTreeWidgetItem, QSlider, QSplitter)

try:  # Supports both `python main.py` and `python -m datacollector.main`.
    from .config import CAMERA_REFERENCE, TOWNS
    from .worker import CarlaWorker
    from .teacher_labeling import (AnnotationWorker, CollectionScanWorker, YOLOPv2Predictor, create_session,
        visualization)
except ImportError:
    from config import CAMERA_REFERENCE, TOWNS
    from worker import CarlaWorker
    from teacher_labeling import (AnnotationWorker, CollectionScanWorker, YOLOPv2Predictor, create_session,
        visualization)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RoadForge · CARLA Dataset Collector")
        self.setMinimumSize(900, 600)
        self.resize(1320, 820)
        self.worker = None
        self.annotation_worker = None
        self.annotation_scan_worker = None
        self.connected = False
        self.town_rows = {}
        self._build()
        self._style()
        self._update_totals()

    def _build(self):
        # Let the window fit shorter displays and expose a vertical scrollbar
        # when the full dashboard cannot fit in the available screen height.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setMinimumSize(0, 0)
        base = QWidget()
        scroll.setWidget(base)
        tabs = QTabWidget()
        tabs.addTab(scroll, "Data Collection")
        self.setCentralWidget(tabs)
        outer = QVBoxLayout(base); outer.setContentsMargins(22,18,22,18); outer.setSpacing(14)
        header = QHBoxLayout()
        title = QLabel("Raw RGB Dataset Collector"); title.setObjectName("title")
        self.status = QLabel("Disconnected"); self.status.setObjectName("badge")
        header.addWidget(title); header.addStretch(); header.addWidget(self.status)
        outer.addLayout(header)

        top = QHBoxLayout(); outer.addLayout(top, 2)
        left = QVBoxLayout(); top.addLayout(left, 3)
        conn = QGroupBox("CARLA Connection"); form=QFormLayout(conn)
        self.host=QLineEdit("127.0.0.1"); self.port=self._spin(1,65535,2000)
        self.timeout=self._dspin(1,180,20,1)
        self.current_town=QLabel("—"); self.connect_btn=QPushButton("Connect"); self.connect_btn.clicked.connect(self.connect)
        hostrow=QHBoxLayout(); hostrow.addWidget(self.host,3); hostrow.addWidget(QLabel("Port")); hostrow.addWidget(self.port,1)
        form.addRow("Host",hostrow); form.addRow("Timeout (s)",self.timeout); form.addRow("Current town",self.current_town); form.addRow(self.connect_btn)
        left.addWidget(conn)

        towns=QGroupBox("Town targets"); tv=QVBoxLayout(towns)
        actions=QHBoxLayout(); self.select_all=QPushButton("Select all"); self.select_none=QPushButton("Select none")
        self.select_all.clicked.connect(lambda:self._set_towns(True)); self.select_none.clicked.connect(lambda:self._set_towns(False))
        actions.addWidget(self.select_all); actions.addWidget(self.select_none); actions.addStretch()
        self.total_label=QLabel("0 / 0 images")
        actions.addWidget(self.total_label); tv.addLayout(actions)
        for town in TOWNS:
            row=QGridLayout(); enabled=QCheckBox(town); enabled.setChecked(True)
            target=self._spin(0,1000000,700); count=QLabel("0 / 700 · Pending"); bar=QProgressBar(); bar.setRange(0,100); bar.setValue(0)
            enabled.stateChanged.connect(self._update_totals); target.valueChanged.connect(self._update_totals)
            row.addWidget(enabled,0,0); row.addWidget(QLabel("Target"),0,1); row.addWidget(target,0,2); row.addWidget(count,0,3); row.addWidget(bar,1,0,1,4)
            tv.layout().addLayout(row); self.town_rows[town]={"enabled":enabled,"target":target,"count":count,"bar":bar,"status":"Pending"}
        left.addWidget(towns,1)

        right=QVBoxLayout(); top.addLayout(right,2)
        camera=QGroupBox("Main RGB camera · sample project reference"); cf=QGridLayout(camera)
        self.camera_fields={}
        specs=[("width","Width",64,4096,1),("height","Height",64,2160,1),("fov","FOV",1,179,.5),
               ("x","X",-20,20,.1),("y","Y",-20,20,.1),("z","Z",-10,20,.1),
               ("pitch","Pitch",-180,180,.5),("yaw","Yaw",-180,180,.5),("roll","Roll",-180,180,.5)]
        for i,(key,label,lo,hi,step) in enumerate(specs):
            lab=QLabel(label); field=self._dspin(lo,hi,CAMERA_REFERENCE[key],step)
            if key in ("width","height"): field.setDecimals(0)
            self.camera_fields[key]=field; cf.addWidget(lab,i//3*2,i%3*2); cf.addWidget(field,i//3*2,i%3*2+1)
            field.valueChanged.connect(self._check_camera)
        self.reference_label=QLabel("Matches reference configuration"); self.reference_label.setObjectName("muted")
        cf.addWidget(self.reference_label,6,0,1,6); right.addWidget(camera)

        settings=QGroupBox("Collection settings"); sf=QFormLayout(settings)
        self.interval=self._dspin(.05,60,1,0.05); self.images_per_spawn=self._spin(1,10000,30)
        self.vehicle=QLineEdit("vehicle.tesla.model3"); self.speed=self._dspin(-100,100,10,1)
        self.traffic=QCheckBox("Enable background vehicles"); self.traffic_count=self._spin(0,500,20)
        traffic_row=QHBoxLayout(); traffic_row.addWidget(self.traffic); traffic_row.addWidget(QLabel("Count")); traffic_row.addWidget(self.traffic_count)
        self.weather=QComboBox(); self.weather.addItems(["Default","ClearNoon","CloudyNoon","WetNoon","WetCloudyNoon","SoftRainNoon","HardRainNoon"])
        self.output=QLineEdit(str(Path(__file__).parent/"datasets")); choose=QPushButton("Browse…"); choose.clicked.connect(self._browse)
        outrow=QHBoxLayout(); outrow.addWidget(self.output); outrow.addWidget(choose)
        sf.addRow("Capture interval (simulation seconds)",self.interval); sf.addRow("Images per spawn session",self.images_per_spawn)
        sf.addRow("Ego blueprint",self.vehicle); sf.addRow("TM speed difference (%)",self.speed); sf.addRow("Traffic",traffic_row); sf.addRow("Weather",self.weather); sf.addRow("Output root or resume folder",outrow)
        right.addWidget(settings)

        control=QHBoxLayout(); self.start_btn=QPushButton("Start collection"); self.pause_btn=QPushButton("Pause"); self.resume_btn=QPushButton("Resume"); self.stop_btn=QPushButton("Stop")
        self.start_btn.setObjectName("primary"); self.start_btn.clicked.connect(self.start_collection); self.pause_btn.clicked.connect(self.pause); self.resume_btn.clicked.connect(self.resume); self.stop_btn.clicked.connect(self.stop)
        for button in (self.start_btn,self.pause_btn,self.resume_btn,self.stop_btn): control.addWidget(button)
        right.addLayout(control)

        dashboard=QGroupBox("Live collection dashboard"); dv=QGridLayout(dashboard)
        self.overall=QLabel("0 / 4,200 · 0%"); self.overall_bar=QProgressBar(); self.overall_bar.setRange(0,1000)
        self.active=QLabel("Town: — · Spawn: — · Ego: idle"); self.metrics=QLabel("Simulation: —   Captures: 0   Failed: 0   Rate: 0.00/s   Elapsed: 00:00:00")
        dv.addWidget(self.overall,0,0); dv.addWidget(self.overall_bar,0,1); dv.addWidget(self.active,1,0,1,2); dv.addWidget(self.metrics,2,0,1,2)
        right.addWidget(dashboard)

        lower=QHBoxLayout(); outer.addLayout(lower,1)
        prev=QGroupBox("Latest RGB frame (display preview)"); pv=QVBoxLayout(prev); self.preview_label=QLabel("Waiting for first capture"); self.preview_label.setAlignment(Qt.AlignCenter); self.preview_label.setMinimumSize(480,260); self.preview_label.setObjectName("preview")
        pv.addWidget(self.preview_label); lower.addWidget(prev,1)
        logbox=QGroupBox("Events and errors"); lv=QVBoxLayout(logbox); self.logs=QPlainTextEdit(); self.logs.setReadOnly(True); lv.addWidget(self.logs); lower.addWidget(logbox,1)
        self.pause_btn.setEnabled(False); self.resume_btn.setEnabled(False); self.stop_btn.setEnabled(False)
        self._build_labeling_tab(tabs)

    def _build_labeling_tab(self, tabs):
        page_scroll = QScrollArea(); page_scroll.setWidgetResizable(True); page_scroll.setFrameShape(QFrame.NoFrame)
        page = QWidget(); page_scroll.setWidget(page)
        layout = QVBoxLayout(page); layout.setContentsMargins(18, 16, 18, 16); layout.setSpacing(12)
        top = QHBoxLayout(); title = QLabel("YOLOPv2 Teacher Labeling"); title.setObjectName("title")
        self.label_status = QLabel("Ready"); self.label_status.setObjectName("badge")
        top.addWidget(title); top.addStretch(); top.addWidget(self.label_status); layout.addLayout(top)

        model_box = QGroupBox("Model and inference"); mf = QGridLayout(model_box)
        self.model_path = QLineEdit(str(Path(__file__).resolve().parent.parent / "YOLOPv2.onnx"))
        model_browse = QPushButton("Select ONNX…"); model_browse.clicked.connect(self._choose_model)
        self.model_info = QLabel("Checking model…"); self.provider_combo = QComboBox(); self.provider_combo.addItems(["Auto", "CUDA", "CPU"])
        self.provider_info = QLabel("Provider will be shown after session initialization")
        self.save_vis = QCheckBox("Save visualizations"); self.save_vis.setChecked(True)
        mf.addWidget(QLabel("Model path"), 0, 0); mf.addWidget(self.model_path, 0, 1); mf.addWidget(model_browse, 0, 2)
        mf.addWidget(self.model_info, 1, 0, 1, 2); mf.addWidget(QLabel("Execution provider"), 1, 2); mf.addWidget(self.provider_combo, 1, 3)
        mf.addWidget(self.provider_info, 2, 0, 1, 3); mf.addWidget(self.save_vis, 2, 3); layout.addWidget(model_box)

        split = QSplitter(Qt.Horizontal)
        selection = QGroupBox("Collections and towns · all selected by default"); sv = QVBoxLayout(selection)
        actions = QHBoxLayout()
        for text, checked in (("Select all", True), ("Deselect all", False)):
            button = QPushButton(text); button.clicked.connect(lambda _, state=checked: self._set_collection_checks(state)); actions.addWidget(button)
        self.rescan_btn = QPushButton("Rescan Collections"); self.rescan_btn.clicked.connect(self._rescan_annotations); actions.addWidget(self.rescan_btn); sv.addLayout(actions)
        self.collection_tree = QTreeWidget(); self.collection_tree.setHeaderLabels(["Collection / town", "Images", "Done", "Remaining", "Failed", "Progress"])
        self.collection_tree.itemChanged.connect(self._annotation_selection_changed); sv.addWidget(self.collection_tree)
        split.addWidget(selection)

        preview_box = QGroupBox("Preview · same inference pipeline as batch"); pv = QVBoxLayout(preview_box)
        pr = QHBoxLayout(); self.preview_btn = QPushButton("Preview Selected Image…"); self.preview_btn.clicked.connect(self._preview_file)
        self.preview_view = QComboBox(); self.preview_view.addItems(["Original RGB", "Drivable Area", "Lane Lines", "Combined classes", "Combined overlay"])
        self.preview_view.currentTextChanged.connect(self._refresh_annotation_preview)
        pr.addWidget(self.preview_btn); pr.addWidget(self.preview_view); pv.addLayout(pr)
        self.annotation_preview = QLabel("Select an image to preview"); self.annotation_preview.setAlignment(Qt.AlignCenter); self.annotation_preview.setMinimumSize(400, 300); self.annotation_preview.setObjectName("preview"); pv.addWidget(self.annotation_preview, 1)
        self.preview_dims = QLabel("Source and mask dimensions: —"); pv.addWidget(self.preview_dims)
        self.preview_legend = QLabel("Legend: Background · Green: Drivable Area · Yellow: Lane Lines"); pv.addWidget(self.preview_legend)
        split.addWidget(preview_box); split.setSizes([520, 650]); layout.addWidget(split, 1)

        dash = QGroupBox("Annotation progress"); dv = QGridLayout(dash)
        self.annotation_counts = QLabel("Images: 0 · Annotated: 0 · Remaining: 0 · Skipped: 0 · Failed: 0")
        self.annotation_bar = QProgressBar(); self.annotation_bar.setRange(0, 1000)
        self.annotation_current = QLabel("Current image: —")
        dv.addWidget(self.annotation_counts, 0, 0, 1, 2); dv.addWidget(self.annotation_bar, 1, 0, 1, 2); dv.addWidget(self.annotation_current, 2, 0, 1, 2)
        layout.addWidget(dash)
        controls = QHBoxLayout()
        self.annotate_start = QPushButton("Start Annotation"); self.annotate_start.setObjectName("primary"); self.annotate_start.clicked.connect(self._start_annotation)
        self.annotate_pause = QPushButton("Pause"); self.annotate_pause.clicked.connect(self._pause_annotation)
        self.annotate_resume = QPushButton("Resume"); self.annotate_resume.clicked.connect(self._resume_annotation)
        self.annotate_stop = QPushButton("Stop"); self.annotate_stop.clicked.connect(self._stop_annotation)
        self.reprocess = QCheckBox("Reprocess existing images (overwrite)")
        for button in (self.annotate_start, self.annotate_pause, self.annotate_resume, self.annotate_stop): controls.addWidget(button)
        controls.addWidget(self.reprocess); layout.addLayout(controls)
        logbox = QGroupBox("Teacher labeling log"); lv = QVBoxLayout(logbox); self.annotation_logs = QPlainTextEdit(); self.annotation_logs.setReadOnly(True); lv.addWidget(self.annotation_logs); layout.addWidget(logbox)
        self.annotate_pause.setEnabled(False); self.annotate_resume.setEnabled(False); self.annotate_stop.setEnabled(False)
        self._annotation_previews = None
        tabs.addTab(page_scroll, "YOLOPv2 Teacher Labeling")
        self._rescan_annotations()

    def _choose_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose YOLOPv2 ONNX model", str(Path(self.model_path.text()).parent), "ONNX models (*.onnx)")
        if path:
            self.model_path.setText(path)
            self._update_model_availability()

    def _update_model_availability(self):
        path = Path(self.model_path.text()).expanduser()
        self.model_info.setText(f"Available · {path}" if path.is_file() else f"Model not found · {path}")
        self.model_info.setStyleSheet("color:#86efac" if path.is_file() else "color:#fca5a5")

    def _rescan_annotations(self):
        if self.annotation_scan_worker and self.annotation_scan_worker.isRunning():
            return
        self._update_model_availability()
        root = Path(__file__).resolve().parent / "datasets"
        worker = CollectionScanWorker(root, self)
        self.annotation_scan_worker = worker
        worker.result.connect(self._annotations_scanned)
        worker.progress.connect(self._annotation_scan_progress)
        worker.finished.connect(lambda current=worker: self._annotation_scan_finished(current))
        self.rescan_btn.setEnabled(False)
        self.annotate_start.setEnabled(False)
        self.label_status.setText("Scanning collections…")
        self.annotation_logs.appendPlainText("Scanning collections and checking existing labels…")
        worker.start()

    def _annotation_scan_progress(self, message):
        self.label_status.setText(message[:48])

    def _annotation_scan_finished(self, worker):
        if self.annotation_scan_worker is worker:
            self.annotation_scan_worker = None

    def _finish_annotation_scan_ui(self):
        self.rescan_btn.setEnabled(True)
        annotation_running = self.annotation_worker and self.annotation_worker.isRunning()
        self.annotate_start.setEnabled(not annotation_running)

    def _annotations_scanned(self, payload):
        self._finish_annotation_scan_ui()
        if payload.get("error"):
            self.label_status.setText("Scan failed")
            self.annotation_logs.appendPlainText(f"Collection scan failed:\n{payload['error']}")
            return

        self._collections = payload.get("collections", {})
        summaries = payload.get("summaries", {})
        self.collection_tree.blockSignals(True); self.collection_tree.clear()
        for collection, towns in self._collections.items():
            collection_total = sum(map(len, towns.values()))
            collection_done = collection_failed = 0
            root_item = QTreeWidgetItem(self.collection_tree, [collection.name, str(collection_total), "0", str(collection_total), "0"])
            root_bar = QProgressBar(); root_bar.setRange(0, max(1, collection_total)); root_bar.setValue(0)
            self.collection_tree.setItemWidget(root_item, 5, root_bar)
            root_item.setFlags(root_item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate)
            root_item.setCheckState(0, Qt.Checked)
            for town, images in towns.items():
                town_summary = summaries.get(collection, {}).get(town, {})
                done = town_summary.get("done", 0)
                failed = town_summary.get("failed", 0)
                collection_done += done; collection_failed += failed
                town_item = QTreeWidgetItem(root_item, [town, str(len(images)), str(done), str(len(images)-done), str(failed)])
                town_bar = QProgressBar(); town_bar.setRange(0, max(1, len(images))); town_bar.setValue(done)
                self.collection_tree.setItemWidget(town_item, 5, town_bar)
                town_item.setFlags(town_item.flags() | Qt.ItemIsUserCheckable)
                town_item.setCheckState(0, Qt.Checked)
                town_item.setData(0, Qt.UserRole, (str(collection), town))
            root_item.setText(2, str(collection_done)); root_item.setText(3, str(collection_total-collection_done)); root_item.setText(4, str(collection_failed))
            root_bar.setValue(collection_done)
            root_item.setData(0, Qt.UserRole, (str(collection), None))
        self.collection_tree.blockSignals(False)
        self._refresh_annotation_totals()
        if payload.get("stopped"):
            self.label_status.setText("Scan stopped")
        else:
            self.label_status.setText("Ready")
            self.annotation_logs.appendPlainText(f"Scan complete · {len(self._collections)} collection(s) found.")

    def _set_collection_checks(self, state):
        self.collection_tree.blockSignals(True)
        for i in range(self.collection_tree.topLevelItemCount()):
            root = self.collection_tree.topLevelItem(i); root.setCheckState(0, Qt.Checked if state else Qt.Unchecked)
            for j in range(root.childCount()): root.child(j).setCheckState(0, Qt.Checked if state else Qt.Unchecked)
        self.collection_tree.blockSignals(False); self._refresh_annotation_totals()

    def _selected_annotation_jobs(self):
        jobs = []
        for collection, towns in self._collections.items():
            for town, images in towns.items():
                selected = False
                for i in range(self.collection_tree.topLevelItemCount()):
                    root = self.collection_tree.topLevelItem(i)
                    if Path(root.data(0, Qt.UserRole)[0]) == collection:
                        for j in range(root.childCount()):
                            child = root.child(j)
                            if child.text(0) == town:
                                selected = child.checkState(0) == Qt.Checked; break
                if selected: jobs.extend((collection, town, image) for image in images)
        return jobs

    def _annotation_selection_changed(self, *_):
        self._refresh_annotation_totals()

    def _refresh_annotation_totals(self):
        if not hasattr(self, "collection_tree"): return
        total = 0
        for i in range(self.collection_tree.topLevelItemCount()):
            root = self.collection_tree.topLevelItem(i)
            for j in range(root.childCount()):
                child = root.child(j)
                if child.checkState(0) == Qt.Checked: total += int(child.text(1))
        self.annotation_counts.setText(f"Selected images: {total:,} · Annotated/remaining/failed counts are shown per town")
        self.annotation_bar.setValue(0)

    def _preview_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose a raw RGB image", str(Path(__file__).resolve().parent / "datasets"), "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)")
        if not path: return
        try:
            model = Path(self.model_path.text()).expanduser()
            session, provider, size, available, fallback_message = create_session(model, self.provider_combo.currentText())
            predictor = YOLOPv2Predictor(session, size)
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if bgr is None: raise ValueError("Image could not be decoded")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); drive, lane, combined = predictor.predict(rgb)
            self._annotation_previews = visualization(rgb, drive, lane, combined)
            self.preview_dims.setText(f"Original: {rgb.shape[1]}×{rgb.shape[0]} · Drivable/Lane/Combined: {drive.shape[1]}×{drive.shape[0]} · Provider: {provider}")
            self._refresh_annotation_preview(); self.annotation_logs.appendPlainText(f"Previewed {path}; provider {provider}, input {size[0]}×{size[1]}.")
            output_text = "; ".join(f"{o.name} {o.shape}" for o in session.get_outputs())
            self.provider_info.setText(f"Available: {available} · preview provider: {provider} · input: {size[0]}×{size[1]} · outputs: {output_text}")
            if fallback_message: self.annotation_logs.appendPlainText(fallback_message)
        except Exception as exc:
            QMessageBox.critical(self, "Preview failed", str(exc)); self.annotation_logs.appendPlainText(f"Preview failed: {exc}")

    def _refresh_annotation_preview(self, *_):
        if not self._annotation_previews: return
        rgb = self._annotation_previews.get(self.preview_view.currentText())
        if rgb is None: return
        h, w = rgb.shape[:2]; q = QImage(rgb.data, w, h, w*3, QImage.Format_RGB888).copy()
        self.annotation_preview.setPixmap(QPixmap.fromImage(q).scaled(self.annotation_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _start_annotation(self):
        jobs = self._selected_annotation_jobs()
        model = Path(self.model_path.text()).expanduser()
        if not model.is_file(): QMessageBox.warning(self, "Model missing", f"ONNX model not found: {model}"); return
        if not jobs: QMessageBox.warning(self, "No images selected", "Select at least one collection town."); return
        if self.reprocess.isChecked():
            answer = QMessageBox.warning(self, "Overwrite existing annotations?", "Reprocessing will overwrite existing masks for selected images. Continue?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes: return
        self.annotation_worker = AnnotationWorker(jobs, model, self.provider_combo.currentText(), self.save_vis.isChecked(), self.reprocess.isChecked())
        w = self.annotation_worker
        w.log.connect(self.annotation_logs.appendPlainText); w.progress.connect(self._annotation_progress)
        w.preview.connect(self._annotation_batch_preview); w.done.connect(self._annotation_done)
        self.annotate_start.setEnabled(False); self.annotate_pause.setEnabled(True); self.annotate_resume.setEnabled(False); self.annotate_stop.setEnabled(True)
        self.rescan_btn.setEnabled(False)
        self.label_status.setText("Starting"); w.start()

    def _pause_annotation(self):
        if self.annotation_worker: self.annotation_worker.pause(); self.annotate_pause.setEnabled(False); self.annotate_resume.setEnabled(True); self.label_status.setText("Paused"); self.annotation_logs.appendPlainText("Paused.")

    def _resume_annotation(self):
        if self.annotation_worker: self.annotation_worker.resume(); self.annotate_pause.setEnabled(True); self.annotate_resume.setEnabled(False); self.label_status.setText("Running"); self.annotation_logs.appendPlainText("Resumed.")

    def _stop_annotation(self):
        if self.annotation_worker: self.annotation_worker.stop(); self.annotate_stop.setEnabled(False); self.label_status.setText("Stopping…")

    def _annotation_progress(self, data):
        total = data.get("total", 0); done = data.get("success", 0); skip = data.get("skipped", 0); fail = data.get("failed", 0); rem = data.get("remaining", 0)
        self.annotation_counts.setText(f"Total: {total:,} · Successful: {done:,} · Remaining: {rem:,} · Skipped: {skip:,} · Failed: {fail:,}")
        self.annotation_bar.setValue(round(1000*(done+skip+fail)/total) if total else 0)
        scopes = data.get("scopes", {})
        for i in range(self.collection_tree.topLevelItemCount()):
            parent = self.collection_tree.topLevelItem(i)
            aggregate = {"total": 0, "success": 0, "failed": 0}
            found_scope = False
            for j in range(parent.childCount()):
                child = parent.child(j); scope = child.data(0, Qt.UserRole)
                key = f"{Path(scope[0]).name}|{scope[1]}" if scope else ""
                counts = scopes.get(key)
                if not counts: continue
                found_scope = True
                child.setText(1, str(counts["total"])); child.setText(2, str(counts["success"]))
                child.setText(3, str(max(0, counts["total"]-counts["success"])))
                child.setText(4, str(counts["failed"]))
                bar = self.collection_tree.itemWidget(child, 5)
                if bar: bar.setRange(0, max(1, counts["total"])); bar.setValue(counts["success"])
                for name in aggregate: aggregate[name] += counts[name]
            if not found_scope: continue
            parent.setText(1, str(aggregate["total"])); parent.setText(2, str(aggregate["success"]))
            parent.setText(3, str(max(0, aggregate["total"]-aggregate["success"])))
            parent.setText(4, str(aggregate["failed"]))
            parent_bar = self.collection_tree.itemWidget(parent, 5)
            if parent_bar: parent_bar.setRange(0, max(1, aggregate["total"])); parent_bar.setValue(aggregate["success"])
        rate = data.get('rate', 0); eta = rem / rate if rate > 0 else None
        eta_text = f" · ETA: {eta:.0f}s" if eta is not None and rem else ""
        self.annotation_current.setText(f"Current: {data.get('current') or '—'} · Provider: {data.get('provider','—')} · Rate: {rate:.2f}/s · Elapsed: {data.get('elapsed',0):.1f}s{eta_text}")
        self.provider_info.setText(f"Active provider: {data.get('provider','—')}")
        if data.get("model_info"): self.model_info.setText(data["model_info"])

    def _annotation_batch_preview(self, payload):
        self._annotation_previews = visualization(payload["rgb"], payload["drive"], payload["lane"], payload["combined"])
        self.preview_dims.setText(f"Original and masks: {payload['dims'][0]}×{payload['dims'][1]} · {payload['path']}")
        self._refresh_annotation_preview()

    def _annotation_done(self, stats):
        self.label_status.setText("Stopped" if stats.get("stopped") else ("Error" if stats.get("error") else "Complete"))
        if stats.get("stopped"):
            self.annotation_logs.appendPlainText("Annotation stopped.")
        elif stats.get("error"):
            self.annotation_logs.appendPlainText(f"Annotation failed: {stats['error']}")
        else:
            self.annotation_logs.appendPlainText(f"Annotation finished. Successful: {stats.get('success',0)}, skipped: {stats.get('skipped',0)}, failed: {stats.get('failed',0)}.")
        self.annotate_start.setEnabled(True); self.annotate_pause.setEnabled(False); self.annotate_resume.setEnabled(False); self.annotate_stop.setEnabled(False)
        self._rescan_annotations()

    @staticmethod
    def _spin(lo,hi,value):
        w=QSpinBox(); w.setRange(lo,hi); w.setValue(value); return w
    @staticmethod
    def _dspin(lo,hi,value,step):
        w=QDoubleSpinBox(); w.setRange(lo,hi); w.setValue(value); w.setSingleStep(step); w.setDecimals(2); return w

    def _style(self):
        self.setStyleSheet("""
            QMainWindow,QWidget{background:#111827;color:#e5e7eb;font-size:13px} QGroupBox{border:1px solid #334155;border-radius:10px;margin-top:10px;padding:12px;font-weight:600} QGroupBox::title{subcontrol-origin:margin;left:12px;padding:0 5px;color:#93c5fd} QLineEdit,QSpinBox,QDoubleSpinBox,QComboBox,QPlainTextEdit{background:#0b1220;border:1px solid #374151;border-radius:6px;padding:7px;color:#f8fafc} QPushButton{background:#263449;border:1px solid #475569;border-radius:7px;padding:9px 14px} QPushButton:hover{background:#334a67} QPushButton:disabled{color:#64748b;background:#1e293b} QPushButton#primary{background:#2563eb;border-color:#3b82f6;font-weight:bold} QLabel#title{font-size:23px;font-weight:700;color:#f8fafc} QLabel#badge{background:#1e3a5f;color:#bfdbfe;padding:7px 12px;border-radius:12px} QLabel#muted{color:#86efac} QLabel#preview{background:#080d16;border:1px dashed #475569;border-radius:8px;color:#94a3b8} QProgressBar{background:#0b1220;border:0;border-radius:5px;height:12px;text-align:center} QProgressBar::chunk{background:#22c55e;border-radius:5px}
        """)

    def _config(self):
        camera={key:int(field.value()) if key in ("width","height") else field.value() for key,field in self.camera_fields.items()}
        camera["sensor_tick"]=CAMERA_REFERENCE["sensor_tick"]
        return {"host":self.host.text().strip(),"port":self.port.value(),"timeout":self.timeout.value(),
            "towns":{t:{"enabled":r["enabled"].isChecked(),"target":r["target"].value()} for t,r in self.town_rows.items()},
            "capture_interval":self.interval.value(),"images_per_spawn":self.images_per_spawn.value(),"vehicle_blueprint":self.vehicle.text().strip(),
            "tm_speed_difference":self.speed.value(),"background_traffic":self.traffic.isChecked(),"background_vehicle_count":self.traffic_count.value(),
            "weather":self.weather.currentText(),"output_directory":self.output.text().strip(),"camera":camera}

    def _check_camera(self):
        is_ref=all(abs(self.camera_fields[k].value()-v)<1e-6 for k,v in CAMERA_REFERENCE.items() if k in self.camera_fields)
        self.reference_label.setText("Matches reference configuration" if is_ref else "⚠ Camera differs from CARLA_Autonomous_driving reference")
        self.reference_label.setStyleSheet("color:#86efac" if is_ref else "color:#fbbf24")

    def _update_totals(self):
        total=sum(row["target"].value() for row in self.town_rows.values() if row["enabled"].isChecked())
        self.total_label.setText(f"Selected target: {total:,}")

    def _set_towns(self,value):
        for row in self.town_rows.values(): row["enabled"].setChecked(value)
    def _browse(self):
        path=QFileDialog.getExistingDirectory(self,"Choose output root or collection folder",self.output.text())
        if path:self.output.setText(path)

    def _start_worker(self,connect_only=False):
        if self.worker and self.worker.isRunning(): return
        self.worker=CarlaWorker(self._config(),connect_only)
        self.worker.status.connect(self._status); self.worker.log.connect(self._log); self.worker.progress.connect(self._progress); self.worker.preview.connect(self._preview)
        self.worker.finished_result.connect(self._finished)
        self.worker.start()

    def connect(self):
        if self.connected:
            if self.worker and self.worker.isRunning():
                self.worker.stop()
            self.connected = False
            self.connect_btn.setText("Connect")
            self._status("Disconnected")
            return
        self._start_worker(True)
        self.connect_btn.setEnabled(False)
    def start_collection(self):
        config=self._config()
        if not any(v["enabled"] and v["target"]>0 for v in config["towns"].values()):
            QMessageBox.warning(self,"No towns selected","Select at least one town with a positive target."); return
        QMessageBox.information(self,"World loading","Collection loads each selected town. CARLA's currently loaded world will be replaced as town changes.")
        self.start_btn.setEnabled(False); self.pause_btn.setEnabled(True); self.stop_btn.setEnabled(True); self.status.setText("Starting")
        self._start_worker(False)
    def pause(self):
        if self.worker:self.worker.pause(); self.pause_btn.setEnabled(False); self.resume_btn.setEnabled(True)
    def resume(self):
        if self.worker:self.worker.resume(); self.pause_btn.setEnabled(True); self.resume_btn.setEnabled(False)
    def stop(self):
        if self.worker:self.worker.stop(); self.stop_btn.setEnabled(False); self.status.setText("Stopping…")
    def _status(self,text):
        self.status.setText(text[:48]); self.logs.appendPlainText(text)
        if text.startswith("Connected ·"): self.current_town.setText(text.split(":",1)[-1].strip())
        elif text.startswith("Loading "): self.current_town.setText(text.split("·",1)[0].replace("Loading", "").strip())
    def _log(self,text): self.logs.appendPlainText(text)
    def _progress(self,data):
        counts=data["counts"]; targets=data["targets"]
        total_target=sum(targets.values()); total=sum(counts.values())
        percent=total/total_target if total_target else 0
        self.overall.setText(f"{total:,} / {total_target:,} · {percent*100:.1f}%")
        self.overall_bar.setValue(round(percent*1000))
        for town,row in self.town_rows.items():
            count=counts.get(town,0); target=targets.get(town,row["target"].value())
            state="Completed" if target and count>=target else ("In Progress" if town==data["town"] else ("Pending" if count==0 else "In Progress"))
            row["count"].setText(f"{count:,} / {target:,} · {state}"); row["bar"].setValue(round(count/target*100) if target else 0)
        self.active.setText(f"Town: {data['town']} · Spawn: {data['spawn']} · Ego: autopilot")
        elapsed=int(data["elapsed"]); h,rem=divmod(elapsed,3600); m,s=divmod(rem,60)
        self.metrics.setText(f"Simulation: {data['sim_time']:.2f}s   Captures: {data['captures']}   Failed: {data['failed']}   Rate: {data['rate']:.2f}/s   Elapsed: {h:02}:{m:02}:{s:02}")
    def _preview(self,rgb):
        h,w,_=rgb.shape; q=QImage(rgb.data,w,h,3*w,QImage.Format_RGB888).copy()
        self.preview_label.setPixmap(QPixmap.fromImage(q).scaled(self.preview_label.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation))
    def _finished(self,result):
        self._status(result)
        if result == "Connected":
            self.connected = True
            self.connect_btn.setText("Disconnect")
        self.start_btn.setEnabled(True); self.pause_btn.setEnabled(False); self.resume_btn.setEnabled(False); self.stop_btn.setEnabled(False); self.connect_btn.setEnabled(True)

    def closeEvent(self,event):
        if self.annotation_scan_worker and self.annotation_scan_worker.isRunning():
            self.annotation_scan_worker.stop()
            if not self.annotation_scan_worker.wait(5000):
                event.ignore()
                self.annotation_logs.appendPlainText("Waiting for collection scan to stop before closing.")
                return
        if self.worker and self.worker.isRunning():
            self.worker.stop(); self.worker.wait(5000)
        event.accept()


def main():
    app=QApplication(sys.argv); app.setStyle("Fusion")
    window=MainWindow(); window.show(); return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
