import time

import pytest
pytest.importorskip("PySide6")

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication

from roadforge.classification import ClassificationPage
from roadforge.classification_data import ensure_dataset, validate_tasks
from roadforge.core.workers import OperationWorker
from roadforge.main import MainWindow


def test_main_window_loads_all_sections_and_controls(tmp_path, monkeypatch):
    import roadforge.main as app_module
    monkeypatch.setattr(app_module, "APP_DIR", tmp_path / "studio")
    monkeypatch.setattr(app_module, "PROJECT_ROOT", tmp_path)
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    assert window.stack.count() == 11
    assert window.PAGES[0] == "Dashboard"
    assert window.PAGES[9] == "Classification Lab"
    assert hasattr(window, "classification_page")
    assert window.model_name.text() == "YOLO26l-sem · semantic"
    assert all(edit.isReadOnly() for edit in window.class_names)
    assert [edit.text() for edit in window.class_names] == ["background", "ego_drivable_area", "lane_lines", "vehicles"]
    assert hasattr(window, "traffic_filter")
    assert hasattr(window, "downsample_enabled")
    assert window.downsample_count.value() == 7000
    assert not window.downsample_enabled.isChecked()
    assert window.balance_towns.isChecked()
    assert window.model_registry_table.columnCount() == 7
    window.model_size.setCurrentText("n")
    assert window.model_name.text() == "YOLO26n-sem · semantic"
    window.close()
    app.processEvents()


def test_worker_runs_off_main_thread_and_delivers_errors():
    app = QApplication.instance() or QApplication([])
    worker = OperationWorker(lambda: (time.sleep(0.2), "ok")[1])
    done = QSignalSpy(worker.done)
    worker.start()
    assert worker.isRunning()
    heartbeat = []
    QTimer.singleShot(20, lambda: heartbeat.append(True))
    assert done.wait(3000)
    worker.wait()
    app.processEvents()
    assert heartbeat
    assert done.at(0)[0] == "ok"

    failure = OperationWorker(lambda: (_ for _ in ()).throw(ValueError("worker test error")))
    errors = QSignalSpy(failure.failed)
    failure.start()
    deadline = time.monotonic() + 3
    while errors.count() == 0 and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    failure.wait()
    assert errors.count() == 1
    assert "worker test error" in errors.at(0)[0]



def test_classification_annotation_reuses_last_saved_labels_and_persists_them(tmp_path):
    import json

    app = QApplication.instance() or QApplication([])
    tasks = validate_tasks([
        {"name": "Obstacle ahead", "classes": ["no", "yes"]},
        {"name": "Sign", "classes": ["right", "left", "straight", "stop", "90r", "90l"]},
    ])
    ensure_dataset(tmp_path, tasks)
    image_dir = tmp_path / "images"
    samples = []
    for index, filename in enumerate(("first.png", "second.png")):
        image = QImage(32, 24, QImage.Format.Format_RGB32)
        image.fill(Qt.GlobalColor.black)
        assert image.save(str(image_dir / filename))
        samples.append({
            "sample_id": f"sample-{index + 1}",
            "image": f"images/{filename}",
            "sha256": f"hash-{index + 1}",
            "session_id": "test-session",
            "captured_at": f"2026-11-27T10:0{index}:00+03:30",
            "labels": {},
        })
    (tmp_path / "samples.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in samples),
        encoding="utf-8",
    )

    page = ClassificationPage(tmp_path)
    page._open_dataset(tmp_path)
    page._show_current_sample()

    # Unsaved choices must not become the default for the next frame.
    assert [combo.currentData() for _task, combo in page.annotation_combos] == ["", ""]
    page.annotation_combos[0][1].setCurrentIndex(page.annotation_combos[0][1].findData("yes"))
    page.annotation_combos[1][1].setCurrentIndex(page.annotation_combos[1][1].findData("right"))
    page.next_image()
    assert [combo.currentData() for _task, combo in page.annotation_combos] == ["", ""]

    page.previous_image()
    page.annotation_combos[0][1].setCurrentIndex(page.annotation_combos[0][1].findData("yes"))
    page.annotation_combos[1][1].setCurrentIndex(page.annotation_combos[1][1].findData("right"))
    page.save_and_next()

    # The next unannotated frame inherits the last successfully saved labels.
    assert page.current_index == 1
    assert [combo.currentData() for _task, combo in page.annotation_combos] == ["yes", "right"]

    # Moving away from unsaved edits restores the saved defaults, not those edits.
    page.annotation_combos[0][1].setCurrentIndex(page.annotation_combos[0][1].findData("no"))
    page.annotation_combos[1][1].setCurrentIndex(page.annotation_combos[1][1].findData("left"))
    page.previous_image()
    page.next_image()
    assert [combo.currentData() for _task, combo in page.annotation_combos] == ["yes", "right"]
    page.close()
    app.processEvents()

    # The last saved choice is reconstructed from samples.jsonl after restart.
    reopened = ClassificationPage(tmp_path)
    reopened._open_dataset(tmp_path)
    assert reopened.current_index == 1
    assert [combo.currentData() for _task, combo in reopened.annotation_combos] == ["yes", "right"]

    reopened.previous_image()
    assert [combo.currentData() for _task, combo in reopened.annotation_combos] == ["yes", "right"]
    reopened.close()
    app.processEvents()

def test_classification_task_layout_expands_single_task_and_supports_three_tasks(tmp_path):
    app = QApplication.instance() or QApplication([])
    page = ClassificationPage(tmp_path)

    page.task_count.setValue(1)
    assert page.task_grid.count() == 1
    assert page.task_grid.itemAtPosition(0, 0).widget() is page.task_boxes[0]
    assert page.task_grid.itemAtPosition(0, 1).widget() is page.task_boxes[0]

    page.task_count.setValue(3)
    assert page.task_grid.count() == 3
    assert all(not page.task_boxes[index].isHidden() for index in range(3))

    page.close()
    app.processEvents()
