import time

import pytest
pytest.importorskip("PySide6")

from PySide6.QtCore import QTimer
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication

from roadforge.core.workers import OperationWorker
from roadforge.main import MainWindow


def test_main_window_loads_all_sections_and_controls(tmp_path, monkeypatch):
    import roadforge.main as app_module
    monkeypatch.setattr(app_module, "APP_DIR", tmp_path / "studio")
    monkeypatch.setattr(app_module, "PROJECT_ROOT", tmp_path)
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    assert window.stack.count() == 10
    assert window.PAGES[0] == "Dashboard"
    assert window.model_name.text() == "YOLO26l-sem · semantic"
    assert all(edit.isReadOnly() for edit in window.class_names)
    assert [edit.text() for edit in window.class_names] == ["background", "ego_drivable_area", "lane_lines", "vehicles"]
    assert hasattr(window, "traffic_filter")
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
