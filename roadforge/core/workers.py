from __future__ import annotations

import json
import threading
import time
import traceback
import inspect
import sys
import uuid
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from . import runtime
from .config import snapshot_config


class OperationWorker(QThread):
    progress = Signal(dict)
    log = Signal(str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, function, *args, **kwargs):
        super().__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            if "progress" in inspect.signature(self.function).parameters:
                self.kwargs["progress"] = lambda *args: self.progress.emit({"args": args})
            self.done.emit(self.function(*self.args, **self.kwargs))
        except Exception:
            self.failed.emit(traceback.format_exc())


class DatasetScanWorker(QThread):
    progress = Signal(str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, roots):
        super().__init__()
        self.roots = roots
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    def run(self):
        try:
            self.progress.emit("Discovering CARLA collections and AVIS image folders…")
            from .dataset import discover_collections
            result = discover_collections(self.roots)
            self.progress.emit(f"Found {len(result)} collections; preparing sample index…")
            self.done.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())


class ValidationWorker(QThread):
    progress = Signal(str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, samples, ignore_label=None):
        super().__init__()
        self.samples = samples
        self.ignore_label = ignore_label

    def run(self):
        try:
            from .dataset import validate_samples
            self.progress.emit(f"Validating {len(self.samples):,} samples and decoding all combined masks…")
            self.done.emit(validate_samples(self.samples, self.ignore_label))
        except Exception:
            self.failed.emit(traceback.format_exc())


class WebcamWorker(QThread):
    frame = Signal(object)
    status = Signal(str)
    done = Signal()
    failed = Signal(str)

    def __init__(self, model_path, imgsz=512, device="auto", opacity=0.42, camera_index=0):
        super().__init__()
        self.model_path = model_path
        self.imgsz = imgsz
        self.device = device
        self.opacity = opacity
        self.camera_index = camera_index
        self._stop = threading.Event()

    def request_stop(self):
        self._stop.set()

    def run(self):
        capture = None
        try:
            import cv2
            from .runtime import SemanticEngine, overlay
            capture = cv2.VideoCapture(self.camera_index)
            if not capture.isOpened():
                raise OSError(f"Could not open webcam index {self.camera_index}.")
            engine = SemanticEngine(self.model_path, self.imgsz, self.device)
            self.status.emit("Webcam is live · click Webcam again to stop")
            frames = 0
            started = time.perf_counter()
            while not self._stop.is_set():
                ok, frame = capture.read()
                if not ok:
                    raise OSError("Webcam stopped returning frames.")
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                prediction = engine.predict(rgb)
                self.frame.emit(overlay(rgb, prediction, self.opacity))
                frames += 1
                if frames % 15 == 0:
                    elapsed = max(time.perf_counter() - started, 1e-6)
                    self.status.emit(f"Webcam inference · {frames/elapsed:.1f} FPS · {frames} frames")
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            if capture:
                capture.release()
            self.done.emit()


class TrainingWorker(QThread):
    progress = Signal(dict)
    log = Signal(str)
    done = Signal(dict)
    failed = Signal(str)

    def __init__(self, config: dict, data_yaml: str, workspace: str, resume: str | None = None):
        super().__init__()
        self.config = snapshot_config(config)
        self.data_yaml = Path(data_yaml).resolve()
        self.workspace = Path(workspace).resolve()
        self.resume_path = Path(resume).resolve() if resume else None
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._pause.set()
        self._log_file = None

    def request_stop(self):
        self._stop.set()
        self._pause.set()
        self.progress.emit({"state": "stop_requested", "message": "Stopping after the current epoch checkpoint…"})

    def request_pause(self):
        self._pause.clear()
        self.progress.emit({"state": "pause_requested", "message": "Pause requested; waiting for the current epoch checkpoint boundary."})

    def request_resume(self):
        self._pause.set()

    def _write_log(self, message: str):
        self.log.emit(message)
        if self._log_file:
            self._log_file.write(f"{datetime.now().astimezone().isoformat()} {message}\n")
            self._log_file.flush()

    def run(self):
        start = time.monotonic()
        run_dir = None
        run_id = ""
        execution_started = False
        try:
            if not self.data_yaml.is_file():
                raise FileNotFoundError(f"Dataset YAML does not exist: {self.data_yaml}")
            manifest_path = self.data_yaml.parent / "split_manifest.json"
            report_path = self.data_yaml.parent / "dataset_report.json"
            if manifest_path.is_file():
                try:
                    split_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise RuntimeError(f"Prepared dataset manifest is unreadable: {manifest_path}: {exc}") from exc
                sample_rows = split_manifest.get("samples", {})
                if not sample_rows.get("train") or not sample_rows.get("val"):
                    raise RuntimeError(
                        "This prepared dataset must contain at least one training sample and one validation sample. "
                        "Prepare non-empty train and validation splits before starting fine-tuning."
                    )
                if report_path.is_file():
                    try:
                        dataset_report = json.loads(report_path.read_text(encoding="utf-8"))
                        distribution = dataset_report.get("class_distribution", {})
                        ignored = sum(int(distribution.get(part, {}).get("ignored_pixel_count", 0)) for part in ("train", "val"))
                    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                        raise RuntimeError(f"Prepared dataset report is unreadable: {report_path}: {exc}") from exc
                    if ignored:
                        raise RuntimeError(
                            f"This prepared dataset contains {ignored:,} pixels marked as ignore_label. "
                            "The current Ultralytics semantic training API does not expose a verified ignore-index setting in this Studio build, "
                            "so training is blocked rather than silently treating those pixels as a real class. "
                            "Prepare a dataset without ignore pixels or use a trainer with explicit ignore-index support."
                        )
            self.workspace.mkdir(parents=True, exist_ok=True)
            if self.resume_path:
                if not self.resume_path.is_file():
                    raise FileNotFoundError(f"Resume checkpoint does not exist: {self.resume_path}")
                run_dir = self.resume_path.parent.parent.resolve()
                runs_root = (self.workspace / "runs").resolve()
                if run_dir.parent != runs_root:
                    raise ValueError(f"Resume checkpoint must belong to a Studio run under {runs_root}: {self.resume_path}")
                run_id = run_dir.name
            else:
                run_id = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
                run_dir = self.workspace / "runs" / run_id
                run_dir.mkdir(parents=True, exist_ok=False)
            self._log_file = (run_dir / "training.log").open("a", encoding="utf-8")
            model_cfg = self.config.get("model", {})
            checkpoint = str(model_cfg.get("checkpoint", "")).strip()
            size = str(model_cfg.get("size", "l")).lower()
            model_spec = checkpoint or f"yolo26{size}-sem.yaml"
            previous_snapshot = {}
            if self.resume_path and (run_dir / "config.json").is_file():
                try:
                    previous_snapshot = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    previous_snapshot = {}
            if self.resume_path:
                original_data = previous_snapshot.get("data_yaml")
                if not original_data:
                    args_path = run_dir / "args.yaml"
                    if args_path.is_file():
                        try:
                            import yaml
                            saved_args = yaml.safe_load(args_path.read_text(encoding="utf-8")) or {}
                            original_data = saved_args.get("data")
                        except Exception:
                            original_data = None
                if not original_data:
                    raise ValueError(
                        "Cannot safely resume this legacy checkpoint because its original dataset YAML cannot be verified. "
                        "Start a new fine-tuning run from best.pt instead."
                    )
                original_path = Path(str(original_data)).expanduser().resolve()
                if original_path != self.data_yaml:
                    raise ValueError(
                        "This is a true Ultralytics resume: it restores the original run arguments and dataset. "
                        f"The checkpoint was trained with {original_path}, but the selected dataset is {self.data_yaml}. "
                        "Select the original dataset YAML to resume, or start a new run from best.pt to fine-tune on a different dataset."
                    )
            if self.resume_path:
                compat = {"task": "semantic", "architecture": "resumed checkpoint", "source_classes": None, "target_classes": len(model_cfg.get("classes", {})), "class_adaptation_required": False}
                transfer = previous_snapshot.get("checkpoint_transfer", {"note": "Resumed from the original run checkpoint."})
                model_spec = str(self.resume_path)
            elif checkpoint and Path(checkpoint).suffix.lower() == ".pt":
                compat = runtime.verify_checkpoint(checkpoint, expected_classes=len(model_cfg.get("classes", {})))
                transfer = runtime.transfer_report(checkpoint, target_classes=len(model_cfg.get("classes", {})))
            else:
                # YAML configs are local architecture definitions; no checkpoint is fetched.
                compat = {"task": "semantic", "architecture": "YOLO26 semantic YAML", "source_classes": None, "target_classes": len(model_cfg.get("classes", {})), "class_adaptation_required": True}
                transfer = {"transferred_count": 0, "transferred": [], "modified_count": 0, "modified": [], "initialized_count": 0, "initialized": [], "skipped_count": 0, "skipped": [], "note": "Training starts from the selected architecture with random initialization."}
            self._write_log(f"Starting {run_id} · model={model_spec} · dataset={self.data_yaml}")
            self._write_log(f"Checkpoint compatibility: task={compat.get('task')} · source classes={compat.get('source_classes')} · target classes={compat.get('target_classes')}")
            self._write_log(f"Transfer report: {transfer['transferred_count']} tensors transferred, {transfer['modified_count']} class-shaped tensors modified, {transfer['initialized_count']} initialized, {transfer['skipped_count']} skipped.")
            try:
                import torch
                import ultralytics
            except ImportError as exc:
                raise RuntimeError(f"Training requires PyTorch and Ultralytics: {exc}") from exc
            effective = dict(self.config.get("training", {}))
            desired_device = str(effective.pop("device", "auto"))
            effective["device"] = "0" if desired_device == "auto" and torch.cuda.is_available() else ("cpu" if desired_device == "auto" else desired_device)
            if effective.get("freeze") in (None, ""):
                effective.pop("freeze", None)
            elif isinstance(effective["freeze"], str):
                effective["freeze"] = [int(v.strip()) for v in effective["freeze"].split(",") if v.strip()]
            from ultralytics.cfg import DEFAULT_CFG
            supported = {key: value for key, value in effective.items() if key in vars(DEFAULT_CFG)}
            unsupported = sorted(set(effective) - set(vars(DEFAULT_CFG)))
            if unsupported:
                self._write_log("Settings unsupported by the installed Ultralytics version were omitted: " + ", ".join(unsupported))
            checkpoint_hash = runtime.file_hash(checkpoint) if checkpoint and Path(checkpoint).is_file() else None
            try:
                split_hash = runtime.file_hash(self.data_yaml.parent / "split_manifest.json")
            except OSError:
                split_hash = None
            environment = runtime.environment_report()
            snapshot = {"format_version": 1, "run_id": run_id, "created_at": datetime.now().astimezone().isoformat(),
                        "config": self.config, "effective_args": supported, "task": "semantic", "data_yaml": str(self.data_yaml),
                        "split_manifest_sha256": split_hash, "checkpoint_sha256": checkpoint_hash, "checkpoint_transfer": transfer,
                        "versions": environment, "python_executable": sys.executable, "resume_from": str(self.resume_path) if self.resume_path else None}
            if not self.resume_path:
                runtime.atomic_json(run_dir / "config.json", snapshot)
            elif not (run_dir / "config.json").is_file():
                # Preserve legacy resumable runs that predate run snapshots without replacing any existing metadata.
                runtime.atomic_json(run_dir / "config.json", snapshot)
            if not self.resume_path and (self.data_yaml.parent / "split_manifest.json").is_file():
                import shutil
                shutil.copy2(self.data_yaml.parent / "split_manifest.json", run_dir / "dataset_snapshot.json")
            from ultralytics import YOLO
            if self.resume_path:
                model = runtime._ultralytics_yolo(self.resume_path, "semantic")
                train_args = {"resume": True}
                self._write_log(f"Resuming interrupted run from {self.resume_path}; original checkpoint arguments will be restored.")
            else:
                model = runtime._ultralytics_yolo(model_spec, "semantic")
                train_args = {"data": str(self.data_yaml), "task": "semantic", **supported,
                              "project": str(run_dir.parent), "name": run_id, "exist_ok": True,
                              "save": True, "plots": True, "verbose": True}
            state = {"last_batch": -1, "epoch_start": time.monotonic()}
            def on_batch_end(trainer):
                batch = int(getattr(trainer, "batch_i", -1))
                if batch == state["last_batch"]:
                    return
                state["last_batch"] = batch
                metrics = getattr(trainer, "metrics", {}) or {}
                self.progress.emit({"state": "running", "epoch": int(getattr(trainer, "epoch", 0)) + 1,
                                    "epochs": int(getattr(trainer, "epochs", 0)), "batch": batch + 1,
                                    "batches": len(getattr(trainer, "train_loader", [])),
                                    "loss": float(getattr(trainer, "tloss", 0).mean().item()) if hasattr(getattr(trainer, "tloss", None), "mean") else None,
                                    "metrics": {str(k): float(v) for k, v in metrics.items() if isinstance(v, (float, int))}})
            def on_epoch_end(trainer):
                metrics = getattr(trainer, "metrics", {}) or {}
                lr = None
                try:
                    lr = float(trainer.optimizer.param_groups[0]["lr"])
                except Exception:
                    pass
                payload = {"state": "epoch_complete", "epoch": int(getattr(trainer, "epoch", 0)) + 1,
                           "epochs": int(getattr(trainer, "epochs", 0)), "lr": lr,
                           "metrics": {str(k): float(v) for k, v in metrics.items() if isinstance(v, (float, int))}}
                self.progress.emit(payload)
                self._write_log(f"Epoch {payload['epoch']}/{payload['epochs']} · lr={lr} · metrics={payload['metrics']}")
                if self._stop.is_set():
                    trainer.stop = True
                    return
                if not self._pause.is_set():
                    self.progress.emit({"state": "paused", "message": "Checkpoint saved; training is paused at the epoch boundary."})
                    while not self._pause.wait(0.2):
                        if self._stop.is_set():
                            trainer.stop = True
                            return
                    self.progress.emit({"state": "running", "message": "Training resumed."})
            model.add_callback("on_train_batch_end", on_batch_end)
            model.add_callback("on_fit_epoch_end", on_epoch_end)
            running_meta = {"run_id": run_id, "run_dir": str(run_dir.resolve()), "status": "running",
                            "best": str((run_dir / "weights" / "best.pt").resolve()), "last": str((run_dir / "weights" / "last.pt").resolve()),
                            "resumed_from": str(self.resume_path) if self.resume_path else None}
            runtime.atomic_json(run_dir / "run_metadata.json", running_meta)
            execution_started = True
            results = model.train(**train_args)
            save_dir = Path(getattr(model.trainer, "save_dir", run_dir)).resolve()
            final = {"run_id": run_id, "run_dir": str(save_dir.resolve()), "best": str((save_dir / "weights" / "best.pt").resolve()),
                     "last": str((save_dir / "weights" / "last.pt").resolve()), "stopped": self._stop.is_set(),
                     "status": "stopped" if self._stop.is_set() else "complete",
                     "elapsed_seconds": time.monotonic() - start, "transfer_report": transfer,
                     "metrics": getattr(results, "results_dict", {}) if results is not None else {}}
            runtime.atomic_json(save_dir / "run_metadata.json", final)
            runtime.atomic_json(save_dir / "metrics.json", {"results": final["metrics"], "stopped": final["stopped"], "elapsed_seconds": final["elapsed_seconds"]})
            self._write_log(f"Run ended after {final['elapsed_seconds']:.1f}s. Best checkpoint: {final['best']}")
            self.done.emit(final)
        except Exception as exc:
            details = traceback.format_exc()
            if "out of memory" in str(exc).lower() or "cuda" in str(exc).lower() and "memory" in str(exc).lower():
                details += "\nMemory guidance: preserve the checkpoint, then try a smaller batch or image size and resume from last.pt. No settings were changed automatically."
            if run_dir is not None and (not self.resume_path or execution_started):
                try:
                    if self._log_file:
                        self._write_log("ERROR · " + str(exc))
                    try:
                        prior = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        prior = {}
                    prior.update({"run_id": run_id, "run_dir": str(run_dir.resolve()), "status": "failed", "error": str(exc),
                                  "last": str((run_dir / "weights" / "last.pt").resolve())})
                    runtime.atomic_json(run_dir / "run_metadata.json", prior)
                except Exception:
                    pass
            self.failed.emit(details)
        finally:
            if self._log_file:
                self._log_file.close()
