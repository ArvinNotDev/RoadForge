from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path


def _registry_path(workspace: str | Path) -> Path:
    return Path(workspace).expanduser().resolve() / "model_registry.json"


def list_models(workspace: str | Path) -> list[dict]:
    path = _registry_path(workspace)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read model registry {path}: {exc}") from exc
    models = value.get("models", []) if isinstance(value, dict) else []
    return [item for item in models if isinstance(item, dict)]


def register_model(workspace: str | Path, model_path: str | Path, details: dict | None = None) -> dict:
    source = Path(model_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    models = list_models(workspace)
    entry = {
        "name": source.stem,
        "path": str(source),
        "format": source.suffix.lower().lstrip("."),
        "task": "semantic",
        "sha256": digest.hexdigest(),
        "registered_at": datetime.now().astimezone().isoformat(),
        "details": details or {},
    }
    match = next((i for i, item in enumerate(models) if item.get("sha256") == entry["sha256"] or item.get("path") == entry["path"]), None)
    if match is None:
        models.append(entry)
    else:
        models[match] = {**models[match], **entry}
    target = _registry_path(workspace)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"format_version": 1, "models": models}, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(target)
    return entry


def remove_model(workspace: str | Path, model_path: str | Path) -> bool:
    target = _registry_path(workspace)
    models = list_models(workspace)
    path = str(Path(model_path).expanduser().resolve())
    remaining = [item for item in models if item.get("path") != path]
    if len(remaining) == len(models):
        return False
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"format_version": 1, "models": remaining}, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(target)
    return True
