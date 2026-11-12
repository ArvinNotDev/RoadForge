"""Lossless RGB PNG and append-only image metadata storage."""
import json
from datetime import datetime
from pathlib import Path
import uuid
import uuid

import numpy as np
from PIL import Image


class DatasetStore:
    def __init__(self, output_root: str, config: dict):
        root = Path(output_root).expanduser()
        if (root / "collection_manifest.json").is_file():
            self.path = root
            self.manifest_path = root / "collection_manifest.json"
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if self.manifest.get("format") != "carla-raw-rgb-v1":
                raise ValueError("Unsupported collection manifest")
        else:
            self.path = root / ("collection_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
            self.path.mkdir(parents=True, exist_ok=False)
            self.manifest_path = self.path / "collection_manifest.json"
            self.manifest = {
                "format": "carla-raw-rgb-v1", "session_id": str(uuid.uuid4()),
                "created_at": datetime.now().astimezone().isoformat(),
                "config": config, "counts": {}, "towns": {},
            }
            self._write_manifest()
        (self.path / "logs").mkdir(exist_ok=True)

    def _write_manifest(self):
        temp = self.manifest_path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
        temp.replace(self.manifest_path)

    def existing_count(self, town: str) -> int:
        folder = self.path / "raw" / town / "images"
        if not folder.exists():
            return 0
        return sum(1 for p in folder.glob(f"{town}_*.png") if p.is_file())

    def save(self, town: str, rgb: np.ndarray, metadata: dict) -> str:
        folder = self.path / "raw" / town / "images"
        folder.mkdir(parents=True, exist_ok=True)
        numbers = []
        for path in folder.glob(f"{town}_*.png"):
            try: numbers.append(int(path.stem.rsplit("_", 1)[1]))
            except (ValueError, IndexError): pass
        number = max(numbers, default=0) + 1
        filename = f"{town}_{number:06d}.png"
        target = folder / filename
        # Commit a complete PNG atomically; never replace a prior frame.
        temp = folder / f".{filename}.{uuid.uuid4().hex}.tmp"
        try:
            Image.fromarray(rgb, mode="RGB").save(temp, format="PNG", optimize=False)
            temp.rename(target)
        finally:
            if temp.exists(): temp.unlink()
        record = dict(metadata, town=town, filename=filename,
                      collection_session_id=self.manifest["session_id"])
        metadata_path = self.path / "raw" / town / "metadata.jsonl"
        with metadata_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
        self.manifest["counts"][town] = number
        self.manifest["towns"][town] = {"last_image": filename, "target": metadata.get("town_target")}
        self._write_manifest()
        return str(target)

    def log(self, message: str):
        with (self.path / "logs" / "collector.log").open("a", encoding="utf-8") as stream:
            stream.write(f"{datetime.now().astimezone().isoformat()} {message}\n")
