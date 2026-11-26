from __future__ import annotations

from pathlib import Path

import pytest


def test_teacher_discovery_accepts_avis_capture_folder_without_collection_manifest(tmp_path):
    pytest.importorskip("PySide6")
    from datacollector.teacher_labeling import discover_collections

    capture_root = tmp_path / "AVIS_RoadForge_Captures"
    image_dir = capture_root / "images"
    image_dir.mkdir(parents=True)
    (image_dir / "frame_0001.jpg").write_bytes(b"not decoded during discovery")
    (capture_root / "manifest.jsonl").write_text(
        '{"image":"images/frame_0001.jpg","session_id":"avis-session"}\n',
        encoding="utf-8",
    )

    discovered = discover_collections(capture_root)
    assert list(discovered) == [capture_root]
    assert set(discovered[capture_root]) == {"AVIS"}
    assert discovered[capture_root]["AVIS"] == [image_dir / "frame_0001.jpg"]


def test_teacher_discovery_accepts_parent_folder_containing_avis_capture_folder(tmp_path):
    pytest.importorskip("PySide6")
    from datacollector.teacher_labeling import discover_collections

    capture_root = tmp_path / "AVIS_RoadForge_Captures"
    image_dir = capture_root / "images"
    image_dir.mkdir(parents=True)
    (image_dir / "frame_0001.jpg").write_bytes(b"not decoded during discovery")

    discovered = discover_collections(tmp_path)
    assert capture_root in discovered
    assert discovered[capture_root]["AVIS"] == [image_dir / "frame_0001.jpg"]
