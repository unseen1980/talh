"""
Helpers for recording paper-facing artifacts for the 0.5B edge study.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _normalize_path(path: str | Path) -> str:
    return str(Path(path))


def load_manifest(manifest_path: str | Path) -> list[dict[str, Any]]:
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        return []
    data = json.loads(manifest_file.read_text())
    if isinstance(data, list):
        return data
    raise ValueError(f"Manifest must be a JSON list: {manifest_file}")


def write_manifest(manifest_path: str | Path, entries: list[dict[str, Any]]) -> None:
    manifest_file = Path(manifest_path)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(json.dumps(entries, indent=2, sort_keys=True))


def register_artifact(
    manifest_path: str | Path,
    *,
    artifact_path: str | Path,
    artifact_type: str,
    source_script: str,
    stage: str,
    metadata: dict[str, Any],
) -> None:
    artifact_path_str = _normalize_path(artifact_path)
    entries = [
        entry for entry in load_manifest(manifest_path)
        if entry.get("artifact_path") != artifact_path_str
    ]
    entries.append({
        "artifact_path": artifact_path_str,
        "artifact_type": artifact_type,
        "source_script": source_script,
        "stage": stage,
        "metadata": metadata,
    })
    entries = sorted(entries, key=lambda entry: entry["artifact_path"])
    write_manifest(manifest_path, entries)
