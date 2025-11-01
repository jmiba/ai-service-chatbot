"""Utilities for persisting and reading vector store status snapshots."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


BASE_DIR = Path(__file__).resolve().parent.parent
STATUS_DIR = BASE_DIR / "state"
STATUS_PATH = STATUS_DIR / "vector_status.json"


def _default_serializer(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)!r} not serializable")


def write_vector_status(data: Dict[str, Any]) -> None:
    """Persist the latest vector store status snapshot to disk."""

    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(data, default=_default_serializer, indent=2), encoding="utf-8")


def read_vector_status() -> Dict[str, Any] | None:
    """Read the cached vector store status snapshot if it exists."""

    if not STATUS_PATH.exists():
        return None
    try:
        content = STATUS_PATH.read_text(encoding="utf-8")
        if not content.strip():
            return None
        return json.loads(content)
    except Exception:
        return None
