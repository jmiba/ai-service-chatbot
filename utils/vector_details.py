"""Persistence helpers for detailed vector store metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

BASE_DIR = Path(__file__).resolve().parents[1]
DETAILS_PATH = BASE_DIR / "state" / "vector_store_details.json"
SET_KEYS = {
    "vs_ids",
    "current_ids",
    "pending_replacement_ids",
    "true_orphan_ids",
    "excluded_live_ids",
    "combined_pending_cleanup_ids",
    "live_ids_display",
}


def _ensure_parent() -> None:
    DETAILS_PATH.parent.mkdir(parents=True, exist_ok=True)


def write_vector_store_details(data: Dict[str, Any]) -> None:
    """Persist vector store details to disk (JSON)."""

    _ensure_parent()
    serializable = dict(data)

    loaded_at = serializable.get("loaded_at")
    if hasattr(loaded_at, "isoformat"):
        serializable["loaded_at"] = loaded_at.isoformat()

    for key in SET_KEYS:
        value = serializable.get(key)
        if isinstance(value, set):
            serializable[key] = sorted(value)

    DETAILS_PATH.write_text(json.dumps(serializable, indent=2, sort_keys=True), encoding="utf-8")


def read_vector_store_details() -> Dict[str, Any] | None:
    """Load vector store details from disk and restore set fields."""

    try:
        raw = DETAILS_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None

    for key in SET_KEYS:
        value = data.get(key)
        if isinstance(value, list):
            data[key] = set(value)

    return data
