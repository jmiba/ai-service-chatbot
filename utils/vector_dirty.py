"""Helpers to mark when manual DB actions require vector snapshot refresh."""

from __future__ import annotations

import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DIRTY_FLAG_PATH = BASE_DIR / "state" / "vector_store_dirty.flag"


def mark_vector_store_dirty() -> None:
    DIRTY_FLAG_PATH.parent.mkdir(parents=True, exist_ok=True)
    DIRTY_FLAG_PATH.write_text(str(time.time()), encoding="utf-8")


def clear_vector_store_dirty() -> None:
    try:
        DIRTY_FLAG_PATH.unlink()
    except FileNotFoundError:
        pass


def is_vector_store_dirty() -> bool:
    return DIRTY_FLAG_PATH.exists()
