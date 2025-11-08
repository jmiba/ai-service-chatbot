"""Helpers for persisting scheduled scraper settings."""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any, Dict

BASE_DIR = Path(__file__).resolve().parents[1]
SCHEDULE_PATH = BASE_DIR / "state" / "scraper_schedule.json"

DEFAULT_SCHEDULE: Dict[str, Any] = {
    "enabled": False,
    "interval_hours": 12.0,  # fallback when no run_times provided
    "run_times": ["00:00", "12:00"],
    "mode": "both",
    "crawl_budget": 30000,
    "keep_query": "",
    "dry_run": False,
    "last_run_at": None,
}


def _ensure_parent() -> None:
    SCHEDULE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _normalize_times(values: Any) -> list[str]:
    if not values:
        return []
    norm: list[str] = []
    seen: set[str] = set()
    for item in values:
        if item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        parts = text.split(":")
        if len(parts) != 2:
            continue
        try:
            hour = int(parts[0])
            minute = int(parts[1])
        except ValueError:
            continue
        if not (0 <= hour < 24 and 0 <= minute < 60):
            continue
        normalized = f"{hour:02d}:{minute:02d}"
        if normalized not in seen:
            seen.add(normalized)
            norm.append(normalized)
    return sorted(norm)


def _normalize_schedule(data: Dict[str, Any] | None) -> Dict[str, Any]:
    result = DEFAULT_SCHEDULE.copy()
    if isinstance(data, dict):
        result.update(data)

    result["enabled"] = bool(result.get("enabled", False))

    try:
        interval_hours = float(result.get("interval_hours", DEFAULT_SCHEDULE["interval_hours"]))
    except (TypeError, ValueError):
        interval_hours = DEFAULT_SCHEDULE["interval_hours"]
    result["interval_hours"] = max(0.5, min(interval_hours, 168.0))

    mode = str(result.get("mode", DEFAULT_SCHEDULE["mode"])).strip().lower()
    if mode not in {"scrape", "vectorize", "both"}:
        mode = DEFAULT_SCHEDULE["mode"]
    result["mode"] = mode

    try:
        crawl_budget = int(result.get("crawl_budget", DEFAULT_SCHEDULE["crawl_budget"]))
    except (TypeError, ValueError):
        crawl_budget = DEFAULT_SCHEDULE["crawl_budget"]
    result["crawl_budget"] = max(1000, min(crawl_budget, 200000))

    keep_query = result.get("keep_query", "") or ""
    result["keep_query"] = str(keep_query)

    result["dry_run"] = bool(result.get("dry_run", False))
    result["run_times"] = _normalize_times(result.get("run_times"))

    last_run_raw = result.get("last_run_at")
    if isinstance(last_run_raw, str):
        try:
            _ = _dt.datetime.fromisoformat(last_run_raw)
        except ValueError:
            result["last_run_at"] = None
    elif last_run_raw is not None:
        result["last_run_at"] = None

    return result


def read_scraper_schedule() -> Dict[str, Any]:
    try:
        raw = SCHEDULE_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return DEFAULT_SCHEDULE.copy()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return DEFAULT_SCHEDULE.copy()

    return _normalize_schedule(data)


def write_scraper_schedule(data: Dict[str, Any]) -> Dict[str, Any]:
    normalized = _normalize_schedule(data)
    _ensure_parent()
    SCHEDULE_PATH.write_text(json.dumps(normalized, indent=2, sort_keys=True), encoding="utf-8")
    return normalized


def update_last_scrape_run(timestamp: _dt.datetime | None = None) -> Dict[str, Any]:
    schedule = read_scraper_schedule()
    ts = timestamp or _dt.datetime.now(_dt.timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=_dt.timezone.utc)
    schedule["last_run_at"] = ts.isoformat()
    return write_scraper_schedule(schedule)
