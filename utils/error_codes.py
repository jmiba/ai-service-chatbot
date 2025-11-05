from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

BASE_DIR = Path(__file__).resolve().parents[1]
ERROR_CONFIG_PATH = BASE_DIR / "config" / "error_codes.json"

DEFAULT_ERROR_LABELS: Dict[str, str] = {
    "E00": "Good answer",
    "E01": "Need more information from user",
    "E02": "Significant gaps – user might be disappointed",
    "E03": "User needs human assistance",
}


def load_error_code_labels() -> Dict[str, str]:
    labels: Dict[str, str] = DEFAULT_ERROR_LABELS.copy()
    try:
        raw = ERROR_CONFIG_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
        if isinstance(data, dict):
            for key, value in data.items():
                labels[str(key).upper()] = str(value)
    except Exception:
        pass

    numeric_aliases: Dict[str, str] = {}
    for code, label in labels.items():
        digits = "".join(ch for ch in code if ch.isdigit())
        if digits:
            numeric_aliases[digits] = label
    labels.update(numeric_aliases)
    return labels


def human_error_label(code: str | None, *, labels: Dict[str, str] | None = None) -> str:
    labels = labels or load_error_code_labels()
    default_label = labels.get("E00") or labels.get("0") or "Good answer"
    if not code:
        return default_label

    normalized = str(code).strip().upper()
    if not normalized:
        return default_label

    if normalized in labels:
        return labels[normalized]

    if normalized.startswith("E"):
        padded = normalized
    else:
        padded = f"E{int(normalized):02d}" if normalized.isdigit() else normalized

    if padded in labels:
        return labels[padded]

    numeric = normalized.lstrip("E").lstrip("0")
    if numeric and numeric.isdigit():
        padded_numeric = f"E{int(numeric):02d}"
        if padded_numeric in labels:
            return labels[padded_numeric]

    return normalized


def format_error_code_legend() -> str:
    labels = load_error_code_labels()
    if not labels:
        return ""
    def sort_key(item):
        code, _ = item
        digits = ''.join(ch for ch in code if ch.isdigit())
        return int(digits) if digits else 0
    parts = []
    for code, label in sorted(labels.items(), key=sort_key):
        digits = ''.join(ch for ch in code if ch.isdigit())
        canonical = digits if digits else code
        parts.append(f"{canonical}={label}")
    return "Error codes: " + ", ".join(parts)
