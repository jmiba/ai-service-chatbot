from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

BASE_DIR = Path(__file__).resolve().parents[1]
ERROR_CONFIG_PATH = BASE_DIR / "config" / "error_codes.json"

DEFAULT_ERROR_METADATA: Dict[str, Dict[str, str]] = {
    "E00": {
        "label": "Question",
        "description": "Information-seeking request (neutral/curious).",
        "sentiment": "wants to know",
    },
    "E01": {
        "label": "Comment",
        "description": "Feedback, praise, or complaint without an action request.",
        "sentiment": "commenting/venting",
    },
    "E02": {
        "label": "Suggestion",
        "description": "User proposes improvements or changes.",
        "sentiment": "suggesting change",
    },
    "E03": {
        "label": "Invective",
        "description": "Insults, threats, or hostile content.",
        "sentiment": "aggressive/hostile",
    },
}


def _normalize_code(code: str | int | None) -> str:
    if code is None:
        return ""
    text = str(code).strip().upper()
    if not text:
        return ""
    if not text.startswith("E") and text.isdigit():
        return f"E{int(text):02d}"
    return text


def load_error_code_metadata() -> Dict[str, Dict[str, str]]:
    metadata = {code: meta.copy() for code, meta in DEFAULT_ERROR_METADATA.items()}
    try:
        raw = ERROR_CONFIG_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
        if isinstance(data, dict):
            for key, value in data.items():
                normalized = _normalize_code(key)
                if not normalized:
                    continue
                if isinstance(value, dict):
                    entry = {str(k): str(v) for k, v in value.items()}
                else:
                    entry = {"label": str(value)}
                merged = metadata.get(normalized, {}).copy()
                merged.update(entry)
                metadata[normalized] = merged
    except Exception:
        pass
    return metadata


def load_error_code_labels() -> Dict[str, str]:
    metadata = load_error_code_metadata()
    labels: Dict[str, str] = {}
    for code, meta in metadata.items():
        label = meta.get("label") or meta.get("description") or code
        labels[code] = str(label)

    numeric_aliases: Dict[str, str] = {}
    for code, label in labels.items():
        digits = "".join(ch for ch in code if ch.isdigit())
        if digits:
            numeric_aliases[digits] = label
    labels.update(numeric_aliases)
    return labels


def human_error_label(code: str | None, *, labels: Dict[str, str] | None = None) -> str:
    labels = labels or load_error_code_labels()
    default_label = labels.get("E00") or labels.get("0") or "Question"
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
    metadata = load_error_code_metadata()
    if not metadata:
        return ""
    canonical_labels: dict[str, str] = {}
    for code, meta in metadata.items():
        label = meta.get("label") or code
        desc = meta.get("description") or meta.get("sentiment")
        if desc:
            display = f"{label} ({desc})"
        else:
            display = label
        digits = "".join(ch for ch in code if ch.isdigit())
        if digits:
            canonical = digits.lstrip("0") or "0"
        else:
            canonical = code
        canonical_labels.setdefault(canonical, display)

    def sort_key(item: tuple[str, str]) -> tuple[int, object]:
        key, _ = item
        if key.isdigit():
            return (0, int(key))
        return (1, key)

    parts = [f"{key}={label}" for key, label in sorted(canonical_labels.items(), key=sort_key)]
    return "request type codes: " + ", ".join(parts)
