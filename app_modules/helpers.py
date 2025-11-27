"""
Helper functions for consistent attribute access and text processing.
Consolidates the duplicated _ga()/_get_attr() helpers that appeared 4+ times in app.py.
"""
from __future__ import annotations

import re
import json
from typing import Any


def get_attr(obj: Any, name: str, default: Any = None) -> Any:
    """
    Unified getter that works with both dicts and objects.
    Replaces the many duplicated _ga()/_get_attr() definitions.
    
    Args:
        obj: Dict or object to read from
        name: Attribute/key name
        default: Default value if not found
        
    Returns:
        The value or default
    """
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def redact_ids(data: Any, keep_prefix: int = 8) -> Any:
    """
    Recursively redact identifiers in dicts/lists.
    Useful for logging without exposing full IDs.
    
    Args:
        data: Data structure to redact
        keep_prefix: Number of characters to keep visible
        
    Returns:
        Data with IDs redacted
    """
    if isinstance(data, dict):
        result = {}
        for k, v in data.items():
            if k in ("id", "file_id", "response_id") and isinstance(v, str) and len(v) > keep_prefix:
                result[k] = v[:keep_prefix] + "..."
            else:
                result[k] = redact_ids(v, keep_prefix)
        return result
    elif isinstance(data, list):
        return [redact_ids(item, keep_prefix) for item in data]
    return data


def safe_output_text(resp_obj: Any) -> str:
    """
    Return the best-effort text from a Responses API object.
    Works even if output_text attribute is missing.
    
    Args:
        resp_obj: OpenAI Responses API response object
        
    Returns:
        The extracted text content or empty string
    """
    txt = getattr(resp_obj, "output_text", None)
    if txt:
        return txt
        
    out = getattr(resp_obj, "output", None) or []
    for item in out:
        content = getattr(item, "content", None) or []
        for part in content:
            t = getattr(part, "text", None)
            if isinstance(t, str) and t.strip():
                return t
            if hasattr(t, "value") and isinstance(t.value, str) and t.value.strip():
                return t.value
            if isinstance(part, dict):
                ptxt = part.get("text")
                if isinstance(ptxt, str) and ptxt.strip():
                    return ptxt
                if isinstance(ptxt, dict) and isinstance(ptxt.get("value"), str) and ptxt["value"].strip():
                    return ptxt["value"]
    return ""


def humanize_debug_text(text: str) -> str:
    """
    Make escaped newline/tab sequences readable for debug printing.
    
    Args:
        text: Text with escaped sequences
        
    Returns:
        Text with sequences replaced by actual characters
    """
    if not isinstance(text, str):
        return str(text)
    return (
        text.replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\r", "\r")
    )


def extract_output_json(resp_obj: Any) -> dict[str, Any]:
    """
    Return first JSON dict emitted via response_format from a Responses API object.
    
    Args:
        resp_obj: OpenAI Responses API response object
        
    Returns:
        Extracted JSON dict or empty dict
    """
    output = getattr(resp_obj, "output", None)
    if not output:
        return {}
        
    for item in output:
        content = getattr(item, "content", None) or []
        for part in content:
            candidate = getattr(part, "json", None)
            if isinstance(candidate, dict):
                return candidate
            if isinstance(part, dict):
                inner = part.get("json")
                if isinstance(inner, dict):
                    return inner
    return {}


def extract_first_json_object(text: str) -> dict[str, Any] | None:
    """
    Extract the first valid JSON object from a text string.
    Useful for parsing LLM outputs that contain JSON.
    
    Args:
        text: Text containing JSON
        
    Returns:
        Parsed dict or None if not found
    """
    if not text or not isinstance(text, str):
        return None
        
    # Try to find JSON object boundaries
    start = text.find("{")
    if start == -1:
        return None
        
    # Track brace depth to find matching close
    depth = 0
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    # Try next occurrence
                    next_start = text.find("{", start + 1)
                    if next_start == -1:
                        return None
                    start = next_start
                    depth = 1
                    
    return None


def iter_content_items(final_response: Any):
    """
    Iterate over output items from a Responses API final object.
    
    Args:
        final_response: Final response object with output attribute
        
    Yields:
        Output items
    """
    output = get_attr(final_response, "output", []) or []
    for item in output:
        yield item


def iter_content_parts(item: Any):
    """
    Iterate over content parts from an output item.
    
    Args:
        item: Output item with content attribute
        
    Yields:
        Content parts
    """
    content = get_attr(item, "content", []) or []
    for part in content:
        yield part


def get_usage_from_final(final_obj: Any) -> tuple[int, int, int, int]:
    """
    Best-effort extraction of usage fields from a Responses API object or dict.
    
    Args:
        final_obj: Final response object or dict with usage info
        
    Returns:
        Tuple of (input_tokens, output_tokens, total_tokens, reasoning_tokens)
    """
    usage = get_attr(final_obj, "usage")
    if usage is None and isinstance(final_obj, dict):
        usage = final_obj.get("usage")
        
    input_tokens = get_attr(usage, "input_tokens", None) if usage is not None else None
    if input_tokens is None:
        input_tokens = get_attr(usage, "prompt_tokens", None) if usage is not None else None
        
    output_tokens = get_attr(usage, "output_tokens", None) if usage is not None else None
    total_tokens = get_attr(usage, "total_tokens", None) if usage is not None else None
    
    # reasoning tokens may be nested
    reasoning_tokens = None
    otd = get_attr(usage, "output_tokens_details", None) if usage is not None else None
    if otd is not None:
        reasoning_tokens = get_attr(otd, "reasoning_tokens", None)
        
    return (
        input_tokens if isinstance(input_tokens, int) else 0,
        output_tokens if isinstance(output_tokens, int) else 0,
        total_tokens if isinstance(total_tokens, int) else ((input_tokens or 0) + (output_tokens or 0)),
        reasoning_tokens if isinstance(reasoning_tokens, int) else 0,
    )


def coerce_text_part(part: Any) -> dict[str, Any] | None:
    """
    Normalize a content part to: {"text": <str>, "annotations": <list>}
    
    Handles:
      - {"type":"output_text","text":"...", "annotations":[...]}
      - {"type":"output_text","text":{"value":"...","annotations":[...]}}
      - pydantic objects with .type, .text(.value/.annotations) or .annotations
      
    Args:
        part: Content part from response
        
    Returns:
        Normalized dict with text and annotations, or None if not text content
    """
    ptype = get_attr(part, "type", None)
    if ptype not in ("output_text", "text"):
        return None

    text_field = get_attr(part, "text", None)

    # A) text is a plain string
    if isinstance(text_field, str):
        annotations = get_attr(part, "annotations", []) or []
        return {"text": text_field, "annotations": annotations}

    # B) text is an object/dict with .value and .annotations
    if text_field is not None:
        value = get_attr(text_field, "value", None)
        ann = get_attr(text_field, "annotations", []) or []
        if isinstance(value, str):
            return {"text": value, "annotations": ann}

    # C) Fallback
    text_str = "" if text_field is None else (text_field if isinstance(text_field, str) else str(text_field))
    annotations = get_attr(part, "annotations", []) or []
    return {"text": text_str, "annotations": annotations}
