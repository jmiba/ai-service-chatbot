"""
Prompt configuration loading and formatting functions.
Handles the prompts.json configuration for system prompts and evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


def load_prompt_config(config_path: Path) -> dict[str, Any]:
    """
    Load prompt configuration JSON and validate required sections.
    
    Args:
        config_path: Path to prompts.json file
        
    Returns:
        Parsed configuration dict
        
    Raises:
        RuntimeError: If config is missing or invalid
    """
    try:
        raw_config = config_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        message = (
            f"Prompt configuration not found at {config_path}. "
            "Create the file and provide the required entries (see README)."
        )
        raise RuntimeError(message) from exc

    try:
        data = json.loads(raw_config)
    except json.JSONDecodeError as exc:
        message = f"Prompt configuration {config_path} is not valid JSON: {exc}."
        raise RuntimeError(message) from exc

    if not isinstance(data, dict):
        message = f"Prompt configuration {config_path} must contain a top-level JSON object."
        raise RuntimeError(message)

    default_prompt = data.get("default_chat_system_prompt")
    if not isinstance(default_prompt, str) or not default_prompt.strip():
        message = "Prompt configuration is missing 'default_chat_system_prompt'."
        raise RuntimeError(message)

    evaluation_section = data.get("evaluation")
    evaluation_prompt = None
    if isinstance(evaluation_section, dict):
        evaluation_prompt = evaluation_section.get("system")
    if not isinstance(evaluation_prompt, str) or not evaluation_prompt.strip():
        message = "Prompt configuration is missing 'evaluation.system'."
        raise RuntimeError(message)

    return data


def build_eval_response_format(allowed_topics: Sequence[str] | None = None) -> dict[str, Any]:
    """
    Generate a json_schema descriptor matching prompts.json requirements.
    
    Args:
        allowed_topics: Optional list of allowed topic classifications
        
    Returns:
        Response format dict for OpenAI API
    """
    unique_topics: list[str] = []
    if allowed_topics:
        for val in allowed_topics:
            if val not in unique_topics:
                unique_topics.append(val)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "request_classification": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "request_type_code": {"type": "integer", "minimum": 0, "maximum": 3},
            "evaluation_notes": {"type": "string"},
        },
        "required": ["request_classification", "confidence", "request_type_code", "evaluation_notes"],
        "additionalProperties": False,
    }
    if unique_topics:
        schema["properties"]["request_classification"]["enum"] = unique_topics

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "interaction_evaluation",
            "strict": True,
            "schema": schema,
        },
    }


def format_prompt(
    template: str,
    *,
    datetime: str,
    doc_count: int | None = None,
) -> str:
    """
    Format a prompt template with runtime values.
    
    Args:
        template: Template string with {datetime} and optionally {doc_count} placeholders
        datetime: Current datetime string
        doc_count: Optional document count
        
    Returns:
        Formatted prompt string
    """
    safe_values = {"datetime": datetime}
    if "{doc_count" in template:
        safe_values["doc_count"] = doc_count if doc_count is not None else 0
    return template.format(**safe_values)
