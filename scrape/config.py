from __future__ import annotations

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROMPT_CONFIG_PATH = BASE_DIR / ".streamlit" / "prompts.json"


def load_summarize_prompts() -> tuple[str, str]:
    """Load LLM prompts from config, raising actionable errors when misconfigured."""
    try:
        raw_config = PROMPT_CONFIG_PATH.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        message = (
            f"Prompt configuration not found at {PROMPT_CONFIG_PATH}. "
            "Create the file and provide a 'summarize_and_tag' entry with 'system' and 'user' prompts."
        )
        raise RuntimeError(message) from exc

    try:
        data = json.loads(raw_config)
    except json.JSONDecodeError as exc:
        message = f"Prompt configuration {PROMPT_CONFIG_PATH} is not valid JSON: {exc}."
        raise RuntimeError(message) from exc

    if not isinstance(data, dict):
        message = f"Prompt configuration {PROMPT_CONFIG_PATH} must contain a top-level JSON object."
        raise RuntimeError(message)

    prompts = data.get("summarize_and_tag")
    if not isinstance(prompts, dict):
        message = "Prompt configuration is missing the 'summarize_and_tag' object."
        raise RuntimeError(message)

    system_prompt = prompts.get("system")
    user_prompt = prompts.get("user")
    if not system_prompt or not user_prompt:
        message = "Prompt configuration must define both 'system' and 'user' keys under 'summarize_and_tag'."
        raise RuntimeError(message)

    return system_prompt, user_prompt
