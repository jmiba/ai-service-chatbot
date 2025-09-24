from __future__ import annotations

from pathlib import Path
import os
import sys
import json
import tomllib  # Python 3.11+

from openai import OpenAI


def load_secrets() -> dict:
    candidates = [
        Path.cwd() / ".streamlit" / "secrets.toml",
        Path(__file__).resolve().parents[1] / ".streamlit" / "secrets.toml",
    ]
    for p in candidates:
        if p.is_file():
            with p.open("rb") as f:
                return tomllib.load(f)
    return {}


def safe_output_text(resp_obj) -> str:
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


secrets = load_secrets()
api_key = os.getenv("OPENAI_API_KEY") or secrets.get("OPENAI_API_KEY")
if not api_key:
    print("Missing OPENAI_API_KEY in environment or .streamlit/secrets.toml", file=sys.stderr)
    sys.exit(1)

model = secrets.get("MODEL", "gpt-4o-mini")
dbis_url = secrets.get("DBIS_MCP_SERVER_URL", "https://particular-green-lemming.fastmcp.app/mcp")

client = OpenAI(api_key=api_key)
resp = client.responses.create(
    model=model,
    input=[
        {
            "role": "user",
            "content": "Call dbis_list_subjects and return the first 3 subjects.",
        }
    ],
    tools=[
        {
            "type": "mcp",
            "server_label": "dbis",
            "server_url": dbis_url,
            "allowed_tools": ["dbis_list_subjects"],
            "require_approval": "never",
        }
    ],
    parallel_tool_calls=False,
    text={"verbosity": "medium"},
)
text = safe_output_text(resp)
if text and text.strip():
    print(text)
else:
    try:
        print(json.dumps(resp.model_dump(), indent=2)[:4000])
    except Exception:
        print(str(resp))