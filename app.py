import streamlit as st
from openai import OpenAI, APIConnectionError, BadRequestError, RateLimitError, APIError
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import json, re, html, os
from urllib.parse import quote_plus
import psycopg2
from functools import lru_cache
import uuid
import threading
import httpx

# ---- utilities ----
from utils import (
    get_connection,
    initialize_default_prompt_if_empty, get_latest_prompt, render_sidebar,
    create_database_if_not_exists, create_llm_settings_table, get_llm_settings,
    supports_reasoning_effort, get_kb_entries, estimate_cost_usd,
    create_request_classifications_table, get_request_classifications,
    get_filter_settings,
    create_filter_settings_table,
    create_knowledge_base_table,
)

# -------------------------------------

st.set_page_config(page_title="Viadrina Library Assistant", layout="wide", initial_sidebar_state="collapsed")


DBIS_MCP_SERVER_LABEL = "dbis"
DBIS_MCP_ENV_KEY = f"OPENAI_MCP_SERVER_{DBIS_MCP_SERVER_LABEL.upper()}"
DBIS_MCP_SERVER_URL_KEY = "DBIS_MCP_SERVER_URL"
DBIS_MCP_AUTH_KEY = "DBIS_MCP_AUTHORIZATION"
DBIS_MCP_HEADERS_KEY = "DBIS_MCP_HEADERS"


def _mcp_preflight(url: str, headers: dict | None, authorization: str | None, timeout_s: float = 3.0) -> tuple[bool, str | None]:
    """Quick reachability/auth check for MCP server URL.
    Returns (ok, reason). Ok means reachable enough for the OpenAI client to try.
    If 401/403, returns ok=False with reason "unauthorized".
    If connect/DNS/TLS errors, returns ok=False with a short reason.
    Note: Some SSE servers delay response headers; a read-timeout is treated as reachable.
    """
    if not url:
        return False, "missing-url"
    req_headers = dict(headers or {})
    if authorization and "Authorization" not in req_headers:
        req_headers["Authorization"] = str(authorization)
    # Many MCP servers require SSE Accept header even for discovery endpoints
    if "Accept" not in req_headers:
        req_headers["Accept"] = "text/event-stream"
    try:
        resp = httpx.get(
            url,
            headers=req_headers or None,
            timeout=httpx.Timeout(timeout_s, read=timeout_s),
            follow_redirects=True,
        )
        if resp.status_code in (401, 403):
            return False, "unauthorized"
        if 200 <= resp.status_code < 500:
            return True, None
        return False, f"server-error-{resp.status_code}"
    except httpx.ReadTimeout:
        # Connected but no bytes in time (common with SSE); treat as reachable
        return True, "slow"
    except (httpx.ConnectTimeout, httpx.ConnectError):
        return False, "connect-error"
    except httpx.HTTPError as e:
        return False, f"http-error: {type(e).__name__}"


# -------------------------------------
def _redact_ids(obj_str: str) -> str:
    # redact typical object/ids like resp_, file_, vs_, msg_, fs_, txt_
    return re.sub(r'\b(resp|file|vs|msg|fs|txt)_[A-Za-z0-9\-]{6,}\b', r'\1_[REDACTED]', obj_str)

def _safe_output_text(resp_obj):
    """
    Return the best-effort text from a Responses API object, even if output_text is missing.
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


def _parse_headers_setting(header_value):
    """Parse headers configuration from secrets/env into a dict of strings."""
    if not header_value:
        return None
    if isinstance(header_value, dict):
        return {str(k): str(v) for k, v in header_value.items()}
    if isinstance(header_value, str):
        trimmed = header_value.strip()
        if not trimmed:
            return None
        try:
            parsed = json.loads(trimmed)
        except json.JSONDecodeError:
            if ":" in trimmed:
                key, _, val = trimmed.partition(":")
                return {key.strip(): val.strip()}
            return None
        else:
            if isinstance(parsed, dict):
                return {str(k): str(v) for k, v in parsed.items()}
    return None

# ---------- UI + constants ----------
# Check if user is authenticated (admin)
is_authenticated = st.session_state.get("authenticated", False)

debug_one = render_sidebar(authenticated=is_authenticated, show_debug=True)

BASE_DIR = Path(__file__).parent
PROMPT_CONFIG_PATH = BASE_DIR / ".streamlit" / "prompts.json"


def _load_prompt_config() -> dict:
    """Load prompt configuration JSON and validate required sections."""

    try:
        raw_config = PROMPT_CONFIG_PATH.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        message = (
            f"Prompt configuration not found at {PROMPT_CONFIG_PATH}. "
            "Create the file and provide the required entries (see README)."
        )
        st.error(message)
        raise RuntimeError(message) from exc

    try:
        data = json.loads(raw_config)
    except json.JSONDecodeError as exc:
        message = f"Prompt configuration {PROMPT_CONFIG_PATH} is not valid JSON: {exc}."
        st.error(message)
        raise RuntimeError(message) from exc

    if not isinstance(data, dict):
        message = f"Prompt configuration {PROMPT_CONFIG_PATH} must contain a top-level JSON object."
        st.error(message)
        raise RuntimeError(message)

    default_prompt = data.get("default_chat_system_prompt")
    if not isinstance(default_prompt, str) or not default_prompt.strip():
        message = "Prompt configuration is missing 'default_chat_system_prompt'."
        st.error(message)
        raise RuntimeError(message)

    evaluation_section = data.get("evaluation")
    evaluation_prompt = None
    if isinstance(evaluation_section, dict):
        evaluation_prompt = evaluation_section.get("system")
    if not isinstance(evaluation_prompt, str) or not evaluation_prompt.strip():
        message = "Prompt configuration is missing 'evaluation.system'."
        st.error(message)
        raise RuntimeError(message)

    return data


PROMPTS_LOAD_ERROR = None
try:
    PROMPT_CONFIG = _load_prompt_config()
except RuntimeError as exc:  # pragma: no cover - requires misconfiguration
    PROMPTS_LOAD_ERROR = str(exc)
    PROMPT_CONFIG = {}

if PROMPTS_LOAD_ERROR:
    st.stop()

DEFAULT_PROMPT = PROMPT_CONFIG["default_chat_system_prompt"]
EVALUATION_SYSTEM_TEMPLATE = PROMPT_CONFIG["evaluation"]["system"]

# Define avatars and icons centrally
def _build_icon_html(icon_path: Path, label: str) -> str:
    """Return inline SVG wrapped for display inside markdown."""
    try:
        svg_markup = icon_path.read_text(encoding="utf-8")
    except OSError:
        return ""

    updated_svg = re.sub(r'fill="#?[0-9a-fA-F]+"', 'fill="currentColor"', svg_markup)
    label_attr = html.escape(label, quote=True)
    return f"<span class='ref-icon' role='img' aria-label='{label_attr}'>{updated_svg}</span>"


AVATAR_ASSISTANT = str(BASE_DIR / "assets/robot_2.svg")
AVATAR_USER = str(BASE_DIR / "assets/face.svg")
WEB_ICON_HTML = _build_icon_html(BASE_DIR / "assets/language.svg", "Website source")
DOC_ICON_HTML = _build_icon_html(BASE_DIR / "assets/doc.svg", "Document source")

# --- Config: request classification options ---
@lru_cache(maxsize=1)
def get_allowed_request_classifications():
    """Load allowed request_classification values from database; ensure 'other' exists."""
    try:
        cats = get_request_classifications()
        # de-duplicate while preserving order
        seen = set(); clean = []
        for v in cats:
            if v not in seen:
                clean.append(v); seen.add(v)
        if 'other' not in clean:
            clean.append('other')
        return clean or ['other']
    except Exception:
        return ['library_hours','book_search','research_help','account_info','facility_info','policy_question','technical_support','other']


def log_interaction(user_input, assistant_response, session_id=None, citation_json=None, citation_count=0, confidence=0.0, error_code=None, request_classification=None, evaluation_notes=None,
                    model=None, usage_input_tokens=None, usage_output_tokens=None, usage_total_tokens=None, usage_reasoning_tokens=None, api_cost_usd=None, response_time_ms=None):
    # Debug: Print session_id to console for troubleshooting
    print(f"🔍 log_interaction called with session_id: {session_id} (type: {type(session_id)})")
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO log_table (
                        timestamp, session_id, user_input, assistant_response, error_code,
                        citation_count, citations, confidence, request_classification, evaluation_notes,
                        model, usage_input_tokens, usage_output_tokens, usage_total_tokens, usage_reasoning_tokens, api_cost_usd,
                        response_time_ms
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    datetime.now(), session_id, user_input, assistant_response, error_code,
                    citation_count, citation_json, confidence, request_classification, evaluation_notes,
                    model, usage_input_tokens, usage_output_tokens, usage_total_tokens, usage_reasoning_tokens, api_cost_usd,
                    response_time_ms
                ))
                conn.commit()
                print(f"✅ Successfully logged interaction with session_id: {session_id}")
    except psycopg2.Error as e:
        print(f"❌ DB logging error: {e}")


def log_interaction_async(*args, **kwargs) -> None:
    """Background wrapper so logging does not block the UI thread."""

    def _runner():
        try:
            log_interaction(*args, **kwargs)
        except Exception as exc:
            print(f"⚠️ async log_interaction failed: {exc}")

    threading.Thread(target=_runner, daemon=True).start()

def get_urls_and_titles_by_file_ids(conn, file_ids):
    """Return metadata keyed by file_id for citation rendering."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT vector_file_id, id, url, title, summary, recordset
            FROM documents
            WHERE vector_file_id = ANY(%s);
            """,
            (file_ids,)
        )
        rows = cur.fetchall()
        return {
            row[0]: {
                "doc_id": row[1],
                "url": row[2],
                "title": row[3],
                "summary": row[4],
                "recordset": row[5],
            }
            for row in rows
        }

# ---------- Cached file metadata lookups ----------
@lru_cache(maxsize=1024)
def _cached_filename(file_id: str) -> str:
    # client will be set later (global), so we access it lazily
    try:
        fi = client.files.retrieve(file_id)
        return getattr(fi, "filename", file_id)
    except Exception:
        return file_id

# ---------- Responses helpers (index-based citations) ----------
def _pick_index(ann, full_len):
    # Prefer explicit starts
    for key in ("start_index", "start", "index", "offset", "position"):
        v = ann.get(key) if isinstance(ann, dict) else getattr(ann, key, None)
        if v is not None:
            try:
                iv = int(v)
                return max(0, min(full_len, iv))
            except Exception:
                pass
    # Some SDKs nest values under the type key, e.g. ann["url_citation"]["start_index"]
    t = ann.get("type") if isinstance(ann, dict) else getattr(ann, "type", None)
    inner = ann.get(t) if isinstance(ann, dict) else getattr(ann, t, None)
    if isinstance(inner, dict):
        for key in ("start_index", "start", "index", "offset", "position"):
            v = inner.get(key)
            if v is not None:
                try:
                    iv = int(v)
                    return max(0, min(full_len, iv))
                except Exception:
                    pass
    return None

# --- helpers: strip markdown links (and grouped parenthesized link lists) from HTML-ish text
# Replace [anchor](url) with just 'anchor', preserving Markdown & spacing.
# - Skips images: ![alt](url)
# - Skips fenced code blocks: ```...```
# - Leaves whitespace/newlines intact

_LINK_RE = re.compile(r'(?<!!)\[([^\]]+)\]\((https?://[^)\s]+)\)')

def _strip_markdown_links_preserve_md(md_text: str) -> str:
    if not md_text:
        return md_text

    # 1) Split by fenced code blocks to avoid touching code sections.
    parts = re.split(r'(```.*?```)', md_text, flags=re.DOTALL)
    for i, part in enumerate(parts):
        if part.startswith("```"):
            continue  # leave code fences untouched

        # 2) Replace standard links with their anchor text.
        #    If the anchor itself looks like a URL, drop it (renders cleaner with your numeric refs).
        def repl(m: re.Match) -> str:
            anchor = m.group(1)
            # keep EXACT spacing/newlines around the match; only change the link itself
            looks_like_url = bool(re.match(r'^(https?://|www\.|[A-ZaZ0-9.-]+\.[A-ZaZ]{2,})$', anchor.strip(), re.I))
            return "" if looks_like_url else anchor

        safe = _LINK_RE.sub(repl, part)

        # 3) Remove truly empty parentheses created by link removal, but ONLY if empty.
        #    (We do *not* remove any other parentheses or compress whitespace.)
        safe = re.sub(r'\(\s*\)', '', safe)

        parts[i] = safe

    return "".join(parts)

# Run this AFTER render_with_citations_by_index(...) and your link stripping.
# It removes parenthesis groups that contain only separators/whitespace (no words),
# e.g. "(, )", "( ; )", "( / , )", "( | )", "( – )", or just "()".

_SEP_ONLY_PARENS = re.compile(
    r"""\(\s*                                   # opening paren
        (?:[,;:/|·•\-–—]|\s|&nbsp;|&\#160;)*     # only separators/whitespace
        \)                                      # closing paren
    """,
    re.VERBOSE
)

# Also remove the very common exact cases quickly (cheap pass):
def _trim_separator_artifacts(s: str) -> str:
    if not s:
        return s

    # 1) Nuke empty/sep-only parenthesis groups like "(, )", "( ; )", "()"
    s = _SEP_ONLY_PARENS.sub("", s)

    # 2) If a dangling comma was left behind right before a block end, newline, or end of string,
    #    remove that comma (avoids "... [7] [8],\n" or "... [7] [8],</p>")
    s = re.sub(r",\s*(?=(?:$|\n|\r|</p>|</li>|</div>|</ul>|</ol>))", "", s, flags=re.IGNORECASE)

    # 3) Collapse any accidental multiple spaces that appear right before punctuation
    s = re.sub(r"\s+([,.;:])", r"\1", s)

    return s



def extract_citations_from_annotations_response_dict(text_part, client_unused=None):
    """
    Robust extractor for Responses API annotations (dicts or objects).
    Expects text_part like: {"text": <str>, "annotations": <list>}
    Returns (citation_map, placements) where:
      - citation_map: {n: {number, file_name, file_id, url, title, summary}}
      - placements:   [(char_index, n), ...]
    """
    def _ga(obj, name, default=None):
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    full_text = (text_part.get("text") if isinstance(text_part, dict) else _ga(text_part, "text")) or ""
    annotations = (text_part.get("annotations") if isinstance(text_part, dict) else _ga(text_part, "annotations")) or []

    citation_map = {}
    placements = []
    if not annotations:
        return citation_map, placements

    # Normalize each annotation into a simple dict
    norm = []
    for a in annotations:
        a_type = (str((a.get("type") if isinstance(a, dict) else getattr(a, "type", "")) or "")
                .strip().lower())

        if a_type == "file_citation":
            file_id = a.get("file_id") if isinstance(a, dict) else getattr(a, "file_id", None)
            if not file_id:
                fc = a.get("file_citation") if isinstance(a, dict) else getattr(a, "file_citation", None)
                if isinstance(fc, dict):
                    file_id = fc.get("file_id")
            if not file_id:
                continue

            filename = a.get("filename") if isinstance(a, dict) else getattr(a, "filename", None)
            idx = _pick_index(a, len(full_text))
            if idx is None:
                idx = len(full_text)

            norm.append({
                "type": "file_citation",
                "file_id": file_id,
                "filename": filename,
                "index": idx
            })

        elif a_type in ("url_citation", "web_citation"):
            # sometimes payload is flat (as in your sample)
            payload = a
            # but tolerate nested shapes like a["url_citation"] = {...}
            inner = a.get("url_citation") if isinstance(a, dict) else getattr(a, "url_citation", None)
            if isinstance(inner, dict):
                payload = inner

            url = payload.get("url") if isinstance(payload, dict) else getattr(payload, "url", None)
            title = payload.get("title") if isinstance(payload, dict) else getattr(payload, "title", None)
            summary = payload.get("summary") if isinstance(payload, dict) else getattr(payload, "summary", None)

            idx = _pick_index(payload, len(full_text))
            # if both start and end exist, prefer end for placement
            if idx is None and "end_index" in payload:
                try:
                    idx = int(payload["end_index"])
                except Exception:
                    pass
            if idx is None:
                idx = len(full_text)


            norm.append({
                "type": "web_citation",   # keep your internal normalized name
                "url": url,
                "title": title,
                "summary": summary,
                "index": idx
            })


    if not norm:
        return citation_map, placements

    # Separate file and web citations for lookup
    file_norm = [n for n in norm if n["type"] == "file_citation"]
    web_norm = [n for n in norm if n["type"] == "web_citation"]

    # File citations: DB lookup
    file_ids = list({n["file_id"] for n in file_norm})
    file_data = {}
    if file_ids:
        conn = get_connection()
        file_data = get_urls_and_titles_by_file_ids(conn, file_ids)

    # Build map and placements in order of appearance
    i = 1
    for n in norm:
        if n["type"] == "file_citation":
            file_id = n["file_id"]
            filename = n["filename"] or _cached_filename(file_id)
            idx = max(0, min(len(full_text), n["index"]))  # clamp
            file_info = file_data.get(file_id, {})
            url = file_info.get("url")
            title = file_info.get("title")
            summary = file_info.get("summary")
            recordset = file_info.get("recordset")
            doc_id = file_info.get("doc_id")
            if not title:
                title = filename.rsplit(".", 1)[0]
            clean_title = html.escape(re.sub(r"^\d+_", "", title).replace("_", " "))
            citation_map[i] = {
                "number": i,
                "source": "file",
                "file_name": filename,
                "file_id": file_id,
                "url": url,
                "title": clean_title,
                "summary": summary,
                "recordset": recordset,
                "doc_id": doc_id,
            }
            placements.append((idx, i))
            i += 1
        elif n["type"] == "web_citation":
            idx = max(0, min(len(full_text), n["index"]))
            url = n["url"]
            title = n["title"] or url or "Web Source"
            summary = n["summary"] or ""
            clean_title = html.escape(title)
            citation_map[i] = {
                "number": i,
                "source": "web",
                "file_name": None,
                "file_id": None,
                "url": url,
                "title": clean_title,
                "summary": summary,
            }
            placements.append((idx, i))
            i += 1

    return citation_map, placements


def render_with_citations_by_index(text_html, citation_map, placements):
    """
    Insert <sup>[n]</sup> at the specified character indices (desc order to avoid shifts).
    Ensures that when multiple citations share the same index they appear left-to-right
    in numeric order and are separated by a space.
    """
    s = text_html or ""
    n = len(s)
    used_nums = set()
    # Sort by (index, num) and reverse so we insert at largest index first.
    # For equal indices, the higher num is inserted first so the final left-to-right
    # order of superscripts is increasing (e.g., [6] [7]).
    for idx, num in sorted(placements or [], key=lambda x: (x[0], x[1]), reverse=True):
        note = citation_map.get(num)
        if not note:
            continue
        if idx is not None and 0 <= idx <= n:
            sup = f"<sup title='Source: {note['title']}'>[{num}]</sup>"
            i = max(0, min(n, idx))
            s = s[:i] + sup + s[i:]
            used_nums.add(num)
    # For any citation not referenced (e.g., missing/invalid index), append at end
    for num, note in citation_map.items():
        if num not in used_nums:
            sup = f"<sup title='Source: {note['title']}'>[{num}]</sup>"
            s += sup
    # Ensure there's a visible space between adjacent supers if they ended up touching
    s = re.sub(r"</sup><sup", "</sup> <sup", s)
    return s


def render_sources_list(citation_map):
    if not citation_map:
        return ""

    def _icon_for_source(src):
        if src == "file":
            return DOC_ICON_HTML
        if src == "web":
            return WEB_ICON_HTML
        return ""

    lines = []
    for c in citation_map.values():
        title = c["title"] or "Untitled"
        summary = html.escape((c["summary"] or "").replace("\n", " ").strip())
        badge = f"[{c['number']}]"
        url = c.get("url")

        # Prepare recordset (plain, in front)
        rs = c.get("recordset")
        if isinstance(rs, (dict, list, tuple)):
            try:
                rs_text = json.dumps(rs, ensure_ascii=False)
            except Exception:
                rs_text = str(rs)
        else:
            rs_text = str(rs) if rs is not None else ""
        rs_text = rs_text.strip()
        rs_html = html.escape(rs_text) if rs_text else ""

        is_internal_doc = False
        if rs_text:
            is_internal_doc = rs_text.lower() == "internal documents"
        if not is_internal_doc and url and url.startswith("internal://"):
            is_internal_doc = True

        # Link rendering (keep url_with_tooltip behavior as-is unless internal document)
        safe_title = html.escape(title)
        if url and not is_internal_doc:
            if summary:
                url_with_tooltip = f'<a href="{url}" title="{summary}" target="_blank" rel="noopener noreferrer">{safe_title}</a>'
                link_part = url_with_tooltip
            else:
                link_part = f"[{safe_title}]({url})"
        elif is_internal_doc:
            target_id = c.get("doc_id") or c.get("file_id")
            if target_id:
                param_name = "doc_id" if c.get("doc_id") else "file_id"
                base_path = st.get_option("server.baseUrlPath") or ""
                if base_path and not base_path.startswith("/"):
                    base_path = "/" + base_path
                viewer_path = "document_viewer"
                viewer_url = f"{base_path}/{viewer_path}?{param_name}={quote_plus(str(target_id))}"
                link_part = (
                    f'<a href="{viewer_url}" title="{summary}" target="doc-viewer" rel="noopener noreferrer">'
                    f"{safe_title}</a>"
                )
            else:
                link_part = safe_title
        else:
            link_part = safe_title

        icon_html = _icon_for_source(c.get("source"))

        parts = [f"* {badge}"]
        if icon_html:
            parts.append(icon_html)
        if rs_html:
            parts.append(f"{rs_html}:")
        parts.append(link_part)

        lines.append(" ".join(parts))

    return "\n".join(lines)


def _extract_markdown_links(text):
    """Return list of (start_index, title, url) for markdown links [title](url) in text."""
    if not text:
        return []
    pattern = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
    results = []
    for m in pattern.finditer(text):
        try:
            start = m.start()
            title = m.group(1).strip()
            url = m.group(2).strip()
            results.append((start, title, url))
        except Exception:
            continue
    return results

# New helper: replace inline filecite markers by ordered supers based on placements
# ...existing code...
def replace_filecite_markers_with_sup(text, citation_map, placements, annotations=None):
    """
    Replace markers like: fileciteturn0file1turn0file3
    with <sup>[n]</sup> using:
      1) token -> annotation -> file_id -> citation_map number (preferred, when annotations provided)
      2) ordered placements (fallback) when tokens cannot be resolved
    Returns the text with markers replaced (or removed if no mapping found).
    """
    if not text:
        return text

    # quick map: file_id -> citation number
    fileid_to_num = {}
    for num, info in (citation_map or {}).items():
        fid = info.get("file_id")
        if fid:
            fileid_to_num[str(fid)] = int(num)

    # helpers to access annotation properties (dict or object)
    def _ga(obj, name, default=None):
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    # attempt to find a file_id for a given token by inspecting annotations
    def find_file_id_for_token(token):
        if not annotations:
            return None
        for a in annotations:
            # check a few plausible fields directly
            for key in ("id", "marker", "token", "file_id", "filename"):
                val = _ga(a, key, None)
                if val is None:
                    continue
                try:
                    sval = str(val)
                except Exception:
                    continue
                if token == sval or token in sval:
                    fid = _ga(a, "file_id", None)
                    if fid:
                        return str(fid)
            # last resort: string search across the whole annotation representation
            try:
                ann_str = json.dumps(a, ensure_ascii=False) if isinstance(a, (dict, list)) else str(a)
            except Exception:
                ann_str = str(a)
            if token in ann_str:
                fid = _ga(a, "file_id", None)
                if fid:
                    return str(fid)
        return None

    # ordered fallback iterator of placement numbers (left-to-right)
    ordered_nums_iter = iter([num for _, num in sorted(placements or [], key=lambda x: x[0])])

    # replacement function: capture inner payload of marker
    def _repl(m):
        payload = m.group(1)  # content between the markers
        tokens = [t for t in payload.split("") if t]
        parts = []
        for tok in tokens:
            num = None
            # 1) try token -> annotation -> file_id -> number
            fid = find_file_id_for_token(tok)
            if fid:
                num = fileid_to_num.get(fid)
            # 2) fallback to consuming next ordered placement number
            if num is None:
                try:
                    num = next(ordered_nums_iter)
                except StopIteration:
                    num = None
            if num is None:
                # unknown token: render a placeholder or drop it
                parts.append("<sup>[?]</sup>")
            else:
                note = citation_map.get(num)
                if note and note.get("title"):
                    parts.append(f"<sup title='Source: {note['title']}'>[{int(num)}]</sup>")
                else:
                    parts.append(f"<sup>[{int(num)}]</sup>")
        return "".join(parts)

    return re.sub(r'filecite([^]+)', _repl, text)

# Human-friendly event label helper (module level)
def human_event_label(event):
    et = getattr(event, "type", "") or ""
    name = (getattr(event, "name", None) or getattr(event, "tool", None) or "").lower()
    if "web_search" in et or "web_search" in name or "web_search" in str(getattr(event, "action", "")).lower():
        if "start" in et or et.endswith("_start") or ".start" in et:
            return "Searching the web…"
        if "complete" in et or et.endswith("_complete") or ".complete" in et:
            return "Web search completed…"
        return "Searching the web…"
    if "file_search" in et or "file_search" in name or "file_search" in str(getattr(event, "queries", "")).lower():
        if "start" in et or et.endswith("_start") or ".start" in et:
            return "Searching my knowledge base…"
        if "complete" in et or et.endswith("_complete") or ".complete" in et:
            return "Knowledge base search completed…"
        return "Searching my knowledge base…"
    if "tool_call" in et or et.endswith("_call") or "tool" in et:
        short = et.split(".")[-1] if "." in et else et
        return f"Using Tools {short.replace('_', ' ')}"
    if et.startswith("response.output_text"):
        if et.endswith("start"):
            return "Generating answer…"
        if et.endswith("complete"):
            return "Answer generation finished"
        return "Generating answer…"
    return None

# ---------- Helpers to robustly read final.output (dicts or objects) ----------
def _get_attr(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)

# Extract usage fields from final response (for logging/costs)
def _get_usage_from_final(final_obj):
    """Best-effort extraction of usage fields from a Responses API object or dict."""
    def ga(o, name, default=None):
        if isinstance(o, dict):
            return o.get(name, default)
        return getattr(o, name, default)
    usage = ga(final_obj, "usage")
    if usage is None and isinstance(final_obj, dict):
        usage = final_obj.get("usage")
    input_tokens = ga(usage, "input_tokens", None) if usage is not None else None
    if input_tokens is None:
        input_tokens = ga(usage, "prompt_tokens", None) if usage is not None else None
    output_tokens = ga(usage, "output_tokens", None) if usage is not None else None
    total_tokens = ga(usage, "total_tokens", None) if usage is not None else None
    # reasoning tokens may be nested
    reasoning_tokens = None
    otd = ga(usage, "output_tokens_details", None) if usage is not None else None
    if otd is not None:
        reasoning_tokens = ga(otd, "reasoning_tokens", None)
    return (
        input_tokens if isinstance(input_tokens, int) else 0,
        output_tokens if isinstance(output_tokens, int) else 0,
        total_tokens if isinstance(total_tokens, int) else ((input_tokens or 0) + (output_tokens or 0)),
        reasoning_tokens if isinstance(reasoning_tokens, int) else 0,
    )

# Iterate items/parts from Responses API final object
def _iter_content_items(final):
    output = _get_attr(final, "output", []) or []
    for item in output:
        yield item

def _iter_content_parts(item):
    content = _get_attr(item, "content", []) or []
    for part in content:
        yield part

def coerce_text_part(part):
    """
    Normalize a content part to: {"text": <str>, "annotations": <list>}
    Handles:
      - {"type":"output_text","text":"...", "annotations":[...]}
      - {"type":"output_text","text":{"value":"...","annotations":[...]}}
      - pydantic objects with .type, .text(.value/.annotations) or .annotations
    """
    ptype = _get_attr(part, "type", None)
    if ptype not in ("output_text", "text"):
        return None

    text_field = _get_attr(part, "text", None)

    # A) text is a plain string
    if isinstance(text_field, str):
        annotations = _get_attr(part, "annotations", []) or []
        return {"text": text_field, "annotations": annotations}

    # B) text is an object/dict with .value and .annotations
    if text_field is not None:
        value = _get_attr(text_field, "value", None)
        ann = _get_attr(text_field, "annotations", []) or []
        if isinstance(value, str):
            return {"text": value, "annotations": ann}

    # C) Fallback
    text_str = "" if text_field is None else (text_field if isinstance(text_field, str) else str(text_field))
    annotations = _get_attr(part, "annotations", []) or []
    return {"text": text_str, "annotations": annotations}

# ---------- Streamed handling ----------
def handle_stream_and_render(user_input, system_instructions, client, retrieval_filters=None, debug_one=False, web_tool_extras=None):
    """
    Responses API (streaming) with status indicator to avoid chat duplication:
      - opens the assistant bubble immediately
      - shows status text beside avatar (no spinner to avoid duplication)
      - streams tokens into same bubble
      - parses final output, inserts index-based citations, shows sources
      - logs to DB
    """
    if "VECTOR_STORE_ID" not in st.secrets:
        st.error("Missing VECTOR_STORE_ID in Streamlit secrets. Please add it to run File Search.")
        st.stop()

    # Build tools with optional web_search based on settings
    tool_cfg = [
        {
            "type": "file_search",
            "vector_store_ids": [st.secrets["VECTOR_STORE_ID"]],
        }
    ]
    web_enabled = True
    if web_tool_extras and isinstance(web_tool_extras, dict):
        web_enabled = bool(web_tool_extras.get("web_search_enabled", True))
    
    # Add DBIS MCP tool if configured
    dbis_mcp_url = (
        str(st.secrets.get(DBIS_MCP_SERVER_URL_KEY, "") or os.getenv(DBIS_MCP_SERVER_URL_KEY, "")).strip()
    ) or None
    dbis_mcp_auth = st.secrets.get(DBIS_MCP_AUTH_KEY)
    if dbis_mcp_auth is None:
        dbis_mcp_auth = os.getenv(DBIS_MCP_AUTH_KEY)
    dbis_mcp_headers_raw = st.secrets.get(DBIS_MCP_HEADERS_KEY)
    if dbis_mcp_headers_raw is None:
        dbis_mcp_headers_raw = os.getenv(DBIS_MCP_HEADERS_KEY)
    dbis_mcp_headers = _parse_headers_setting(dbis_mcp_headers_raw)
    if dbis_mcp_headers_raw and dbis_mcp_headers is None:
        print(
            "Warning: DBIS_MCP_HEADERS could not be parsed into a dictionary. "
            "Provide JSON (e.g. '{\"Authorization\": \"Bearer ...\"}') or 'Header: value' format.",
            flush=True,
        )

    # Ensure the organization id is transmitted to the MCP server via header
    try:
        _dbis_org = st.secrets.get("DBIS_ORGANIZATION_ID")
        if _dbis_org is None:
            _dbis_org = os.getenv("DBIS_ORGANIZATION_ID")
        if _dbis_org:
            if dbis_mcp_headers is None:
                dbis_mcp_headers = {}
            # Do not overwrite if user provided explicitly
            dbis_mcp_headers.setdefault("X-DBIS-Organization-Id", str(_dbis_org))
    except Exception:
        pass

    last_url = st.session_state.get("dbis_mcp_last_url")
    if dbis_mcp_url and dbis_mcp_url != last_url:
        st.session_state["dbis_mcp_last_url"] = dbis_mcp_url
        st.session_state.pop("dbis_mcp_disabled", None)

    dbis_disabled = st.session_state.get("dbis_mcp_disabled", False)
    dbis_mcp_command = os.getenv(DBIS_MCP_ENV_KEY)

    print(DBIS_MCP_ENV_KEY, dbis_mcp_command)
    # if dbis_mcp_url:
    #     print(DBIS_MCP_SERVER_URL_KEY, dbis_mcp_url)

    if dbis_disabled:
        print(
            "DBIS MCP connector disabled for this session due to previous errors.",
            flush=True,
        )

    # Preflight MCP reachability/auth once per session or when URL changes
    if dbis_mcp_url and not dbis_disabled:
        ok, reason = _mcp_preflight(dbis_mcp_url, dbis_mcp_headers, dbis_mcp_auth)
        if not ok:
            st.session_state["dbis_mcp_disabled"] = True
            warn = "⚠️ DBIS MCP is disabled for this session: "
            if reason == "unauthorized":
                warn += "server responded with 401/403 (check DBIS_MCP_AUTHORIZATION/DBIS_MCP_HEADERS)."
            elif reason and reason.startswith("server-error"):
                warn += f"server error ({reason.split('-')[-1]})."
            else:
                warn += "server is unreachable."
            try:
                st.warning(warn)
            except Exception:
                print(warn, flush=True)
            dbis_disabled = True

    if not dbis_disabled:
        if dbis_mcp_url:
            dbis_tool_cfg = {
                "type": "mcp",
                "server_label": DBIS_MCP_SERVER_LABEL,
                "server_url": dbis_mcp_url,
                "allowed_tools": [
                    "dbis_top_resources",
                    "dbis_list_subjects",
                    "dbis_list_resource_ids",
                    "dbis_get_resource",
                    "dbis_list_resource_ids_by_subject",
                ],
                "require_approval": "never",
            }
            if dbis_mcp_auth:
                dbis_tool_cfg["authorization"] = str(dbis_mcp_auth)
            if dbis_mcp_headers:
                dbis_tool_cfg["headers"] = dbis_mcp_headers
            tool_cfg.append(dbis_tool_cfg)
        elif dbis_mcp_command:
            print(
                "DBIS MCP command is configured but no DBIS_MCP_SERVER_URL was provided. "
                "Skipping MCP tool registration because the OpenAI API requires an accessible server URL.",
                flush=True,
            )
            
    # Add web_search when enabled; filters are attached only if present
    if web_enabled:
        tool_cfg.append({"type": "web_search"})

    if retrieval_filters is not None:
        # Apply filters only to the web search tool
        for tool in tool_cfg:
            if tool.get("type") == "web_search":
                tool["filters"] = retrieval_filters
                break    
    # Attach user_location if provided
    if web_tool_extras and isinstance(web_tool_extras, dict):
        ul = web_tool_extras.get("user_location")
        if ul:
            for tool in tool_cfg:
                if tool.get("type") == "web_search":
                    tool["user_location"] = ul
                    break

    # Build conversation context (last 10 exchanges to stay within limits)
    def build_conversation_context():
        conversation = [
            {"role": "system", "content": system_instructions},
        ]
        # Instruct the model to include the org ID when invoking DBIS MCP tools
        try:
            _dbis_org = os.getenv("DBIS_ORGANIZATION_ID") or str(st.secrets.get("DBIS_ORGANIZATION_ID", "")).strip()
        except Exception:
            _dbis_org = None
        print("DBIS_ORGANIZATION_ID:", _dbis_org)
        if _dbis_org:
            conversation.append({
                "role": "system",
                "content": (
                    f"When calling any DBIS MCP tools (dbis_*), always include the argument "
                    f"organization_id='{_dbis_org}'. If organization_id is omitted, default to '{_dbis_org}'."
                )
            })
        
        # Add recent conversation history (each user has their own st.session_state)
        recent_messages = st.session_state.messages[-10:] if st.session_state.messages else []
        
        for msg in recent_messages:
            if msg["role"] == "user":
                conversation.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                # Use raw_text if available (cleaner), otherwise rendered
                content = msg.get("raw_text", msg.get("rendered", msg.get("content", "")))
                # Remove HTML tags for cleaner context
                content = re.sub(r'<[^>]+>', '', content) if content else ""
                conversation.append({"role": "assistant", "content": content})
        
        # Add current user input
        conversation.append({"role": "user", "content": user_input})
        return conversation

    conversation_input = build_conversation_context()

    # Safe inits to avoid "referenced before assignment"
    buf = ""
    final = None
    rendered = ""
    sources_md = ""
    citation_map = {}
    cleaned = ""
    response_text = ""
    # Track evaluation call usage to include in totals/cost
    eval_input_tok = 0
    eval_output_tok = 0
    eval_total_tok = 0
    eval_reasoning_tok = 0

    # Assistant bubble first so spinner sits next to avatar
    with st.chat_message("assistant", avatar= AVATAR_ASSISTANT):
        # Create status row first (spinner + status text on same line)
        status_row = st.container()
        with status_row:
            col_spinner, col_status = st.columns([0.02, 0.98])
            with col_spinner:
                spinner_ctx = st.spinner("")  # minimal spinner, no text
                spinner_ctx.__enter__()
            with col_status:
                status_placeholder = st.empty()
                try:
                    status_placeholder.markdown("_Starting…_")
                except Exception:
                    pass
        
        # Content placeholder below the status row
        content_placeholder = st.empty()
        buf = ""
        final = None
        tool_event_shown = False

        try:
            # Get current LLM configuration from database
            llm_config = get_current_llm_config()
            
            # Build API parameters conditionally based on model support
            api_params = {
                'model': llm_config['model'],
                'input': conversation_input,
                'tools': tool_cfg,
                'parallel_tool_calls': llm_config['parallel_tool_calls']
            }
            
            # Add reasoning effort for supported models
            if supports_reasoning_effort(llm_config['model']):
                api_params['reasoning'] = {'effort': llm_config['reasoning_effort']}
            
            # Add text verbosity (all models support at least 'medium')
            api_params['text'] = {'verbosity': llm_config['text_verbosity']}
            
            # Measure latency of main API call
            t_start = datetime.now()
            with client.responses.stream(timeout=30, **api_params) as stream:
                for event in stream:
                    # Keep spinner until the model is done emitting text.
                    if event.type == "response.output_text.complete":
                        try:
                            spinner_ctx.__exit__(None, None, None)
                        except Exception:
                            pass

                    # Always show a concise human-friendly status when available.
                    # In normal mode we only show the friendly label; debug_one toggles verbose info.
                    try:
                        label = human_event_label(event)
                        if label:
                            try:
                                status_placeholder.markdown(f"_{label}_")
                            except Exception:
                                pass
                        elif debug_one:
                            status_placeholder.markdown(f"`event.type` = `{event.type}`")
                    except Exception:
                        # best-effort: avoid breaking stream on status failures
                        if debug_one:
                            print("status update failed for event:", event.type)

                    # If verbose debug requested, show more detailed per-tool phases
                    if debug_one and event.type and ("tool_call" in event.type or event.type.endswith("_call") or "tool" in event.type):
                        tool_name = getattr(event, "name", None) or getattr(event, "tool", None) or event.type.split(".")[-1]
                        phase = event.type.split(".")[-1]
                        try:
                            status_placeholder.markdown(f"_{tool_name}: {phase}_")
                        except Exception:
                            pass
                        tool_event_shown = True
                        continue

                    # Stream tokens
                    if event.type == "response.output_text.delta":
                        buf += event.delta
                        content_placeholder.markdown(buf, unsafe_allow_html=True)
                        continue

                    # clear the status when text ends (keep UI minimal)
                    if event.type == "response.output_text.complete":
                        status_placeholder.empty()
                        continue

                # fetch the full object for your citation pass later
                final = stream.get_final_response()
            t_end = datetime.now()
            main_latency_ms = int((t_end - t_start).total_seconds() * 1000)

        except (BadRequestError, APIConnectionError, RateLimitError, APIError) as e:
            status_code = getattr(getattr(e, "response", None), "status_code", None)
            error_text = getattr(e, "message", "") or str(e)
            mcp_tool_present = any(tool.get("type") == "mcp" for tool in tool_cfg)
            auth_error = "401" in error_text or "Unauthorized" in error_text
            is_mcp_tool_error = (
                status_code == 424
                and "Error retrieving tool list" in error_text
                and mcp_tool_present
            )
            fallback_api_params = None

            if is_mcp_tool_error:
                st.session_state["dbis_mcp_disabled"] = True
                warning_msg = (
                    "⚠️ DBIS tools are temporarily disabled because the MCP server "
                    "could not be reached. The chat will continue without DBIS access."
                )
                if auth_error:
                    warning_msg += " (Server responded with 401 Unauthorized—check MCP credentials.)"
                try:
                    st.warning(warning_msg)
                except Exception:
                    print(warning_msg, flush=True)
                tool_cfg = [tool for tool in tool_cfg if tool.get("type") != "mcp"]
                # Prepare fallback params without MCP tools so the non-streaming retry succeeds
                llm_config = get_current_llm_config()
                fallback_api_params = {
                    'model': llm_config['model'],
                    'input': conversation_input,
                    'tools': tool_cfg,
                    'parallel_tool_calls': llm_config['parallel_tool_calls']
                }
                if supports_reasoning_effort(llm_config['model']):
                    fallback_api_params['reasoning'] = {'effort': llm_config['reasoning_effort']}
                fallback_api_params['text'] = {'verbosity': llm_config['text_verbosity']}

            print(f"❌ OpenAI API error during streaming: {e}", flush=True)
            # Ensure spinner is closed before fallback/error
            try:
                spinner_ctx.__exit__(None, None, None)
            except Exception:
                pass

            if isinstance(e, (BadRequestError, APIError)):
                # Fallback: non-streaming request rendered inside same bubble
                with st.spinner("Thinking…"):
                    # Get current LLM configuration from database
                    if fallback_api_params is None:
                        llm_config = get_current_llm_config()
                        # Build API parameters conditionally based on model support
                        fallback_api_params = {
                            'model': llm_config['model'],
                            'input': conversation_input,
                            'tools': tool_cfg,
                            'parallel_tool_calls': llm_config['parallel_tool_calls']
                        }
                        if supports_reasoning_effort(llm_config['model']):
                            fallback_api_params['reasoning'] = {'effort': llm_config['reasoning_effort']}
                        fallback_api_params['text'] = {'verbosity': llm_config['text_verbosity']}
                    t_start = datetime.now()
                    final = client.responses.create(timeout=30, **fallback_api_params)
                    t_end = datetime.now()
                    main_latency_ms = int((t_end - t_start).total_seconds() * 1000)
                    buf = _get_attr(final, "output_text", "") or ""
                    content_placeholder.markdown(buf, unsafe_allow_html=True)
            elif isinstance(e, RateLimitError):
                st.error("We’re being rate-limited right now. Please try again in a moment.")
                return
            else:
                st.error(f"❌ OpenAI connection error: {e}")
                return

        finally:
            # If no event arrived, ensure spinner is closed
            try:
                spinner_ctx.__exit__(None, None, None)
            except Exception:
                pass


        # --- Post-process final for citations & re-render ---
        if not final:
            st.warning("I couldn’t generate a response this time.")
            return

        response_text = _get_attr(final, "output_text", "") or buf

        # Capture usage after final is available
        input_tok, output_tok, total_tok, reasoning_tok = _get_usage_from_final(final)

        # Find and normalize the first text-bearing content part (dict or object)
        normalized_part = None
        for item in _iter_content_items(final):
            if _get_attr(item, "type") == "message":
                for part in _iter_content_parts(item):
                    maybe = coerce_text_part(part)  # -> {"text": "...", "annotations": [...]}
                    if maybe:
                        normalized_part = maybe
                        break
            if normalized_part:
                break

        # Build citations (index-based) and re-render with superscripts
        if normalized_part:
            citation_map, placements = extract_citations_from_annotations_response_dict(normalized_part)
        else:
            citation_map, placements = ({}, [])


        # If we have the normalized text part (canonical, no inline markers), use it as the base
        if normalized_part:
            cleaned = normalized_part.get("text", "").strip()
            # Detect inline web citations (markdown links) and add them to citation_map
            # Pattern: [text](http...) or [text](https...)
            md_links = _extract_markdown_links(normalized_part.get("text", "") or "")
            # If annotations didn't provide web citations, use inline links as fallback
            # Rebuild the text so links are replaced by their anchor text and compute placements
            if md_links and not any(v.get("url") for v in citation_map.values()):
                # Build a new cleaned string where each markdown link is replaced by its anchor text
                text_orig = normalized_part.get("text", "") or ""
                rebuilt = []
                placements_map = []  # (char_index_in_rebuilt, title, url)
                last = 0
                pattern = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
                # We'll scan for either grouped parentheses containing multiple links,
                # e.g. "( [a](url1), [b](url2), [c](url3) )", or single links.
                # This ensures we remove surrounding parentheses and any commas between links.
                i = 0
                while True:
                    m = pattern.search(text_orig, i)
                    if not m:
                        break

                    # look left for a '(' (skip whitespace)
                    pos_left = m.start() - 1
                    while pos_left >= 0 and text_orig[pos_left].isspace():
                        pos_left -= 1
                    has_wrap_left = (pos_left >= 0 and text_orig[pos_left] == "(")

                    post_advance = m.end()
                    group_links = [m]

                    # if there's a '(' before this link, try to see if this is a comma-separated group
                    group_end = None
                    if has_wrap_left:
                        scan_pos = m.end()
                        while True:
                            # look for next link after scan_pos
                            nm = pattern.search(text_orig, scan_pos)
                            if not nm:
                                break
                            # find next non-space char after nm.end()
                            temp = nm.end()
                            while temp < len(text_orig) and text_orig[temp].isspace():
                                temp += 1
                            if temp < len(text_orig) and text_orig[temp] == ",":
                                # part of same group; continue scanning
                                group_links.append(nm)
                                last_match = nm
                                scan_pos = temp + 1
                                continue
                            # if the next non-space char is ')', we ended a group
                            if temp < len(text_orig) and text_orig[temp] == ")":
                                group_links.append(nm)
                                last_match = nm
                                group_end = temp
                                break
                            # otherwise not a grouped parenthesis sequence
                            break

                    if group_end is not None:
                        # We found a parenthesized group of links from pos_left .. group_end
                        pre = text_orig[last:pos_left]
                        rebuilt.append(pre)
                        # iterate each link in the group and append only anchor-display (or empty)
                        for idx_link, lm in enumerate(group_links):
                            anchor = lm.group(1).strip()
                            url = lm.group(2).strip()
                            looks_like_url = bool(re.match(r'^(https?://|www\.|[A-ZaZ0-9.-]+\.[A-ZaZ]{2,})$', anchor.strip(), re.I))
                            display_anchor = "" if looks_like_url else anchor
                            start_idx = sum(len(p) for p in rebuilt)
                            rebuilt.append(display_anchor)
                            placements_map.append((start_idx, anchor, url))
                            # add a single separating space between anchors so supers don't concatenate
                            if idx_link != len(group_links) - 1:
                                rebuilt.append(" ")
                        last = group_end + 1
                        i = last
                        continue

                    # not a grouped parenthesis case — treat single link normally
                    pos_right = m.end()
                    while pos_right < len(text_orig) and text_orig[pos_right].isspace():
                        pos_right += 1
                    has_wrap_right = (pos_right < len(text_orig) and text_orig[pos_right] == ")")
                    if has_wrap_left and has_wrap_right:
                        pre = text_orig[last:pos_left]
                        post_advance = pos_right + 1
                    else:
                        pre = text_orig[last:m.start()]
                        post_advance = m.end()

                    rebuilt.append(pre)
                    anchor = m.group(1).strip()
                    url = m.group(2).strip()
                    looks_like_url = bool(re.match(r'^(https?://|www\.|[A-ZaZ0-9.-]+\.[A-ZaZ]{2,})$', anchor.strip(), re.I))
                    display_anchor = "" if looks_like_url else anchor
                    start_idx = sum(len(p) for p in rebuilt)
                    rebuilt.append(display_anchor)
                    placements_map.append((start_idx, anchor, url))
                    last = post_advance
                    i = last
            
                rebuilt.append(text_orig[last:])
                cleaned = "".join(rebuilt).strip()

                # Add to citation_map using positions computed in rebuilt text
                next_num = max([int(k) for k in citation_map.keys()], default=0) + 1
                for start_idx, title, url in placements_map:
                    clean_title = html.escape(title)
                    citation_map[next_num] = {
                        "number": next_num,
                        "file_name": None,
                        "file_id": None,
                        "url": url,
                        "title": clean_title,
                        "summary": "",
                    }
                    placements.append((max(0, min(len(cleaned), start_idx)), next_num))
                    next_num += 1
            else:
                # No inline link rebuild needed; use original cleaned text
                cleaned = normalized_part.get("text", "").strip()
            rendered = render_with_citations_by_index(cleaned, citation_map, placements)

            has_url_annotations = any(
                (a.get("type") if isinstance(a, dict) else getattr(a, "type", "")) in ("url_citation","web_citation")
                for a in (normalized_part.get("annotations") or [])
            )

            if has_url_annotations:
                rendered = _strip_markdown_links_preserve_md(rendered)
                
            # 🔧 NEW: remove leftover "(, )" (and similar) artifacts
            rendered = _trim_separator_artifacts(rendered)

        else:
            # Fallback: we only have output_text which may contain inline filecite markers.
            # Replace markers in-order with supers based on placements (best-effort).
            cleaned = response_text.strip()
            # For fallback raw text: rebuild so links are replaced by anchor text and compute placements
            text_orig = cleaned
            pattern = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
            rebuilt = []
            placements_map = []
            last = 0
            i = 0
            while True:
                m = pattern.search(text_orig, i)
                if not m:
                    break

                # look left for '(' skipping whitespace
                pos_left = m.start() - 1
                while pos_left >= 0 and text_orig[pos_left].isspace():
                    pos_left -= 1
                has_wrap_left = (pos_left >= 0 and text_orig[pos_left] == "(")

                post_advance = m.end()
                group_links = [m]
                group_end = None
                if has_wrap_left:
                    scan_pos = m.end()
                    while True:
                        nm = pattern.search(text_orig, scan_pos)
                        if not nm:
                            break
                        temp = nm.end()
                        while temp < len(text_orig) and text_orig[temp].isspace():
                            temp += 1
                        if temp < len(text_orig) and text_orig[temp] == ",":
                            group_links.append(nm)
                            scan_pos = temp + 1
                            continue
                        if temp < len(text_orig) and text_orig[temp] == ")":
                            group_links.append(nm)
                            group_end = temp
                        break

                if group_end is not None:
                    # emit pre, then each link anchor (no commas), separated by single spaces
                    pre = text_orig[last:pos_left]
                    rebuilt.append(pre)
                    for idx_link, lm in enumerate(group_links):
                        anchor = lm.group(1).strip()
                        url = lm.group(2).strip()
                        looks_like_url = bool(re.match(r'^(https?://|www\.|[A-ZaZ0-9.-]+\.[A-Za:z]{2,})$', anchor.strip(), re.I))
                        display_anchor = "" if looks_like_url else anchor
                        start_idx = sum(len(p) for p in rebuilt)
                        rebuilt.append(display_anchor)
                        placements_map.append((start_idx, anchor, url))
                        if idx_link != len(group_links) - 1:
                            rebuilt.append(" ")
                    last = group_end + 1
                    i = last
                    continue

                # single link fallback
                pos_right = m.end()
                while pos_right < len(text_orig) and text_orig[pos_right].isspace():
                    pos_right += 1
                has_wrap_right = (pos_right < len(text_orig) and text_orig[pos_right] == ")")
                if has_wrap_left and has_wrap_right:
                    pre = text_orig[last:pos_left]
                    post_advance = pos_right + 1
                else:
                    pre = text_orig[last:m.start()]
                    post_advance = m.end()

                rebuilt.append(pre)
                anchor = m.group(1).strip()
                url = m.group(2).strip()
                looks_like_url = bool(re.match(r'^(https?://|www\.|[A-ZaZ0-9.-]+\.[A-ZaZ]{2,})$', anchor.strip(), re.I))
                display_anchor = "" if looks_like_url else anchor
                start_idx = sum(len(p) for p in rebuilt)
                rebuilt.append(display_anchor)
                placements_map.append((start_idx, anchor, url))
                last = post_advance
                i = last
            rebuilt.append(text_orig[last:])
            cleaned = "".join(rebuilt).strip()

            if placements_map:
                next_num = max([int(k) for k in citation_map.keys()], default=0) + 1
                for start_idx, title, url in placements_map:
                    clean_title = html.escape(title)
                    citation_map[next_num] = {
                        "number": next_num,
                        "file_name": None,
                        "file_id": None,
                        "url": url,
                        "title": clean_title,
                        "summary": "",
                    }
                    placements.append((max(0, min(len(cleaned), start_idx)), next_num))
                    next_num += 1

            rendered = replace_filecite_markers_with_sup(cleaned, citation_map, placements, annotations=None)

        # Safety cleanup: remove any remaining filecite markers that might have been missed
        # Match the full marker like: filecite... (non-greedy, DOTALL so newlines are covered)
        rendered = re.sub(r'filecite.*?', '', rendered, flags=re.DOTALL)
        # Fallback: remove any leftover ascii-ish remnants ("filecite..." or orphan tokens)
        rendered = re.sub(r'filecite[^\s]*', '', rendered)
        # Also remove any orphaned citation tokens that might be left behind
        rendered = re.sub(r'\bturn\d+file\d+(?:turn\d+file\d+)*\b', '', rendered)

        sources_md = render_sources_list(citation_map)
        

        # Overwrite the streamed text in the SAME bubble with the enriched version
        content_placeholder.markdown(rendered, unsafe_allow_html=True)
        if sources_md:
            with st.expander("References", icon=":material/info:", expanded=False):
                st.markdown(sources_md, unsafe_allow_html=True)

        # --- Optional debug dump of final (sidebar toggle) ---
        if debug_one and final:
            with st.expander("Debug: final.model_dump()", expanded=False):
                try:
                    dump = final.model_dump()
                    st.json(dump)
                except Exception:
                    # Fallback if model_dump isn't available
                    st.code(_redact_ids(repr(final)), language="json")

    # Persist chat history and log
    st.session_state.messages.append({
        "role": "assistant",
        "raw_text": response_text,
        "rendered": rendered,
        "sources": sources_md,
        "citation_json": json.dumps(citation_map, ensure_ascii=False) if citation_map else None,
    })

    # Keep memory footprint bounded
    MAX_HISTORY = 50
    if len(st.session_state.messages) > MAX_HISTORY:
        st.session_state.messages = st.session_state.messages[-MAX_HISTORY:]

    # --- Structured evaluation follow-up (request classification + response evaluation) ---
    request_classification = "other"
    evaluation_notes = ""
    confidence = 0.0
    error_code = None
    try:
        try:
            evaluation_system_prompt = EVALUATION_SYSTEM_TEMPLATE.format(
                citation_count=len(citation_map),
                allowed_topics=", ".join(get_allowed_request_classifications()),
            )
        except KeyError as exc:
            raise RuntimeError(f"Evaluation prompt template missing placeholder: {exc}") from exc

        # Get current LLM configuration from database
        llm_config = get_current_llm_config()
        structured = client.responses.create(
            timeout=30,
            model=llm_config['model'],
            input=[
                {"role": "system", "content": evaluation_system_prompt},
                {"role": "user", "content": f"Original user request: {user_input}"},
                {"role": "assistant", "content": f"Assistant response: {cleaned}"},
                {"role": "user", "content": "Please evaluate this interaction and return the JSON evaluation."}
            ]
        )

        # Capture evaluation usage to include in totals/cost
        try:
            e_in, e_out, e_total, e_reason = _get_usage_from_final(structured)
            eval_input_tok = int(e_in or 0)
            eval_output_tok = int(e_out or 0)
            eval_total_tok = int(e_total or 0)
            eval_reasoning_tok = int(e_reason or 0)
        except Exception:
            pass

        payload_text = _safe_output_text(structured)
        if not payload_text:
            st.warning("No evaluation payload returned from structured call.")
        else:
            # Show evaluation when debugging
            if debug_one:
                with st.expander("🔎 Response Evaluation (JSON)", expanded=False):
                    st.code(payload_text, language="json")

            payload = json.loads(payload_text)
            confidence = float(payload.get("confidence", 0.0))
            ec = payload.get("error_code")
            error_code = f"E0{ec}" if ec is not None else None
            request_classification = payload.get("request_classification", "other")
            evaluation_notes = payload.get("evaluation_notes", "")

    except BadRequestError as e:
        st.error(f"Evaluation call failed (BadRequest): {e}")
    except RateLimitError as e:
        st.error("Evaluation call rate-limited. Please try again shortly.")
    except APIConnectionError as e:
        st.error(f"Evaluation call connection error: {e}")
    except APIError as e:
        st.error(f"Evaluation call server error: {e}")
    except Exception as e:
        if debug_one:
            st.error(f"Evaluation call unexpected error: {e}")
        # Silently continue with default values for production

    try:
        # Debug: Check session ID before logging
        print(f"🔎 About to log interaction with session_id: {st.session_state.session_id}")
        # Determine model (from current config)
        llm_config = get_current_llm_config()

        # Include evaluation usage in totals before computing cost
        input_tok = (input_tok or 0) + eval_input_tok
        output_tok = (output_tok or 0) + eval_output_tok
        total_tok = (total_tok or 0) + eval_total_tok
        reasoning_tok = (reasoning_tok or 0) + eval_reasoning_tok

        cost_usd = estimate_cost_usd(llm_config['model'], input_tok, output_tok)
        log_interaction_async(
            user_input=user_input,
            assistant_response=cleaned,
            session_id=st.session_state.session_id,
            citation_json=json.dumps(citation_map, ensure_ascii=False) if citation_map else None,
            citation_count=len(citation_map),
            confidence=confidence,
            error_code=error_code,
            request_classification=request_classification,
            evaluation_notes=evaluation_notes,
            model=llm_config['model'],
            usage_input_tokens=input_tok,
            usage_output_tokens=output_tok,
            usage_total_tokens=total_tok,
            usage_reasoning_tokens=reasoning_tok,
            api_cost_usd=cost_usd,
            response_time_ms=main_latency_ms if 'main_latency_ms' in locals() else None,
        )
    except Exception as log_error:
        # Silently fail for logging - don't interrupt user experience
        print(f"Warning: Failed to log interaction: {log_error}")

# ---------- App init ----------
# Check for required OpenAI API key
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except KeyError:
    st.error("🔑 **OpenAI API Key Required**")
    st.info("""
    Please add your OpenAI API key to Streamlit Cloud settings:
    
    1. **Go to your app settings** in Streamlit Cloud
    2. **Add this secret**:
       ```
       OPENAI_API_KEY = "your-openai-api-key-here"
       ```
    3. **Get an API key** from [OpenAI Platform](https://platform.openai.com/api-keys)
    4. **Redeploy the app**
    
    The app cannot function without an OpenAI API key.
    """)
    st.stop()

dbis_cmd = st.secrets.get(DBIS_MCP_ENV_KEY)
dbis_url_secret = st.secrets.get(DBIS_MCP_SERVER_URL_KEY)
if dbis_cmd and not dbis_url_secret:
    os.environ.setdefault(DBIS_MCP_ENV_KEY, dbis_cmd)
if dbis_url_secret:
    os.environ.setdefault(DBIS_MCP_SERVER_URL_KEY, str(dbis_url_secret))
dbis_auth_secret = st.secrets.get(DBIS_MCP_AUTH_KEY)
if dbis_auth_secret:
    os.environ.setdefault(DBIS_MCP_AUTH_KEY, str(dbis_auth_secret))
dbis_headers_secret = st.secrets.get(DBIS_MCP_HEADERS_KEY)
if dbis_headers_secret:
    if isinstance(dbis_headers_secret, dict):
        os.environ.setdefault(DBIS_MCP_HEADERS_KEY, json.dumps(dbis_headers_secret))
    else:
        os.environ.setdefault(DBIS_MCP_HEADERS_KEY, str(dbis_headers_secret))

dbis_org = st.secrets.get("DBIS_ORGANIZATION_ID")
if dbis_org:
    os.environ.setdefault("DBIS_ORGANIZATION_ID", str(dbis_org))
# Helper function to get current LLM configuration
def get_current_llm_config():
    """Get current LLM settings from database with fallback to defaults"""
    try:
        settings = get_llm_settings()
        return {
            'model': settings['model'],
            'parallel_tool_calls': settings['parallel_tool_calls'],
            'reasoning_effort': settings.get('reasoning_effort', 'medium'),
            'text_verbosity': settings.get('text_verbosity', 'medium')
        }
    except Exception:
        # Fallback to defaults if database not available
        return {
            'model': st.secrets.get("MODEL", "gpt-4o-mini"),
            'parallel_tool_calls': True,
            'reasoning_effort': 'medium',
            'text_verbosity': 'medium'
        }

# Initialize database and tables
database_available = False
try:
    database_available = create_database_if_not_exists()
    if database_available:
        create_knowledge_base_table()
        initialize_default_prompt_if_empty(DEFAULT_PROMPT)
        create_llm_settings_table()  # Initialize LLM settings table
        create_request_classifications_table()  # Initialize request classifications
        create_filter_settings_table()  # Initialize filter settings table
        #print("✅ Database initialization completed successfully.")
    else:
        # Show database configuration instructions when no database is available
        st.warning("🔧 **Database Configuration Needed**")
        st.info("""
        To enable full functionality, please configure a PostgreSQL database:
        
        1. **Create a cloud PostgreSQL database**
            Free Cloud PostgreSQL Options:
            - [Neon.tech](https://neon.tech) - 500MB free, serverless PostgreSQL
            - [Supabase](https://supabase.com) - 500MB free, includes real-time features
            - [ElephantSQL](https://elephantsql.com) - 20MB free "Tiny Turtle" plan
            - [Railway](https://railway.app) - PostgreSQL with free tier
        2. **Add database secrets** in Streamlit Cloud settings:
           ```
           [postgres]
           host = "your-postgres-host"
           port = "5432"
           database = "your-database-name"
           user = "your-username"
           password = "your-password"
           ```
        3. **Redeploy the app**
        
        The chat functionality will work without database, but logging and admin features will be disabled.
        """)
        print("⚠️ Database not available. Continuing without database functionality.")
except Exception as e:
    error_str = str(e)
    # Check for localhost/missing database OR missing postgres configuration entirely
    if ("localhost" in error_str or "127.0.0.1" in error_str or 
        "postgres" in error_str or "Missing PostgreSQL" in error_str or
        ("KeyError" in error_str and "postgres" in error_str.lower())):
        st.warning("🔧 **Database Configuration Needed**")
        st.info("""
        To enable full functionality, please configure a PostgreSQL database:
        
        1. **Create a cloud PostgreSQL database** (e.g., Neon.tech, Supabase, or ElephantSQL)
        2. **Add database secrets** in Streamlit Cloud settings:
           ```
           [postgres]
           host = "your-postgres-host"
           port = "5432"
           database = "your-database-name"
           user = "your-username"
           password = "your-password"
           ```
        3. **Redeploy the app**
        
        The chat functionality will work without database, but logging and admin features will be disabled.
        """)
    else:
        st.error(f"Database initialization failed: {e}")
        st.warning("The app may have limited functionality without database access.")

berlin_time = datetime.now(ZoneInfo("Europe/Berlin"))
formatted_time = berlin_time.strftime("%A, %Y-%m-%d %H:%M:%S %Z")

def _format_prompt(template: str, *, datetime: str, doc_count: int | None = None) -> str:
    safe_values = {"datetime": datetime}
    if "{doc_count" in template:
        safe_values["doc_count"] = doc_count if doc_count is not None else 0
    return template.format(**safe_values)


if database_available:
    try:
        # Get current document count and latest prompt from database 
        all_entries = get_kb_entries()
        doc_count = len(all_entries)
        current_prompt, _ = get_latest_prompt()
        CUSTOM_INSTRUCTIONS = _format_prompt(current_prompt, datetime=formatted_time, doc_count=doc_count)
    except Exception as e:
        st.warning("Using default prompt due to database connection issues. Error: " + str(e))
        CUSTOM_INSTRUCTIONS = _format_prompt(DEFAULT_PROMPT, datetime=formatted_time)
else:
    # Use default prompt when database is not available
    CUSTOM_INSTRUCTIONS = _format_prompt(DEFAULT_PROMPT, datetime=formatted_time)

col1, col2 = st.columns([3, 3])
with col1:
    st.image(BASE_DIR / "assets/viadrina-ub-logo.png", width=300)
with col2:
       st.markdown('<div id="chatbot-title"><h1>Viadrina Library Assistant</h1></div>', unsafe_allow_html=True)
st.markdown('<div id="special-hr"><hr></div>', unsafe_allow_html=True)

# replay chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize unique session ID for conversation tracking
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    print(f"🆕 Generated new session_id: {st.session_state.session_id}")
else:
    print(f"♻️ Using existing session_id: {st.session_state.session_id}")

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user", avatar= AVATAR_USER).markdown(msg["content"])
    elif msg["role"] == "assistant":
        if "rendered" in msg:
            with st.chat_message("assistant", avatar= AVATAR_ASSISTANT):
                st.markdown(msg["rendered"], unsafe_allow_html=True)
                if msg.get("sources"):
                    with st.expander("References", icon=":material/info:", expanded=False):
                        st.markdown(msg["sources"], unsafe_allow_html=True)
        else:
            st.chat_message("assistant", avatar= AVATAR_ASSISTANT).markdown(msg["content"])

# main input
user_input = st.chat_input("Ask me library-related questions in any language ...")
if user_input:
    st.chat_message("user", avatar= AVATAR_USER).markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Optional: scope retrieval (adjust to your ingestion metadata)
    # retrieval_filters = {"department": "library", "year": {"$in": [2023, 2024, 2025]}}
    # Build web_search filters from admin settings
    fs = get_filter_settings()
    web_filters = {}
    tool_extras = {}
    try:
        fs = get_filter_settings()
    except Exception as e:
        print(f"⚠️ Could not load filter_settings (using defaults): {e}")
        fs = {}
    if fs:
        # enable flag
        tool_extras['web_search_enabled'] = bool(fs.get('web_search_enabled', True))
        # allowed_domains: only include if provided; otherwise no restrictions
        domains = fs.get('web_domains') or []
        if domains:
            web_filters['allowed_domains'] = domains
        # user_location
        ul_type = (fs.get('web_userloc_type') or '').strip()
        ul_country = (fs.get('web_userloc_country') or '').strip()
        ul_city = (fs.get('web_userloc_city') or '').strip()
        ul_region = (fs.get('web_userloc_region') or '').strip()
        ul_timezone = (fs.get('web_userloc_timezone') or '').strip()
        if ul_type:
            loc = {
                'type': ul_type,
                'country': ul_country or None,
                'city': ul_city or None,
            }
            if ul_region:
                loc['region'] = ul_region
            if ul_timezone:
                loc['timezone'] = ul_timezone
            tool_extras['user_location'] = loc
    # If no filters defined, pass None so tool has no restrictions
    retrieval_filters = web_filters if web_filters else None

    handle_stream_and_render(user_input, CUSTOM_INSTRUCTIONS, client, retrieval_filters, debug_one=debug_one, web_tool_extras=tool_extras)
