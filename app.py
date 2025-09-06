import streamlit as st
from openai import OpenAI, APIConnectionError, BadRequestError, RateLimitError
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import json, re, html
import psycopg2
from functools import lru_cache
import uuid

# ---- your utilities ----
from utils import (
    get_connection, create_prompt_versions_table, create_log_table,
    initialize_default_prompt_if_empty, get_latest_prompt, render_sidebar,
    create_database_if_not_exists
)
# -------------------------------------

# Response evaluation schema for the second API call
RESPONSE_EVALUATION_SCHEMA = {
    "name": "response_evaluation",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "request_classification": {
                "type": "string",
                "enum": ["library_hours", "book_search", "research_help", "account_info", "facility_info", "policy_question", "technical_support", "literature search", "citation search", "other"],
                "description": "Classification of the user's request type"
            },
            "confidence": {
                "type": "number",
                "description": "Confidence in the response quality (0.0-1.0)",
                "minimum": 0,
                "maximum": 1
            },
            "error_code": {
                "type": "integer",
                "description": "Error assessment: 0=perfect response, 1=minor issues/uncertainty, 2=significant gaps, 3=unable to answer properly"
            },
            "evaluation_notes": {
                "type": "string",
                "description": "Brief explanation of confidence/error assessment"
            }
        },
        "required": ["request_classification", "confidence", "error_code", "evaluation_notes"],
        "additionalProperties": False
    }
}

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

# ---------- UI + constants ----------
# Check if user is authenticated (admin)
is_authenticated = st.session_state.get("authenticated", False)

render_sidebar(authenticated=is_authenticated)
with st.sidebar:
    if is_authenticated:
        st.markdown("### Developer")
        debug_one = st.checkbox("Debug: show next response object", value=False, help="Shows final.model_dump() for the next assistant reply.")
        
        # Show session ID for debugging
        if "session_id" in st.session_state:
            st.caption(f"Session ID: `{st.session_state.session_id[:8]}...`")
    else:
        debug_one = False

BASE_DIR = Path(__file__).parent

def load_css(file_path):
    with open(BASE_DIR / file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def log_interaction(user_input, assistant_response, session_id=None, citation_json=None, citation_count=0, confidence=0.0, error_code=None, request_classification=None, evaluation_notes=None):
    # Debug: Print session_id to console for troubleshooting
    print(f"🔍 log_interaction called with session_id: {session_id} (type: {type(session_id)})")
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO log_table (timestamp, session_id, user_input, assistant_response, error_code, citation_count, citations, confidence, request_classification, evaluation_notes)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (datetime.now(), session_id, user_input, assistant_response, error_code, citation_count, citation_json, confidence, request_classification, evaluation_notes))
                conn.commit()
                print(f"✅ Successfully logged interaction with session_id: {session_id}")
    except psycopg2.Error as e:
        print(f"❌ DB logging error: {e}")

def get_urls_and_titles_by_file_ids(conn, file_ids):
    """Returns {file_id: (url, title, summary)}"""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT vector_file_id, url, title, summary
            FROM documents
            WHERE vector_file_id = ANY(%s);
        """, (file_ids,))
        rows = cur.fetchall()
        return {row[0]: (row[1], row[2], row[3]) for row in rows}

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
        a_type = _ga(a, "type")
        if a_type != "file_citation":
            continue

        # file_id may be on the annotation or nested under .file_citation
        file_id = _ga(a, "file_id")
        if not file_id:
            fc = _ga(a, "file_citation")
            file_id = _ga(fc, "file_id") if fc is not None else None
        if not file_id:
            continue

        filename = _ga(a, "filename")
        idx = _ga(a, "index")
        try:
            idx = int(idx) if idx is not None else len(full_text)
        except Exception:
            idx = len(full_text)

        norm.append({"file_id": file_id, "filename": filename, "index": idx})

    if not norm:
        return citation_map, placements

    # Deduplicate for DB lookup
    file_ids = list({n["file_id"] for n in norm})
    conn = get_connection()
    file_data = get_urls_and_titles_by_file_ids(conn, file_ids)  # {file_id: (url, title, summary)}

    # Build map and placements in order of appearance
    for i, n in enumerate(norm, start=1):
        file_id = n["file_id"]
        filename = n["filename"] or _cached_filename(file_id)
        idx = max(0, min(len(full_text), n["index"]))  # clamp

        url, title, summary = file_data.get(file_id, (None, None, None))
        if not title:
            title = filename.rsplit(".", 1)[0]

        clean_title = html.escape(re.sub(r"^\d+_", "", title).replace("_", " "))
        clean_summary = html.escape(summary or "")

        citation_map[i] = {
            "number": i,
            "file_name": filename,
            "file_id": file_id,
            "url": url,
            "title": clean_title,
            "summary": clean_summary,
        }
        placements.append((idx, i))

    return citation_map, placements

def render_with_citations_by_index(text_html, citation_map, placements):
    """
    Insert <sup>[n]</sup> at the specified character indices (desc order to avoid shifts).
    """
    s = text_html or ""
    n = len(s)
    for idx, num in sorted(placements, key=lambda x: x[0], reverse=True):
        note = citation_map.get(num)
        if not note:
            continue
        sup = f"<sup title='Source: {note['title']}'>[{num}]</sup>"
        i = max(0, min(n, idx))
        s = s[:i] + sup + s[i:]
    return s

def render_sources_list(citation_map):
    if not citation_map:
        return ""
    lines = []
    for c in citation_map.values():
        title = c["title"] or "Untitled"
        summary = html.escape(c["summary"] or "")
        badge = f"[{c['number']}]"
        if c["url"]:
            # Always escape the title too (already escaped earlier, but harmless to do again)
            safe_title = html.escape(title)
            # Plain Markdown link (safe) if no summary; HTML anchor with title tooltip otherwise
            if summary:
                url_with_tooltip = f'<a href="{c["url"]}" title="{summary}" target="_blank" rel="noopener noreferrer">{safe_title}</a>'
                lines.append(f"* {badge} {url_with_tooltip}")
            else:
                lines.append(f"* {badge} [{safe_title}]({c['url']})")
        else:
            lines.append(f"* {badge} {title}")
    return "\n".join(lines)

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
        return f"Tool: {short.replace('_', ' ')}"
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
def handle_stream_and_render(user_input, system_instructions, client, retrieval_filters=None, debug_one=False):
    """
    Responses API (streaming) with spinner INSIDE the assistant chat bubble:
      - opens the assistant bubble immediately
      - shows real spinner beside avatar until first event
      - streams tokens into same bubble
      - parses final output, inserts index-based citations, shows sources
      - logs to DB
    """
    if "VECTOR_STORE_ID" not in st.secrets:
        st.error("Missing VECTOR_STORE_ID in Streamlit secrets. Please add it to run File Search.")
        st.stop()

    tool_cfg = [
        {
            "type": "file_search",
            "vector_store_ids": [st.secrets["VECTOR_STORE_ID"]],
            # "max_num_results": 4,  # tune for latency/recall
        },
        {
            "type": "web_search"},
    ]
    if retrieval_filters is not None:
        # Apply filters only to the web_search tool
        for tool in tool_cfg:
            if tool.get("type") == "web_search":
                tool["filters"] = retrieval_filters
                break
            
    # Build conversation context (last 10 exchanges to stay within limits)
    def build_conversation_context():
        conversation = [{"role": "system", "content": system_instructions}]
        
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

    # Assistant bubble first so spinner sits next to avatar
    with st.chat_message("assistant"):
        # Create status row first (spinner + status text on same line)
        status_row = st.container()
        with status_row:
            col_spinner, col_status = st.columns([0.03, 0.97])
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
        got_first_event = False
        buf = ""
        final = None
        tool_event_shown = False

        try:
            with client.responses.stream(
                model=st.secrets.get("MODEL", "gpt-5-mini"),
                input=conversation_input,
                tools=tool_cfg,
                parallel_tool_calls=True,  # force serial tool calls so tool events stream live (if SDK supports)
            ) as stream:
                for event in stream:
                    etype = getattr(event, "type", "") or ""

                    # Keep spinner until the model is done emitting text.
                    if etype == "response.output_text.complete":
                        try:
                            spinner_ctx.__exit__(None, None, None)
                        except Exception:
                            pass
                        got_first_event = True

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
                            status_placeholder.markdown(f"`event.type` = `{etype}`")
                    except Exception:
                        # best-effort: avoid breaking stream on status failures
                        if debug_one:
                            print("status update failed for event:", etype)

                    # If verbose debug requested, show more detailed per-tool phases
                    if debug_one and etype and ("tool_call" in etype or etype.endswith("_call") or "tool" in etype):
                        tool_name = getattr(event, "name", None) or getattr(event, "tool", None) or etype.split(".")[-1]
                        phase = etype.split(".")[-1]
                        try:
                            status_placeholder.markdown(f"_{tool_name}: {phase}_")
                        except Exception:
                            pass
                        tool_event_shown = True
                        continue

                    # Stream tokens
                    if etype == "response.output_text.delta":
                        buf += event.delta
                        content_placeholder.markdown(buf, unsafe_allow_html=True)
                        continue

                    # clear the status when text ends (keep UI minimal)
                    if etype == "response.output_text.complete":
                        status_placeholder.empty()
                        continue

                # fetch the full object for your citation pass later
                final = stream.get_final_response()
                
                # If no tool events were streamed, but final.output shows tool calls,
                # render a short post-hoc status so the UI reflects that tools ran.
                if not tool_event_shown and final is not None:
                    try:
                        out = _get_attr(final, "output", []) or []
                        called = []
                        for item in out:
                            t = _get_attr(item, "type", "") or ""
                            if t.endswith("_call") and t not in called:
                                called.append(t)
                        if called:
                            try:
                                    # Map technical names to human-readable descriptions
                                    friendly_names = []
                                    for tool_type in called:
                                        if "web_search" in tool_type:
                                            friendly_names.append("web search")
                                        elif "file_search" in tool_type:
                                            friendly_names.append("knowledge base search")
                                        else:
                                            friendly_names.append(tool_type.replace("_", " "))
                                    status_placeholder.markdown(f"_Response based on {', '.join(friendly_names)}_")

                            except Exception:
                                pass
                    except Exception as e:
                        print("post-stream tool detection failed:", e)


        except (BadRequestError, APIConnectionError, RateLimitError) as e:
            print(f"❌ OpenAI API error during streaming: {e}", flush=True)
            # Ensure spinner is closed before fallback/error
            try:
                spinner_ctx.__exit__(None, None, None)
            except Exception:
                pass

            if isinstance(e, BadRequestError):
                # Fallback: non-streaming request rendered inside same bubble
                with st.spinner("Thinking…"):
                    final = client.responses.create(
                        model=st.secrets.get("MODEL", "gpt-5-mini"),
                        input=conversation_input,
                        tools=tool_cfg,
                    )
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
            rendered = render_with_citations_by_index(cleaned, citation_map, placements)
            # No need to process filecite markers when we have normalized text with proper annotations
        else:
            # Fallback: we only have output_text which may contain inline filecite markers.
            # Replace markers in-order with supers based on placements (best-effort).
            cleaned = response_text.strip()
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
            with st.expander("Show sources"):
                st.markdown(sources_md, unsafe_allow_html=True)

        # --- Optional debug dump of final (sidebar toggle) ---
        if debug_one and final:
            with st.expander("🔎 Debug: final.model_dump()", expanded=False):
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
    confidence = 0.0
    error_code = None
    request_classification = "other"
    evaluation_notes = ""

    try:
        evaluation_system_prompt = """
        You are an expert evaluator for a library assistant chatbot. Your task is to:
        1. Classify the user's request type
        2. Evaluate the assistant's response quality and accuracy
        
        Consider these factors in your evaluation:
        - How well the user's question was answered
        - Quality and relevance of sources cited ({citation_count} sources used)
        - Completeness of the response
        - Any potential inaccuracies or gaps
        - Whether the response format is appropriate for the request type
        
        IMPORTANT: Return ONLY a valid JSON object with this exact structure:
        {{
            "request_classification": "one of: library_hours, book_search, research_help, account_info, facility_info, policy_question, technical_support, other",
            "confidence": 0.0-1.0,
            "error_code": 0-3,
            "evaluation_notes": "brief explanation"
        }}
        
        Error codes: 0=perfect response, 1=minor issues, 2=significant gaps, 3=unable to answer properly
        """.format(citation_count=len(citation_map))

        structured = client.responses.create(
            model=st.secrets.get("MODEL", "gpt-5-mini"),
            input=[
                {"role": "system", "content": evaluation_system_prompt},
                {"role": "user", "content": f"Original user request: {user_input}"},
                {"role": "assistant", "content": f"Assistant response: {cleaned}"},
                {"role": "user", "content": "Please evaluate this interaction and return the JSON evaluation."}
            ]
        )

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
    except Exception as e:
        if debug_one:
            st.error(f"Evaluation call unexpected error: {e}")
        # Silently continue with default values for production

    try:
        # Debug: Check session ID before logging
        print(f"🔎 About to log interaction with session_id: {st.session_state.session_id}")
        log_interaction(
            user_input=user_input,
            assistant_response=cleaned,
            session_id=st.session_state.session_id,
            citation_json=json.dumps(citation_map, ensure_ascii=False) if citation_map else None,
            citation_count=len(citation_map),
            confidence=confidence,
            error_code=error_code,
            request_classification=request_classification,
            evaluation_notes=evaluation_notes
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

PROMPT_PATH = BASE_DIR / ".streamlit/system_prompt.txt"

with PROMPT_PATH.open("r", encoding="utf-8") as f:
    DEFAULT_PROMPT = f.read()

# Initialize database and tables
database_available = False
try:
    database_available = create_database_if_not_exists()
    if database_available:
        create_prompt_versions_table()
        initialize_default_prompt_if_empty(DEFAULT_PROMPT)
        create_log_table()
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

if database_available:
    try:
        current_prompt, current_note = get_latest_prompt()
        CUSTOM_INSTRUCTIONS = current_prompt.format(datetime=formatted_time)
    except Exception as e:
        st.warning("Using default prompt due to database connection issues.")
        CUSTOM_INSTRUCTIONS = DEFAULT_PROMPT.format(datetime=formatted_time)
else:
    # Use default prompt when database is not available
    CUSTOM_INSTRUCTIONS = DEFAULT_PROMPT.format(datetime=formatted_time)

st.set_page_config(page_title="Viadrina Library Assistant", layout="wide", initial_sidebar_state="collapsed")
load_css("css/styles.css")

col1, col2 = st.columns([3, 3])
with col1:
    st.image(BASE_DIR / "assets/viadrina-logo.png", width=300)
with col2:
    st.markdown("# Viadrina Library Assistant")
st.markdown("---")

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
        st.chat_message("user").markdown(msg["content"])
    elif msg["role"] == "assistant":
        if "rendered" in msg:
            with st.chat_message("assistant"):
                st.markdown(msg["rendered"], unsafe_allow_html=True)
                if msg.get("sources"):
                    with st.expander("Show sources"):
                        st.markdown(msg["sources"], unsafe_allow_html=True)
        else:
            st.chat_message("assistant").markdown(msg["content"])

# main input
user_input = st.chat_input("Ask me library-related questions in any language ...")
if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Optional: scope retrieval (adjust to your ingestion metadata)
    # retrieval_filters = {"department": "library", "year": {"$in": [2023, 2024, 2025]}}
    retrieval_filters = None

    handle_stream_and_render(user_input, CUSTOM_INSTRUCTIONS, client, retrieval_filters, debug_one=debug_one)
