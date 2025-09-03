import streamlit as st
from openai import OpenAI, APIConnectionError, BadRequestError, RateLimitError
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import json, re, html
import psycopg2
from functools import lru_cache

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
    else:
        debug_one = False

BASE_DIR = Path(__file__).parent

def load_css(file_path):
    with open(BASE_DIR / file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def log_interaction(user_input, assistant_response, citation_json=None, citation_count=0, confidence=0.0, error_code=None, request_classification=None, evaluation_notes=None):
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO log_table (timestamp, user_input, assistant_response, error_code, citation_count, citations, confidence, request_classification, evaluation_notes)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (datetime.now(), user_input, assistant_response, error_code, citation_count, citation_json, confidence, request_classification, evaluation_notes))
                conn.commit()
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
      - shows real spinner beside avatar until first token
      - streams tokens into same bubble
      - parses final output, inserts index-based citations, shows sources
      - logs to DB
    """
    if "VECTOR_STORE_ID" not in st.secrets:
        st.error("Missing VECTOR_STORE_ID in Streamlit secrets. Please add it to run File Search.")
        st.stop()

    tool_cfg = {
        "type": "file_search",
        "vector_store_ids": [st.secrets["VECTOR_STORE_ID"]],
        "max_num_results": 4,  # tune for latency/recall
    }
    if retrieval_filters:
        tool_cfg["filters"] = retrieval_filters

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
        content_placeholder = st.empty()

        spinner_ctx = st.spinner("Thinking…")
        spinner_ctx.__enter__()
        first_token = False

        try:
            with client.responses.stream(
                model=st.secrets.get("MODEL", "gpt-5-mini"),
                input=conversation_input,
                tools=[tool_cfg],
            ) as stream:
                for event in stream:
                    if event.type == "response.output_text.delta":
                        if not first_token:
                            # close spinner on first token
                            try:
                                spinner_ctx.__exit__(None, None, None)
                            except Exception:
                                pass
                            first_token = True

                        buf += event.delta
                        content_placeholder.markdown(buf, unsafe_allow_html=True)

                final = stream.get_final_response()


        except (BadRequestError, APIConnectionError, RateLimitError) as e:
            # Ensure spinner is closed before fallback/error
            try:
                spinner_ctx.__exit__(None, None, None)
            except Exception:
                pass

            if isinstance(e, BadRequestError):
                # Fallback: non-streaming request rendered inside same bubble
                with st.spinner("Thinking…"):
                    final = client.responses.create(
                        model=st.secrets.get("MODEL", "gpt-4o-mini"),
                        input=conversation_input,
                        tools=[tool_cfg],
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
            # If no token ever arrived, ensure spinner is closed
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

        cleaned = response_text.strip()
        rendered = render_with_citations_by_index(cleaned, citation_map, placements)
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
            model=st.secrets.get("MODEL", "gpt-4o-mini"),
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
        log_interaction(
            user_input=user_input,
            assistant_response=cleaned,
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
        print("✅ Database initialization completed successfully.")
    else:
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

