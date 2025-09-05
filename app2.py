import streamlit as st
from openai import OpenAI, APIConnectionError, BadRequestError, RateLimitError
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import json, re, html
import psycopg2
from functools import lru_cache
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
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

def generate_support_ticket_id():
    """Generate a unique support ticket ID"""
    timestamp = datetime.now().strftime("%Y%m%d")
    unique_id = str(uuid.uuid4())[:8].upper()
    return f"VIALIB-{timestamp}-{unique_id}"

def get_library_contact_email():
    """Get the appropriate email address for library contacts based on debug mode"""
    try:
        debug_mode = st.secrets.get("DEBUG_EMAIL_MODE", False)
        if debug_mode:
            debug_email = st.secrets.get("DEBUG_EMAIL_ADDRESS", "test@example.com")
            return debug_email, True  # Return email and debug flag
        else:
            production_email = st.secrets.get("PRODUCTION_EMAIL_ADDRESS", "ub_ausk@europa-uni.de")
            return production_email, False
    except Exception:
        # Fallback to production email if secrets are not available
        return "ub_ausk@europa-uni.de", False

def create_library_contact_email(user_query, ai_response, ticket_id, user_email=None):
    """Create a professional email template for library staff"""
    
    # Get recipient email and debug status
    recipient_email, is_debug_mode = get_library_contact_email()
    
    # Determine query language for appropriate response
    is_german = any(word in user_query.lower() for word in ['der', 'die', 'das', 'und', 'ist', 'sind', 'haben', 'kann', 'welche', 'wie'])
    
    # Add debug mode prefix to subject if in debug mode
    debug_prefix = "[DEBUG MODE] " if is_debug_mode else ""
    
    if is_german:
        subject = f"{debug_prefix}Forschungsanfrage von AI-Assistent - Ticket {ticket_id}"
        email_body = f"""{"🧪 DEBUG MODE: Diese E-Mail wird zu Testzwecken an " + recipient_email + " gesendet, nicht an die echte Bibliothek." if is_debug_mode else ""}

Sehr geehrte Damen und Herren,

ein Nutzer des Viadrina Library AI-Assistenten benötigt weiterführende Unterstützung bei seiner Forschungsanfrage.

📋 TICKET-DETAILS:
Ticket-ID: {ticket_id}
Zeitstempel: {datetime.now().strftime('%d.%m.%Y, %H:%M:%S')}
{f"Nutzer-Email: {user_email}" if user_email else "Nutzer-Email: Nicht angegeben"}
{"DEBUG-Modus: Aktiv (E-Mail an Test-Adresse)" if is_debug_mode else ""}

❓ URSPRÜNGLICHE ANFRAGE:
{user_query}

🤖 AI-ANTWORT (Auszug):
{ai_response[:500]}{"..." if len(ai_response) > 500 else ""}

📝 ERFORDERLICHE UNTERSTÜTZUNG:
Der AI-Assistent konnte die Anfrage nicht vollständig beantworten. Eine persönliche Beratung durch die Bibliotheksmitarbeiter wäre hilfreich.

---
Automatisch generiert vom Viadrina Library AI-Assistant
{"Produktiv-E-Mail: ub_ausk@europa-uni.de" if is_debug_mode else "Für Rückfragen zu diesem System: ub_ausk@europa-uni.de"}"""
    else:
        subject = f"{debug_prefix}Research Inquiry from AI Assistant - Ticket {ticket_id}"
        email_body = f"""{"🧪 DEBUG MODE: This email is being sent to " + recipient_email + " for testing purposes, not to the actual library." if is_debug_mode else ""}

Dear Library Staff,

A user of the Viadrina Library AI Assistant requires additional support for their research inquiry.

📋 TICKET DETAILS:
Ticket ID: {ticket_id}
Timestamp: {datetime.now().strftime('%Y-%m-%d, %H:%M:%S')}
{f"User Email: {user_email}" if user_email else "User Email: Not provided"}
{"Debug Mode: Active (Email sent to test address)" if is_debug_mode else ""}

❓ ORIGINAL QUERY:
{user_query}

🤖 AI RESPONSE (Excerpt):
{ai_response[:500]}{"..." if len(ai_response) > 500 else ""}

📝 REQUIRED ASSISTANCE:
The AI assistant could not fully answer this inquiry. Personal consultation with library staff would be helpful.

---
Automatically generated by Viadrina Library AI Assistant
{"Production Email: ub_ausk@europa-uni.de" if is_debug_mode else "For questions about this system: ub_ausk@europa-uni.de"}"""

    return subject, email_body, recipient_email

def send_library_contact_email(user_query, ai_response, user_email=None, error_code=None):
    """Send contact request to library staff"""
    try:
        ticket_id = generate_support_ticket_id()
        subject, email_body, recipient_email = create_library_contact_email(user_query, ai_response, ticket_id, user_email)
        
        # Try to send actual email if SMTP is configured
        email_sent = False
        smtp_error = None
        
        try:
            smtp_config = st.secrets.get("smtp", None)
            if smtp_config:
                email_sent = send_smtp_email(recipient_email, subject, email_body, smtp_config)
            else:
                smtp_error = "SMTP not configured - email template created but not sent"
        except Exception as e:
            smtp_error = f"SMTP sending failed: {str(e)}"
        
        # Log the contact request to database if available
        if database_available:
            try:
                with get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            INSERT INTO library_contacts (ticket_id, user_query, ai_response, user_email, timestamp, status, notes, error_code)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """, (ticket_id, user_query, ai_response, user_email, datetime.now(), 
                              'sent' if email_sent else 'template_only', 
                              None if email_sent else smtp_error, error_code))
                        conn.commit()
            except Exception as e:
                print(f"Error logging library contact: {e}")
        
        return ticket_id, subject, email_body, recipient_email, email_sent, smtp_error
        
    except Exception as e:
        print(f"Error creating library contact: {e}")
        return None, None, None, None, False, str(e)

def send_smtp_email(to_email, subject, body, smtp_config):
    """Send email via SMTP"""
    try:
        msg = MIMEMultipart()
        msg['From'] = smtp_config['username']
        msg['To'] = to_email
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        
        server = smtplib.SMTP(smtp_config['host'], smtp_config['port'])
        if smtp_config.get('use_tls', True):
            server.starttls()
        server.login(smtp_config['username'], smtp_config['password'])
        server.send_message(msg)
        server.quit()
        
        print(f"✅ Email sent successfully to {to_email}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return False

def create_library_contacts_table():
    """Create table for tracking library contact requests"""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS library_contacts (
                        id SERIAL PRIMARY KEY,
                        ticket_id VARCHAR(50) UNIQUE NOT NULL,
                        user_query TEXT NOT NULL,
                        ai_response TEXT,
                        user_email VARCHAR(255),
                        timestamp TIMESTAMP NOT NULL,
                        status VARCHAR(20) DEFAULT 'pending',
                        notes TEXT,
                        error_code VARCHAR(10)
                    );
                """)
                conn.commit()
                print("✅ Library contacts table ready.")
    except Exception as e:
        print(f"Error creating library_contacts table: {e}")

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
        
        if a_type == "file_citation":
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

            norm.append({"file_id": file_id, "filename": filename, "index": idx, "type": "file_citation"})
        
        # Handle web search citations - try multiple possible annotation types
        elif a_type in ["url_citation", "web_search_citation", "web_citation"]:
            # Web search results - try different possible field names
            url = _ga(a, "url") or _ga(a, "link") or _ga(a, "href")
            title = _ga(a, "title") or _ga(a, "name") or _ga(a, "text")
            
            # Web citations use start_index/end_index, not index
            idx = _ga(a, "start_index") or _ga(a, "index")
            
            try:
                idx = int(idx) if idx is not None else len(full_text)
            except Exception:
                idx = len(full_text)
            
            if url:  # Only add if we have a URL
                norm.append({
                    "type": "url_citation",
                    "url": url,
                    "title": title or "Web Result",
                    "index": idx
                })

    if not norm:
        return citation_map, placements

    # Separate file citations and web citations
    file_citations = [n for n in norm if n.get("type") == "file_citation"]
    web_citations = [n for n in norm if n.get("type") == "url_citation"]

    # Handle file citations (existing logic)
    file_data = {}
    if file_citations:
        file_ids = list({n["file_id"] for n in file_citations})
        conn = get_connection()
        file_data = get_urls_and_titles_by_file_ids(conn, file_ids)  # {file_id: (url, title, summary)}

    # Build map and placements in order of appearance
    for i, n in enumerate(norm, start=1):
        idx = max(0, min(len(full_text), n["index"]))  # clamp
        
        if n.get("type") == "file_citation":
            # File citation processing
            file_id = n["file_id"]
            filename = n["filename"] or _cached_filename(file_id)
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
                "source": "document"
            }
        elif n.get("type") == "url_citation":
            # Web search citation processing
            citation_map[i] = {
                "number": i,
                "file_name": "Web Search",
                "file_id": "web_search",
                "url": n["url"],
                "title": html.escape(n["title"] or "Web Result"),
                "summary": "",
                "source": "web_search"
            }
        
        placements.append((idx, i))

    return citation_map, placements

def render_with_citations_by_index(text_html, citation_map, placements):
    """
    Insert citations at the specified character indices.
    """
    s = text_html or ""
    n = len(s)
    
    # Sort placements in reverse order to avoid index shifting
    for idx, num in sorted(placements, key=lambda x: x[0], reverse=True):
        note = citation_map.get(num)
        if not note:
            continue
        
        # Use clean citation markers that preserve markdown
        citation = f" [{num}]"
        i = max(0, min(n, idx))
        s = s[:i] + citation + s[i:]
    
    return s

def render_sources_list(citation_map):
    if not citation_map:
        return ""
    lines = []
    for c in citation_map.values():
        title = c["title"] or "Untitled"
        summary = html.escape(c["summary"] or "")
        badge = f"[{c['number']}]"
        source_type = c.get("source", "document")
        
        # Add source type indicator
        if source_type == "web_search":
            badge = f"🌐 {badge}"
        else:
            badge = f"📄 {badge}"
            
        if c["url"]:
            safe_title = html.escape(title)
            if summary:
                lines.append(f"* {badge} [{safe_title}]({c['url']}) - {summary[:100]}{'...' if len(summary) > 100 else ''}")
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

def get_specialized_domains(user_input):
    """Get specialized search domains based on query content (max 20 domains for OpenAI API)"""
    query_lower = user_input.lower()
    
    # Start with core domains (always included)
    domains = [
        "europa-uni.de",
        "scholar.google.com",
        "arxiv.org",
        "researchgate.net",
        "worldcat.org",
        "jstor.org",
        "springer.com",
        "cambridge.org"
    ]
    
    # Add specialized domains based on query type (limit total to 20)
    remaining_slots = 20 - len(domains)
    
    # Medical/Health research
    if any(term in query_lower for term in ["medical", "health", "medicine", "clinical", "patient"]):
        medical_domains = [
            "pubmed.ncbi.nlm.nih.gov",
            "nih.gov",
            "who.int",
            "cochrane.org",
            "nejm.org",
            "bmj.com"
        ]
        domains.extend(medical_domains[:remaining_slots])
    
    # Legal research
    elif any(term in query_lower for term in ["law", "legal", "court", "legislation", "regulation"]):
        legal_domains = [
            "gesetze-im-internet.de",
            "bundesverfassungsgericht.de",
            "curia.europa.eu",
            "hudoc.echr.coe.int"
        ]
        domains.extend(legal_domains[:remaining_slots])
    
    # General academic for other queries
    else:
        general_domains = [
            "nature.com",
            "science.org",
            "wikipedia.org",
            "mit.edu"
        ]
        domains.extend(general_domains[:remaining_slots])
    
    # Ensure we never exceed 20 domains
    return domains[:20]

# ---------- Streamed handling ----------
def handle_stream_and_render(user_input, system_instructions, client, retrieval_filters=None, debug_one=False):
    """
    Responses API (streaming) with spinner INSIDE the assistant chat bubble
    """
    if "VECTOR_STORE_ID" not in st.secrets:
        st.error("Missing VECTOR_STORE_ID in Streamlit secrets. Please add it to run File Search.")
        st.stop()

    tool_cfg = {
        "type": "file_search",
        "vector_store_ids": [st.secrets["VECTOR_STORE_ID"]],
        "max_num_results": 4,
    }
    if retrieval_filters:
        tool_cfg["filters"] = retrieval_filters

    # Web search tool
    specialized_domains = get_specialized_domains(user_input)
    web_search_tool = {
        "type": "web_search",
        "search_context_size": "medium",
        "filters": {
            "allowed_domains": specialized_domains
        }
    }
    
    tools = [tool_cfg, web_search_tool]

    # Build conversation context
    def build_conversation_context():
        conversation = [{"role": "system", "content": system_instructions}]
        
        # Add recent conversation history
        recent_messages = st.session_state.messages[-10:] if st.session_state.messages else []
        
        for msg in recent_messages:
            if msg["role"] == "user":
                conversation.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                content = msg.get("raw_text", msg.get("rendered", msg.get("content", "")))
                content = re.sub(r'<[^>]+>', '', content) if content else ""
                conversation.append({"role": "assistant", "content": content})
        
        conversation.append({"role": "user", "content": user_input})
        return conversation

    conversation_input = build_conversation_context()

    # Initialize variables
    buf = ""
    final = None
    rendered = ""
    sources_md = ""
    citation_map = {}
    response_text = ""
    confidence = 0.0
    error_code = None
    request_classification = "other"
    evaluation_notes = ""

    # Assistant bubble first so spinner sits next to avatar
    with st.chat_message("assistant"):
        content_placeholder = st.empty()

        spinner_ctx = st.spinner("Thinking…")
        spinner_ctx.__enter__()
        first_token = False

        try:
            with client.responses.stream(
                model=st.secrets.get("MODEL", "gpt-4o-mini"),
                input=conversation_input,
                tools=tools,
            ) as stream:
                for event in stream:
                    if event.type == "response.output_text.delta":
                        if not first_token:
                            try:
                                spinner_ctx.__exit__(None, None, None)
                            except Exception:
                                pass
                            first_token = True

                        buf += event.delta
                        content_placeholder.markdown(buf)

                final = stream.get_final_response()

        except (BadRequestError, APIConnectionError, RateLimitError) as e:
            try:
                spinner_ctx.__exit__(None, None, None)
            except Exception:
                pass

            if isinstance(e, BadRequestError):
                with st.spinner("Thinking…"):
                    final = client.responses.create(
                        model=st.secrets.get("MODEL", "gpt-4o-mini"),
                        input=conversation_input,
                        tools=tools,
                    )
                    buf = _get_attr(final, "output_text", "") or ""
                    content_placeholder.markdown(buf)
            elif isinstance(e, RateLimitError):
                st.error("We're being rate-limited right now. Please try again in a moment.")
                return
            else:
                st.error(f"❌ OpenAI connection error: {e}")
                return

        finally:
            try:
                spinner_ctx.__exit__(None, None, None)
            except Exception:
                pass

        # Post-process final for citations & re-render
        if not final:
            st.warning("I couldn't generate a response this time.")
            return

        response_text = _get_attr(final, "output_text", "") or buf

        # Find and normalize the first text-bearing content part
        normalized_part = None
        for item in _iter_content_items(final):
            item_type = _get_attr(item, "type")
            if item_type == "message":
                for part in _iter_content_parts(item):
                    maybe = coerce_text_part(part)
                    if maybe:
                        normalized_part = maybe
                        break
            if normalized_part:
                break

        # Build citations
        if normalized_part:
            citation_map, placements = extract_citations_from_annotations_response_dict(normalized_part)
        else:
            citation_map, placements = ({}, [])

        cleaned = response_text.strip()
        rendered = render_with_citations_by_index(cleaned, citation_map, placements)
        sources_md = render_sources_list(citation_map)

        # Overwrite the streamed text with the enriched version
        content_placeholder.markdown(rendered)
        if sources_md:
            with st.expander("Show sources"):
                st.markdown(sources_md)

        # Optional debug dump
        if debug_one and final:
            with st.expander("🔎 Debug: final.model_dump()", expanded=False):
                try:
                    dump = final.model_dump()
                    st.json(dump)
                except Exception:
                    st.code(_redact_ids(repr(final)), language="json")

    # Persist chat history
    st.session_state.messages.append({
        "role": "assistant",
        "raw_text": response_text,
        "rendered": rendered,
        "sources": sources_md,
        "citation_json": json.dumps(citation_map, ensure_ascii=False) if citation_map else None,
        "error_code": error_code,
        "evaluation_notes": evaluation_notes,
        "confidence": confidence,
        "request_classification": request_classification,
    })

    # Keep memory footprint bounded
    MAX_HISTORY = 50
    if len(st.session_state.messages) > MAX_HISTORY:
        st.session_state.messages = st.session_state.messages[-MAX_HISTORY:]

    # Structured evaluation follow-up
    try:
        evaluation_system_prompt = f"""
        You are an expert evaluator for a library assistant chatbot. Your task is to:
        1. Classify the user's request type
        2. Evaluate the assistant's response quality and accuracy
        
        Consider these factors:
        - How well the user's question was answered
        - Quality and relevance of sources cited ({len(citation_map)} sources used)
        - Completeness of the response
        - Any potential inaccuracies or gaps
        
        Return ONLY a valid JSON object:
        {{
            "request_classification": "one of: library_hours, book_search, research_help, account_info, facility_info, policy_question, technical_support, literature search, citation search, other",
            "confidence": 0.0-1.0,
            "error_code": 0-3,
            "evaluation_notes": "brief explanation"
        }}
        
        Error codes: 0=perfect response, 1=minor issues, 2=significant gaps, 3=unable to answer properly
        """

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
        if payload_text:
            if debug_one:
                with st.expander("🔎 Response Evaluation (JSON)", expanded=False):
                    st.code(payload_text, language="json")

            payload = json.loads(payload_text)
            confidence = float(payload.get("confidence", 0.0))
            ec = payload.get("error_code")
            error_code = f"E0{ec}" if ec is not None else None
            request_classification = payload.get("request_classification", "other")
            evaluation_notes = payload.get("evaluation_notes", "")

            # Update the last message with evaluation data
            if st.session_state.messages:
                st.session_state.messages[-1].update({
                    "error_code": error_code,
                    "evaluation_notes": evaluation_notes,
                    "confidence": confidence,
                    "request_classification": request_classification,
                })

            # Show library contact suggestion for incomplete responses
            if ec is not None and ec >= 2:
                with st.container():
                    st.info("""
                    🤔 **Need more comprehensive help?** 
                    
                    It seems this question might benefit from personalized assistance. Our library staff can provide 
                    in-depth research support and access to specialized resources.
                    """)
                # Set a flag to show contact button
                st.session_state.show_contact_button = True

    except Exception as e:
        if debug_one:
            st.error(f"Evaluation call error: {e}")

    # Log interaction
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
        create_library_contacts_table()
        print("✅ Database initialization completed successfully.")
    else:
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
        print("⚠️ Database not available. Continuing without database functionality.")
except Exception as e:
    error_str = str(e)
    if ("localhost" in error_str or "127.0.0.1" in error_str or 
        "postgres" in error_str or "Missing PostgreSQL" in error_str or
        ("KeyError" in error_str and "postgres" in error_str.lower())):
        st.warning("🔧 **Database Configuration Needed**")
        st.info("""
        To enable full functionality, please configure a PostgreSQL database.
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
    CUSTOM_INSTRUCTIONS = DEFAULT_PROMPT.format(datetime=formatted_time)

st.set_page_config(page_title="Viadrina Library Assistant", layout="wide", initial_sidebar_state="collapsed")
load_css("css/styles.css")

col1, col2 = st.columns([3, 3])
with col1:
    st.image(BASE_DIR / "assets/viadrina-logo.png", width=300)
with col2:
    st.markdown("# Viadrina Library Assistant")
st.markdown("---")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_contact_button" not in st.session_state:
    st.session_state.show_contact_button = False
if "show_contact_form" not in st.session_state:
    st.session_state.show_contact_form = False

# Replay chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    elif msg["role"] == "assistant":
        if "rendered" in msg:
            with st.chat_message("assistant"):
                st.markdown(msg["rendered"])
                if msg.get("sources"):
                    with st.expander("Show sources"):
                        st.markdown(msg["sources"])
        else:
            st.chat_message("assistant").markdown(msg["content"])

# Main input
user_input = st.chat_input("Ask me library-related questions in any language ...")
if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Reset contact button state for new queries
    st.session_state.show_contact_button = False

    retrieval_filters = None
    handle_stream_and_render(user_input, CUSTOM_INSTRUCTIONS, client, retrieval_filters, debug_one=debug_one)

# Contact button logic - show if flagged by evaluation
if st.session_state.get("show_contact_button", False):
    st.markdown("---")
    st.markdown("### 📋 Need Additional Help?")
    st.markdown("If the AI assistant couldn't fully answer your question, our library staff can provide personalized research assistance.")
    
    if st.button("📧 Contact Library Staff", type="secondary"):
        st.session_state.show_contact_form = True

# Contact Form Modal
if st.session_state.get("show_contact_form", False):
    with st.container():
        st.markdown("#### 📬 Contact Library Staff")
        
        # Get the last user query and AI response
        last_user_query = ""
        last_ai_response = ""
        last_error_code = ""
        
        if st.session_state.messages:
            for i in range(len(st.session_state.messages) - 1, -1, -1):
                msg = st.session_state.messages[i]
                if msg["role"] == "user" and not last_user_query:
                    last_user_query = msg["content"]
                elif msg["role"] == "assistant" and not last_ai_response:
                    last_ai_response = msg.get("raw_text", msg.get("rendered", ""))
                    last_error_code = msg.get("error_code", "")
                if last_user_query and last_ai_response:
                    break
        
        with st.form("library_contact_form"):
            st.markdown("**Your Recent Query:**")
            st.text_area("Query", value=last_user_query, height=100, disabled=True)
            
            st.markdown("**Your Email (optional):**")
            user_email = st.text_input("Email", placeholder="your.email@example.com")
            
            st.markdown("**Additional Details (optional):**")
            additional_details = st.text_area("Additional context or specific questions", height=100)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                submitted = st.form_submit_button("📤 Send Request", type="primary")
            with col2:
                if st.form_submit_button("❌ Cancel"):
                    st.session_state.show_contact_form = False
                    st.rerun()
            
            if submitted:
                if last_user_query:
                    full_context = last_ai_response
                    if additional_details:
                        full_context += f"\n\nAdditional context from user:\n{additional_details}"
                    
                    ticket_id, subject, email_body, recipient_email, email_sent, smtp_error = send_library_contact_email(
                        last_user_query, 
                        full_context, 
                        user_email if user_email else None,
                        last_error_code
                    )
                    
                    if ticket_id:
                        _, is_debug_mode = get_library_contact_email()
                        
                        if email_sent:
                            status_icon = "✅"
                            status_text = "Request Sent Successfully!"
                            delivery_text = f"Your inquiry has been sent via email to our library staff at `{recipient_email}`."
                        else:
                            status_icon = "⚠️"
                            status_text = "Request Created (Email Pending)"
                            delivery_text = f"Your inquiry has been recorded. Template was created for `{recipient_email}`."
                        
                        st.success(f"""
                        {status_icon} **{status_text}**
                        
                        **Ticket ID:** `{ticket_id}`
                        
                        {delivery_text}
                        They will review your question and respond as soon as possible.
                        
                        Please save your ticket ID for reference.
                        """)
                        
                        st.session_state.show_contact_form = False
                        st.session_state.show_contact_button = False
                    else:
                        st.error("❌ Error sending request. Please try again or contact ub_ausk@europa-uni.de directly.")
                else:
                    st.error("❌ No recent query found. Please ask a question first, then use this contact form.")
