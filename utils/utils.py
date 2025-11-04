import psycopg2
import streamlit as st
import hashlib
from pathlib import Path
import json
from functools import lru_cache
import uuid
from datetime import datetime
import base64
import re
import html

try:  # check if running inside Streamlit runtime
    from streamlit.runtime.runtime import Runtime

    STREAMLIT_RUNTIME_EXISTS = Runtime.exists()
except Exception:  # pragma: no cover - runtime absent in certain contexts
    STREAMLIT_RUNTIME_EXISTS = False


_SUP_REF_PATTERN = re.compile(r"<sup[^>]*>\s*\[(\d+)\]\s*</sup>")
_ANCHOR_PATTERN = re.compile(r"<a[^>]*href=\"([^\"]+)\"[^>]*>(.*?)</a>", re.IGNORECASE | re.DOTALL)


def _sanitize_html(content: str) -> str:
    if not content:
        return ""

    def _anchor_repl(match: re.Match) -> str:
        href = match.group(1).strip()
        text = re.sub(r"\s+", " ", match.group(2)).strip()
        text = html.unescape(text)
        return f"[{text}]({href})"

    text = html.unescape(content)
    text = _ANCHOR_PATTERN.sub(_anchor_repl, text)
    text = re.sub(r"<span[^>]*>.*?</span>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<svg[^>]*>.*?</svg>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</?p[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _parse_sources(raw_sources: str) -> list[tuple[str, str]]:
    if not raw_sources:
        return []

    cleaned = _sanitize_html(raw_sources)
    entries: list[tuple[str, str]] = []
    current_key: str | None = None
    buffer: list[str] = []

    def _flush_buffer() -> None:
        nonlocal current_key, buffer
        if current_key and buffer:
            text = " ".join(part.strip() for part in buffer if part.strip())
            if text:
                entries.append((current_key, text))
        current_key = None
        buffer = []

    for line in cleaned.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = re.match(r"^[\*\-]\s*\[(\d+)\]\s*(.*)$", stripped)
        if match:
            _flush_buffer()
            current_key = match.group(1)
            remaining = match.group(2).strip()
            buffer = [remaining] if remaining else []
        else:
            if current_key:
                buffer.append(stripped)
    _flush_buffer()
    return entries


def build_chat_markdown(messages: list[dict]) -> str:
    """Create a markdown transcript from chat history with footnotes."""
    if not messages:
        return ""

    session_id = st.session_state.get("session_id", "session")
    header = [
        "# Viadrina Library Assistant Chat",
        f"*Exported:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"*Session:* {session_id}",
        "",
    ]

    body: list[str] = []
    footnote_lookup: dict[str, int] = {}
    footnotes: list[tuple[int, str]] = []
    next_footnote = 1

    for idx, msg in enumerate(messages, start=1):
        role = msg.get("role", "assistant")
        speaker = "Assistant" if role == "assistant" else "User"
        raw_content = msg.get("content") or msg.get("rendered") or ""
        raw_sources = msg.get("sources") or ""

        source_entries = _parse_sources(raw_sources)
        source_map: dict[str, int] = {}
        for original_id, source_text in source_entries:
            normalized = source_text.strip()
            if not normalized:
                continue
            footnote_id = footnote_lookup.get(normalized)
            if footnote_id is None:
                footnote_id = next_footnote
                footnote_lookup[normalized] = footnote_id
                footnotes.append((footnote_id, normalized))
                next_footnote += 1
            source_map[original_id] = footnote_id

        def _sup_repl(match: re.Match) -> str:
            original = match.group(1)
            footnote_id = source_map.get(original)
            if footnote_id is None:
                return match.group(0)
            return f"[^{footnote_id}]"

        processed_content = html.unescape(raw_content)
        processed_content = _SUP_REF_PATTERN.sub(_sup_repl, processed_content)
        processed_content = _ANCHOR_PATTERN.sub(lambda m: f"[{re.sub(r'\\s+', ' ', html.unescape(m.group(2))).strip()}]({m.group(1).strip()})", processed_content)
        processed_content = re.sub(r"<span[^>]*>.*?</span>", "", processed_content, flags=re.DOTALL | re.IGNORECASE)
        processed_content = re.sub(r"<svg[^>]*>.*?</svg>", "", processed_content, flags=re.DOTALL | re.IGNORECASE)
        processed_content = re.sub(r"<br\s*/?>", "\n", processed_content, flags=re.IGNORECASE)
        processed_content = re.sub(r"</?p[^>]*>", "\n", processed_content, flags=re.IGNORECASE)
        processed_content = re.sub(r"<sup[^>]*>.*?</sup>", "", processed_content, flags=re.DOTALL | re.IGNORECASE)
        processed_content = re.sub(r"<[^>]+>", "", processed_content)

        if source_map:
            processed_content = re.sub(
                r"\[(\d+)\]",
                lambda m: f"[^{source_map.get(m.group(1), m.group(1))}]",
                processed_content,
            )

        processed_content = html.unescape(processed_content)
        processed_content = processed_content.replace("\r\n", "\n").strip()
        processed_content = re.sub(r"\n{3,}", "\n\n", processed_content)
        if not processed_content:
            processed_content = "_(empty message)_"

        body.extend([f"## {idx}. {speaker}", "", processed_content, ""])

    if footnotes:
        body.append("---")
        body.append("")
        for footnote_id, text in footnotes:
            body.append(f"[^{footnote_id}]: {text}")

    return "\n".join(header + body).strip()


def render_save_chat_button(slot, messages: list[dict] | None = None) -> None:
    """Render or refresh the sidebar save button inside the provided slot."""
    if slot is None:
        return

    slot.empty()
    messages = messages or []
    transcript_md = build_chat_markdown(messages)
    filename = "chat-transcript.md"
    session_id = st.session_state.get("session_id")
    if session_id:
        filename = f"chat-{str(session_id)[:8]}.md"

    slot.download_button(
        "Save chat",
        data=transcript_md,
        file_name=filename,
        mime="text/markdown",
        key="sidebar_save_chat_button",
        disabled=not bool(messages),
        help="Download the current conversation as Markdown.",
        icon=":material/save_alt:",
    )

from .db_migrations import run_migrations
from .oidc import (
    allow_password_fallback,
    build_oidc_login_url,
    exchange_code_for_tokens,
    fetch_userinfo,
    generate_pkce_pair,
    get_default_next_path,
    get_oidc_allowlist,
    get_redirect_uri,
    is_oidc_configured,
)

BASE_DIR = Path(__file__).parent.parent
ICON_PATH = (BASE_DIR / "assets" / "Key.png")

BLOCK_UI_HTML = """
<style>
#global-block-ui-overlay {
    position: fixed;
    inset: 0;
    background: rgba(255, 255, 255, 0.65);
    backdrop-filter: blur(2px);
    z-index: 99999;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.05rem;
    font-weight: 600;
    color: #1f2937;
    pointer-events: all;
}
[data-testid="stSidebar"],
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarNav"],
[data-testid="stHeader"] button,
[data-testid="stToolbar"] button {
    pointer-events: none !important;
    filter: grayscale(55%);
    opacity: 0.45;
}
</style>
<div id="global-block-ui-overlay">Working… please hold on.</div>
"""

def load_css(file_path: str = "css/styles.css") -> None:
    """Inject a CSS file into the current Streamlit app."""
    css_path = BASE_DIR / file_path
    with css_path.open("r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Pricing & cost estimation ---
# Source: config/pricing.json (per-1K tokens)

@lru_cache(maxsize=1)
def _load_pricing_config():
    path = BASE_DIR / "config" / "pricing.json"
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Pricing file not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in pricing file {path}: {e}")

    # Validate schema
    if not isinstance(data, dict) or "models" not in data or not isinstance(data["models"], dict):
        raise ValueError("Invalid pricing.json schema: expected top-level 'models' object.")
    return data

def _get_pricing_for_model(model: str):
    if not model:
        raise ValueError("Model name is required for pricing lookup.")
    data = _load_pricing_config()
    models = data["models"]

    # Case-insensitive exact key match only (no fallbacks)
    lower_map = {k.lower(): v for k, v in models.items()}
    entry = lower_map.get((model or "").lower())
    if entry is None:
        raise KeyError(f"Model '{model}' not found in config/pricing.json 'models'.")

    try:
        return {"input": float(entry["input"]), "output": float(entry["output"]) }
    except Exception:
        raise ValueError(
            f"Invalid pricing entry for model '{model}' in pricing.json. Expected numeric 'input' and 'output' per 1K tokens."
        )


def estimate_cost_usd(model: str, input_tokens: int = 0, output_tokens: int = 0) -> float:
    """
    Estimate API cost in USD using per-1K token pricing from config/pricing.json.
    """
    p = _get_pricing_for_model(model)
    # prices are per-1K tokens
    cost = (input_tokens / 1000.0) * p.get("input", 0.0) + (output_tokens / 1000.0) * p.get("output", 0.0)
    return round(cost, 6)

# Function to get a connection to the Postgres database
def get_connection():
    try:
        return psycopg2.connect(
            host=st.secrets["postgres"]["host"],
            port=st.secrets["postgres"]["port"],
            user=st.secrets["postgres"]["user"],
            password=st.secrets["postgres"]["password"],
            dbname=st.secrets["postgres"]["database"]
        )
    except KeyError as e:
        raise ConnectionError(f"Missing PostgreSQL configuration: {e}")
    except psycopg2.OperationalError as e:
        raise ConnectionError(f"Cannot connect to PostgreSQL database: {e}")

def create_database_if_not_exists():
    """
    Create the database if it doesn't exist.
    This connects to the 'postgres' system database to create the target database.
    """
    try:
        # Check if postgres secrets are configured
        if "postgres" not in st.secrets:
            print("⚠️ PostgreSQL secrets not configured in Streamlit Cloud. Skipping database creation.")
            return False
            
        # First, try to connect to the target database
        get_connection()
        print(f"✅ Database '{st.secrets['postgres']['database']}' exists and is accessible.")
        return True
    except KeyError as key_error:
        print(f"⚠️ Missing PostgreSQL configuration key: {key_error}. Skipping database creation.")
        return False
    except psycopg2.OperationalError as e:
        if "does not exist" in str(e):
            print(f"🔨 Database '{st.secrets['postgres']['database']}' does not exist. Creating...")
            try:
                # Connect to postgres system database to create our database
                conn = psycopg2.connect(
                    host=st.secrets["postgres"]["host"],
                    port=st.secrets["postgres"]["port"],
                    user=st.secrets["postgres"]["user"],
                    password=st.secrets["postgres"]["password"],
                    dbname="postgres"  # Connect to system database
                )
                conn.autocommit = True
                cursor = conn.cursor()
                
                # Create the database
                cursor.execute(f'CREATE DATABASE "{st.secrets["postgres"]["database"]}"')
                cursor.close()
                conn.close()
                
                print(f"✅ Database '{st.secrets['postgres']['database']}' created successfully.")
                return True
                
            except psycopg2.Error as create_error:
                print(f"❌ Failed to create database: {create_error}")
                return False
        else:
            print(f"❌ Database connection error: {e}")
            return False

# Function to create the knowledge_base table if it doesn't exist
def create_knowledge_base_table():
    """Ensure the database schema is up to date."""
    run_migrations(get_connection)
    
    
# Function to get knowledge base entries
def get_kb_entries(limit=None):
    """
    Retrieve knowledge base entries from the 'documents' table.
    If limit is None, retrieve all entries.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        if limit is not None:
            cursor.execute("""
                SELECT id, url, title, safe_title, crawl_date, lang, summary, tags, markdown_content, recordset, vector_file_id, page_type, no_upload, is_stale
                FROM documents
                ORDER BY updated_at DESC
                LIMIT %s
            """, (limit,))
        else:
            cursor.execute("""
                SELECT id, url, title, safe_title, crawl_date, lang, summary, tags, markdown_content, recordset, vector_file_id, page_type, no_upload, is_stale
                FROM documents
                ORDER BY updated_at DESC
            """)
        rows = cursor.fetchall()
        return rows
    except Exception as e:
        print(f"[DB ERROR] Failed to fetch KB entries: {e}")
        return []
    finally:
        cursor.close()
        conn.close()


def get_document_metrics() -> dict[str, int]:
    """Return lightweight aggregate counts for documents."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT
                COUNT(*) AS total_pages,
                COALESCE(
                    SUM(
                        CASE
                            WHEN vector_file_id IS NULL AND (no_upload IS NOT TRUE) THEN 1
                            ELSE 0
                        END
                    ),
                    0
                ) AS pending_sync,
                COALESCE(SUM(CASE WHEN is_stale IS TRUE THEN 1 ELSE 0 END), 0) AS stale_pages
            FROM documents
            """
        )
        total_pages, pending_sync, stale_pages = cursor.fetchone()
        return {
            "total_pages": int(total_pages or 0),
            "pending_sync": int(pending_sync or 0),
            "stale_pages": int(stale_pages or 0),
        }
    except Exception as exc:
        print(f"[DB ERROR] Failed to compute document metrics: {exc}")
        raise RuntimeError("Failed to compute document metrics") from exc
    finally:
        cursor.close()
        conn.close()


def is_job_locked(name: str) -> bool:
    """Return True if the named job lock exists."""
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM job_locks WHERE name=%s LIMIT 1", (name,))
            return cur.fetchone() is not None
    except Exception as exc:
        raise RuntimeError(f"Failed to check job lock '{name}': {exc}") from exc
    finally:
        if conn:
            conn.close()


def release_job_lock(name: str) -> None:
    """Best-effort removal of a named job lock."""
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM job_locks WHERE name=%s", (name,))
        conn.commit()
    except Exception:
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


def get_document_by_identifier(*, doc_id: int | None = None, file_id: str | None = None):
    """Fetch a single document row by numeric id or vector/legacy file id."""
    if doc_id is None and not file_id:
        return None

    conn = get_connection()
    cursor = conn.cursor()
    try:
        if doc_id is not None:
            cursor.execute(
                """
                SELECT id, title, url, summary, tags, markdown_content, recordset, updated_at
                FROM documents
                WHERE id = %s
                LIMIT 1
                """,
                (doc_id,),
            )
        else:
            cursor.execute(
                """
                SELECT id, title, url, summary, tags, markdown_content, recordset, updated_at
                FROM documents
                WHERE vector_file_id = %s OR old_file_id = %s
                LIMIT 1
                """,
                (file_id, file_id),
            )

        row = cursor.fetchone()
        if not row:
            return None

        return {
            "id": row[0],
            "title": row[1] or "Untitled Document",
            "url": row[2],
            "summary": row[3],
            "tags": row[4] or [],
            "markdown": row[5],
            "recordset": row[6],
            "updated_at": row[7],
        }
    finally:
        cursor.close()
        conn.close()

# Function to create the log_table if it doesn't exist   
def create_log_table():
    """Ensure log_table schema migrations have run."""
    run_migrations(get_connection)

def check_log_table_schema():
    """
    Debug function to check the current log_table schema.
    Returns column information.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT column_name, data_type, is_nullable 
            FROM information_schema.columns 
            WHERE table_name = 'log_table' 
            ORDER BY ordinal_position;
        """)
        columns = cursor.fetchall()
        cursor.close()
        conn.close()
        return columns
    except Exception as e:
        print(f"Error checking schema: {e}")
        return []

def force_add_session_id_column():
    """
    Force add session_id column if it doesn't exist.
    Use this if the automatic migration didn't work.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Check if column exists
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'log_table' AND column_name = 'session_id';
        """)
        
        if cursor.fetchone() is None:
            # Column doesn't exist, add it
            cursor.execute("ALTER TABLE log_table ADD COLUMN session_id VARCHAR(36);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_log_table_session_id ON log_table(session_id);")
            conn.commit()
            print("✅ Successfully added session_id column to log_table")
        else:
            print("✅ session_id column already exists in log_table")
            
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Error adding session_id column: {e}")
        return False
    
# Function to create the prompt_versions table if it doesn't exist
def create_prompt_versions_table():
    """Ensure prompt_versions table exists via migrations."""
    run_migrations(get_connection)

def create_url_configs_table():
    """Ensure url_configs table exists via migrations."""
    run_migrations(get_connection)

# Function to create an initial system prompt
def initialize_default_prompt_if_empty(default_prompt, edited_by="system"):
    conn = get_connection()
    cursor = conn.cursor()

    # Check if table is empty
    cursor.execute("SELECT COUNT(*) FROM prompt_versions;")
    result = cursor.fetchone()
    is_empty = result[0] == 0

    if is_empty:
        cursor.execute("""
            INSERT INTO prompt_versions (prompt, edited_by, note)
            VALUES (%s, %s, %s);
        """, (default_prompt, edited_by, "Initial default prompt"))
        conn.commit()

    cursor.close()
    conn.close()
    
# Function to get the latest prompt from the database    
def get_latest_prompt():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT prompt, note FROM prompt_versions
        ORDER BY timestamp DESC
        LIMIT 1;
    """)
    result = cursor.fetchone()
    cursor.close()
    conn.close()

    if result:
        prompt, note = result
        return prompt, note
    else:
        return "", ""

def _query_params_to_dict() -> dict:
    qp = getattr(st, "query_params", None)
    if qp is not None:
        if hasattr(qp, "to_dict"):
            return qp.to_dict()  # type: ignore[no-any-return]
        return dict(qp)

    get_qp = getattr(st, "experimental_get_query_params", None)
    if callable(get_qp):
        try:
            return get_qp()
        except Exception:
            pass
    return {}



def _remove_query_params(keys: list[str]) -> None:
    qp = getattr(st, "query_params", None)
    if qp is not None:            # modern API; no experimental warning
        for key in keys:
            qp.pop(key, None)     # pop avoids KeyError for missing keys
        return

    # Only run if the new attribute is genuinely unavailable
    get_qp = getattr(st, "experimental_get_query_params", None)
    set_qp = getattr(st, "experimental_set_query_params", None)
    if callable(get_qp) and callable(set_qp):
        filtered = {k: v for k, v in get_qp().items() if k not in keys}
        set_qp(**filtered)


def _safe_rerun() -> None:
    try:
        st.rerun()
    except RuntimeError as exc:
        if "Event loop is closed" in str(exc):
            return
        raise
    except AttributeError:
        try:
            st.experimental_rerun()  # type: ignore[attr-defined]
        except RuntimeError as exc:
            if "Event loop is closed" in str(exc):
                return
            raise
        except AttributeError:
            pass


def _oidc_user_allowed(email: str) -> bool:
    allowlist = get_oidc_allowlist()
    if not allowlist:
        return True
    candidate = email.strip().lower()
    return bool(candidate and candidate in allowlist)


def admin_authentication(return_to: str | None = None):
    """
    Authenticate admin users via OpenID Connect (when configured) or fallback password.
    """
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    if ICON_PATH.exists():
        encoded_icon = base64.b64encode(ICON_PATH.read_bytes()).decode("utf-8")
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:.75rem;">
                <img src="data:image/png;base64,{encoded_icon}" width="48" height="48"/>
                <h1 style="margin:0;">Admin Login</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.header("Admin Login")

    oidc_ready = is_oidc_configured()
    qp = _query_params_to_dict()

    if oidc_ready and not st.session_state.authenticated:
        error = qp.get("error")
        if error:
            description = qp.get("error_description") or "Authentication failed."
            st.error(f"OIDC error: {error} — {description}")
            _remove_query_params(["error", "error_description", "state", "code"])

        code = qp.get("code")
        state_param = qp.get("state")
        expected_state = st.session_state.get("oidc_state")
        code_verifier = st.session_state.get("oidc_code_verifier")

        if code and state_param and expected_state and code_verifier:
            if state_param != expected_state:
                st.error("Login session mismatch. Please try again.")
            else:
                try:
                    token_payload = exchange_code_for_tokens(
                        code,
                        code_verifier,
                        redirect_uri=get_redirect_uri(),
                    )
                    access_token = token_payload.get("access_token")
                    if not access_token:
                        raise RuntimeError("OIDC token response did not include an access token.")
                    userinfo = fetch_userinfo(access_token)
                except Exception as exc:
                    st.error(f"Authentication failed: {exc}")
                else:
                    email = (userinfo.get("email") or userinfo.get("preferred_username") or "").strip()
                    if not email:
                        st.error("OIDC provider did not return an email address.")
                    elif not _oidc_user_allowed(email):
                        st.error("Your account is not authorized for admin access.")
                    else:
                        st.session_state.authenticated = True
                        st.session_state["admin_email"] = email or st.secrets.get("ADMIN_EMAIL")
                        if userinfo.get("name"):
                            st.session_state["admin_name"] = userinfo.get("name")
                        st.session_state["oidc_claims"] = userinfo
                        _remove_query_params(["code", "state", "session_state"])
                        st.session_state.pop("oidc_state", None)
                        st.session_state.pop("oidc_code_verifier", None)
                        st.session_state.pop("oidc_code_challenge", None)
                        _safe_rerun()
                        return True

            _remove_query_params(["code", "state"])
            st.session_state.pop("oidc_state", None)
            st.session_state.pop("oidc_code_verifier", None)
            st.session_state.pop("oidc_code_challenge", None)

        if not st.session_state.authenticated:
            if "oidc_state" not in st.session_state:
                st.session_state["oidc_state"] = uuid.uuid4().hex
                verifier, challenge = generate_pkce_pair()
                st.session_state["oidc_code_verifier"] = verifier
                st.session_state["oidc_code_challenge"] = challenge
            elif "oidc_code_challenge" not in st.session_state or "oidc_code_verifier" not in st.session_state:
                verifier, challenge = generate_pkce_pair()
                st.session_state["oidc_code_verifier"] = verifier
                st.session_state["oidc_code_challenge"] = challenge

            login_url = build_oidc_login_url(
                st.session_state["oidc_state"],
                st.session_state["oidc_code_challenge"],
            )
            try:
                st.link_button("Continue with SSO", login_url, type="primary")
            except AttributeError:
                st.markdown(f"[Continue with SSO]({login_url})")
            st.caption("Use your institutional credentials to access the admin tools.")

            if not allow_password_fallback():
                return False

            st.markdown("---")
            st.caption("Fallback password login (for emergency use only)")

    password = st.text_input("Admin Password", type="password")
    if password == st.secrets.get("ADMIN_PASSWORD") and password:
        st.session_state.authenticated = True
        _safe_rerun()
    elif password:
        st.error("Incorrect password.")
    return False


# Function to render the sidebar with common elements
def render_sidebar(
    authenticated=False,
    show_debug=False,
    show_new_chat=False,
    show_save_chat=False,
):
    """
    Renders common sidebar elements.
    Args:
        authenticated: Whether user is authenticated
        show_debug: Whether to show debug controls (only for main chat page)
        show_new_chat: Whether to render the sidebar "New chat" button
        show_save_chat: Whether to reserve space for the sidebar "Save chat" download button
    Returns:
        Tuple containing debug state and optional save-chat slot
    """
    load_css()
    st.sidebar.page_link("app.py", label="Chat Assistant", icon=":material/chat_bubble:")

    save_chat_slot = st.sidebar.empty()
    if show_save_chat and save_chat_slot is not None:
        with save_chat_slot.container():
            st.download_button(
                "Save chat",
                data="",
                file_name="chat-transcript.md",
                mime="text/markdown",
                key="sidebar_save_chat_placeholder",
                disabled=True,
                icon=":material/save_alt:",
            )
    else:
        if save_chat_slot is not None:
            save_chat_slot.empty()
            save_chat_slot = None

    if show_new_chat:
        if st.sidebar.button(
            "New chat",
            type="secondary",
            help="Start a fresh session",
            icon=":material/add_comment:",
            key="sidebar_new_chat_button",
        ):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()

    def _perform_logout():
        st.session_state.authenticated = False
        for key in (
            "oidc_state",
            "oidc_code_verifier",
            "oidc_code_challenge",
            "oidc_claims",
            "admin_name",
            "admin_email",
        ):
            st.session_state.pop(key, None)
    

    # Debug checkbox right beneath chat assistant (only on main page when authenticated)
    debug_one = False
    if authenticated and show_debug:
        debug_one = st.sidebar.checkbox("Debug", value=False, 
                                       help="Shows final.model_dump() for the next assistant reply.")
        
        # Show session ID for debugging
        if "session_id" in st.session_state:
            st.sidebar.caption(f"Session ID: `{st.session_state.session_id[:8]}...`")
    st.html("""
    <style>
        .st-key-sidebar_bottom {
            position: absolute;
            bottom: 10px;
        }
    </style>
    """)
    if authenticated:
        st.sidebar.success("Authenticated as admin.")
        st.sidebar.page_link("pages/logs.py", label="Logs & Analytics", icon=":material/search_activity:")
        st.sidebar.page_link("pages/scrape.py", label="Content Indexing", icon=":material/home_storage:")
        st.sidebar.page_link("pages/vectorize.py", label="Vector Store", icon=":material/owl:")
        st.sidebar.page_link("pages/admin.py", label="Settings", icon=":material/settings:")
        #st.sidebar.page_link("pages/manage_users.py", label="👥 Manage Users")
        
        st.sidebar.button("Logout", on_click=_perform_logout, icon=":material/logout:")
        with st.sidebar.container(key="sidebar_bottom"):
            st.caption("Source code on [GitHub](https://github.com/jmiba/ai-service-chatbot)")
    else:
        with st.sidebar.container(key="sidebar_bottom"):
            st.page_link("pages/logs.py", label="Admin Login", icon=":material/key:")
            st.caption("Source code on [GitHub](https://github.com/jmiba/ai-service-chatbot)")
        
    
    return debug_one, save_chat_slot

# Functions to save a document to the knowledge base
def compute_sha256(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def render_log_output(
    log_lines: str,
    *,
    element_id: str,
    height: int = 320,
    waiting_message: str = "Waiting for output...",
    label: str | None = None,
    background: str = "#0f172a",
    text_color: str = "#e2e8f0",
) -> None:
    """Render a scrollable log output area with safe HTML fallback."""

    content = log_lines or waiting_message
    if STREAMLIT_RUNTIME_EXISTS:
        auto_scroll_script = f"""
<script>
const logContainer = document.getElementById("{element_id}");
if (logContainer) {{
  logContainer.scrollTop = logContainer.scrollHeight;
}}
</script>
"""
        log_html = f"""
<div id="{element_id}-wrapper">
  <div id="{element_id}" style="
      height: {height}px;
      overflow-y: auto;
      padding: 0.5rem;
      background: {background};
      color: {text_color};
      font-family: 'Fira Code', monospace;
      font-size: 0.85rem;
      border-radius: 0;
      border: 1px solid rgba(148, 163, 184, 0.25);
      white-space: pre-wrap;
      line-height: 1.35;
  ">
  {html.escape(content)}
  </div>
</div>
{auto_scroll_script}
"""
        st.components.v1.html(log_html, height=height + 40)
    else:
        st.text_area(
            label or "Log output",
            value=content,
            height=height,
            disabled=True,
            label_visibility="collapsed",
        )


def show_blocking_overlay():
    placeholder = st.empty()
    placeholder.markdown(BLOCK_UI_HTML, unsafe_allow_html=True)
    return placeholder


def hide_blocking_overlay(placeholder) -> None:
    try:
        placeholder.empty()
    except Exception:
        pass

# URL Configuration Management Functions
def save_url_configs(url_configs):
    """Save URL configurations to the database, replacing existing ones."""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Clear existing configurations
        cursor.execute("DELETE FROM url_configs")
        
        # Insert new configurations
        for config in url_configs:
            if config["url"].strip():  # Only save non-empty URLs
                cursor.execute("""
                    INSERT INTO url_configs (url, recordset, depth, exclude_paths, include_lang_prefixes)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    config["url"],
                    config["recordset"],
                    config["depth"],
                    config["exclude_paths"],
                    config["include_lang_prefixes"]
                ))
        
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()
        conn.close()

def load_url_configs():
    """Load URL configurations from the database."""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT url, recordset, depth, exclude_paths, include_lang_prefixes
            FROM url_configs
            ORDER BY id
        """)
        
        configs = []
        for row in cursor.fetchall():
            url, recordset, depth, exclude_paths, include_lang_prefixes = row
            configs.append({
                "url": url,
                "recordset": recordset,
                "depth": depth,
                "exclude_paths": exclude_paths or [],
                "include_lang_prefixes": include_lang_prefixes or []
            })
        
        return configs
    except Exception as e:
        # If table doesn't exist or other error, return empty list
        return []
    finally:
        cursor.close()
        conn.close()

def initialize_default_url_configs():
    """Initialize default URL configurations if none exist."""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT COUNT(*) FROM url_configs")
        count = cursor.fetchone()[0]
        
        if count == 0:
            # Add some default configurations
            default_configs = [
                {
                    "url": "https://www.europa-uni.de",
                    "recordset": "university_main",
                    "depth": 2,
                    "exclude_paths": ["/en/", "/pl/", "/_ablage-alte-www/", "/site-euv/", "/site-zwe-ikm/"],
                    "include_lang_prefixes": ["/de/"]
                }
            ]
            
            for config in default_configs:
                cursor.execute("""
                    INSERT INTO url_configs (url, recordset, depth, exclude_paths, include_lang_prefixes)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    config["url"],
                    config["recordset"],
                    config["depth"],
                    config["exclude_paths"],
                    config["include_lang_prefixes"]
                ))
            
            conn.commit()
    except Exception:
        # Ignore errors during initialization
        pass
    finally:
        cursor.close()
        conn.close()

# LLM Configuration Management Functions
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_available_openai_models():
    """
    Fetch available OpenAI models dynamically from the API.
    Returns a curated list of the most useful chat models, filtering out:
    - Duplicate dated versions (keeps latest canonical)
    - Specialized variants (audio, realtime, etc.) unless commonly used
    - Legacy/deprecated models
    Cached for 1 hour to improve performance.
    """
    try:
        import openai
        client = openai.OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
        
        # Get all available models
        models = client.models.list()
        
        # Extract all GPT models first
        all_gpt_models = []
        for model in models.data:
            model_id = model.id
            if (model_id.startswith('gpt-') and 
                not any(exclude in model_id.lower() for exclude in ['embedding', 'whisper', 'tts', 'dall-e', 'ada', 'babbage', 'curie', 'davinci'])):
                all_gpt_models.append(model_id)
        
        # Smart filtering to get canonical/latest versions
        curated_models = []
        
        # Define preferred models in priority order
        preferred_patterns = [
            # GPT-5 series (latest)
            ('gpt-5-mini', r'^gpt-5-mini$'),
            ('gpt-5', r'^gpt-5$'),
            
            # GPT-4o series (current flagship)
            ('gpt-4o-mini', r'^gpt-4o-mini$'),
            ('gpt-4o', r'^gpt-4o$'),
            
            # GPT-4 series (established)
            ('gpt-4-turbo', r'^gpt-4-turbo$'),
            ('gpt-4', r'^gpt-4$'),
            
            # GPT-3.5 series (legacy but still useful)
            ('gpt-3.5-turbo', r'^gpt-3.5-turbo$'),
        ]
        
        # Add canonical models if they exist
        import re
        for display_name, pattern in preferred_patterns:
            matching_models = [m for m in all_gpt_models if re.match(pattern, m)]
            if matching_models:
                curated_models.append(matching_models[0])  # Take first match
        
        # Add some commonly used specialized models if available
        useful_specialized = [
            'chatgpt-4o-latest',  # Latest ChatGPT model
            'gpt-4o-2024-11-20',  # Specific stable version
            'gpt-4o-mini-2024-07-18',  # Specific stable mini version
        ]
        
        for specialized in useful_specialized:
            if specialized in all_gpt_models and specialized not in curated_models:
                curated_models.append(specialized)
        
        # If we have a good curated list, use it
        if len(curated_models) >= 5:  # Ensure we have reasonable selection
            return curated_models
        else:
            # Fallback: use all models but with better sorting
            model_priority = {
                'gpt-5-mini': 1, 'gpt-5': 2,
                'gpt-4o-mini': 3, 'gpt-4o': 4,
                'gpt-4-turbo': 5, 'gpt-4': 6,
                'gpt-3.5-turbo': 7
            }
            
            def sort_key(model_name):
                return (model_priority.get(model_name, 999), model_name)
            
            all_gpt_models.sort(key=sort_key)
            return all_gpt_models
            
    except Exception as e:
        print(f"⚠️ Could not fetch models from OpenAI API: {e}")
        # Fallback to curated hardcoded list
        return [
            "gpt-5-mini",
            "gpt-5", 
            "gpt-4o-mini",
            "gpt-4o", 
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo"
        ]

def create_llm_settings_table():
    """Create table for storing LLM configuration settings (future-ready with reasoning effort and verbosity)"""
    run_migrations(get_connection)
    conn = get_connection()
    cursor = conn.cursor()
    # Insert default settings if table is empty
    cursor.execute("SELECT COUNT(*) FROM llm_settings")
    if cursor.fetchone()[0] == 0:
        cursor.execute("""
            INSERT INTO llm_settings (model, parallel_tool_calls, reasoning_effort, text_verbosity, updated_by)
            VALUES (%s, %s, %s, %s, %s)
        """, ('gpt-4o-mini', True, 'medium', 'medium', 'system'))
    
    conn.commit()
    cursor.close()
    conn.close()

def save_llm_settings(model, parallel_tool_calls=True, reasoning_effort="medium", text_verbosity="medium", updated_by="admin"):
    """Save LLM settings to database (future-ready with reasoning effort and verbosity)"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Update the single row (we only keep one active configuration)
    cursor.execute(
        """
        UPDATE llm_settings SET 
            model = %s,
            parallel_tool_calls = %s,
            reasoning_effort = %s,
            text_verbosity = %s,
            updated_by = %s,
            updated_at = NOW()
        WHERE id = (SELECT MIN(id) FROM llm_settings)
    """, (model, parallel_tool_calls, reasoning_effort, text_verbosity, updated_by))
    
    conn.commit()
    cursor.close()
    conn.close()
    
    # Invalidate cached reads
    try:
        st.cache_data.clear()
    except Exception:
        pass

def supports_reasoning_effort(model_name):
    """Check if a model supports the reasoning.effort parameter"""
    # GPT-5 models support reasoning effort
    gpt5_models = ['gpt-5', 'gpt-5-mini', 'gpt-5-turbo']
    return any(gpt5_model in model_name.lower() for gpt5_model in gpt5_models)

def get_supported_verbosity_options(model_name):
    """Get supported verbosity options for a model"""
    if supports_reasoning_effort(model_name):  # GPT-5 models
        return ["low", "medium", "high"]
    else:  # GPT-4 models
        return ["medium"]  # Only medium is supported

def supports_full_verbosity(model_name):
    """Check if a model supports all verbosity options (low/medium/high)"""
    return supports_reasoning_effort(model_name)  # Same as reasoning effort for now

@st.cache_data(ttl=300, show_spinner=False)
def get_llm_settings():
    """Get current LLM settings from database (future-ready with reasoning effort and verbosity)"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT model, parallel_tool_calls, reasoning_effort, text_verbosity, updated_by, updated_at
        FROM llm_settings
        ORDER BY updated_at DESC
        LIMIT 1
    """
    )
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if result:
        return {
            'model': result[0],
            'parallel_tool_calls': result[1],
            'reasoning_effort': result[2],
            'text_verbosity': result[3],
            'updated_by': result[4],
            'updated_at': result[5]
        }
    else:
        # Return defaults if no settings found
        return {
            'model': 'gpt-4o-mini',
            'parallel_tool_calls': True,
            'reasoning_effort': 'medium',
            'text_verbosity': 'medium',
            'updated_by': 'system',
            'updated_at': None
        }

# Filter Settings Management Functions
def create_filter_settings_table():
    """Create table for storing web search filter settings (locale, domains, user_location, enable flag)."""
    run_migrations(get_connection)
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM filter_settings")
    if cursor.fetchone()[0] == 0:
        cursor.execute(
            "INSERT INTO filter_settings (web_search_enabled, web_locale, web_domains, web_domains_mode, web_userloc_type, web_userloc_country, web_userloc_city, web_userloc_region, web_userloc_timezone, updated_by) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (True, 'de-DE', [], 'include', 'approximate', None, None, None, None, 'system')
        )
    conn.commit()
    cursor.close()
    conn.close()


def save_filter_settings(settings, updated_by="admin"):
    """Save web search filter settings (enable flag, locale, domains, and user_location) to DB"""
    conn = get_connection()
    cursor = conn.cursor()

    enabled = bool(settings.get('web_search_enabled', True))
    web_locale = (settings.get('web_locale') or 'de-DE').strip()

    # Normalize domains: split, strip scheme/path, drop empties, dedupe
    domains_raw = settings.get('web_domains')
    if isinstance(domains_raw, str):
        parts = [p.strip() for p in domains_raw.replace(',', '\n').splitlines()]
        web_domains = [d for d in parts if d]
    else:
        web_domains = [str(d).strip() for d in (domains_raw or []) if str(d).strip()]

    def _normalize_domain(d: str) -> str:
        s = d.strip()
        if '://' in s:
            s = s.split('://', 1)[1]
        s = s.split('/', 1)[0]
        if s.startswith('www.'):
            s = s[4:]
        return s.lower()

    web_domains = [_normalize_domain(d) for d in web_domains if _normalize_domain(d)]
    seen = set(); web_domains_norm = []
    for d in web_domains:
        if d not in seen:
            web_domains_norm.append(d); seen.add(d)

    # Force mode to 'include' to satisfy API requirement for allowed_domains
    mode = 'include'

    ul_type = (settings.get('web_userloc_type') or 'approximate').strip()
    ul_country = (settings.get('web_userloc_country') or '').strip() or None
    ul_city = (settings.get('web_userloc_city') or '').strip() or None
    ul_region = (settings.get('web_userloc_region') or '').strip() or None
    ul_timezone = (settings.get('web_userloc_timezone') or '').strip() or None

    # New MCP fields
    dbis_mcp_enabled = bool(settings.get('dbis_mcp_enabled', True))
    dbis_org_id = (settings.get('dbis_org_id') or '').strip() or None

    cursor.execute(
        """
        UPDATE filter_settings SET 
            web_search_enabled = %s,
            web_locale = %s,
            web_domains = %s,
            web_domains_mode = %s,
            web_userloc_type = %s,
            web_userloc_country = %s,
            web_userloc_city = %s,
            web_userloc_region = %s,
            web_userloc_timezone = %s,
            dbis_mcp_enabled = %s,
            dbis_org_id = %s,
            updated_by = %s,
            updated_at = NOW()
        WHERE id = (SELECT MIN(id) FROM filter_settings)
        """,
        (enabled, web_locale, web_domains_norm, mode, ul_type, ul_country, ul_city, ul_region, ul_timezone,
         dbis_mcp_enabled, dbis_org_id, updated_by)
    )

    conn.commit(); cursor.close(); conn.close()
    # Invalidate cached reads
    try:
        st.cache_data.clear()
    except Exception:
        pass


@st.cache_data(ttl=120, show_spinner=False)
def get_filter_settings():
    """Get current web search filter settings from database"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT web_search_enabled, web_locale, web_domains, web_domains_mode,
               web_userloc_type, web_userloc_country, web_userloc_city,
               web_userloc_region, web_userloc_timezone,
               dbis_mcp_enabled, dbis_org_id,
               updated_by, updated_at
        FROM filter_settings
        ORDER BY updated_at DESC
        LIMIT 1
        """
    )
    row = cursor.fetchone(); cursor.close(); conn.close()
    if row:
        mode = row[3] or 'include'
        if mode not in ('include',):
            mode = 'include'
        return {
            'web_search_enabled': bool(row[0]) if row[0] is not None else True,
            'web_locale': row[1] or 'de-DE',
            'web_domains': row[2] or [],
            'web_domains_mode': mode,
            'web_userloc_type': row[4] or 'approximate',
            'web_userloc_country': row[5] or '',
            'web_userloc_city': row[6] or '',
            'web_userloc_region': row[7] or '',
            'web_userloc_timezone': row[8] or '',
            'dbis_mcp_enabled': bool(row[9]) if row[9] is not None else True,
            'dbis_org_id': row[10] or '',
            'updated_by': row[11],
            'updated_at': row[12]
        }
    else:
        return {
            'web_search_enabled': True,
            'web_locale': 'de-DE',
            'web_domains': [],
            'web_domains_mode': 'include',
            'web_userloc_type': 'approximate',
            'web_userloc_country': '',
            'web_userloc_city': '',
            'web_userloc_region': '',
            'web_userloc_timezone': '',
            'dbis_mcp_enabled': True,
            'dbis_org_id': '',
            'updated_by': 'system',
            'updated_at': None
        }

# Request Classification Settings (DB-backed)
DEFAULT_REQUEST_CLASSIFICATIONS = [
    'library hours', 'book search', 'research help', 'account info',
    'facility info', 'policy question', 'technical support', 'other'
]

def create_request_classifications_table():
    """Create table to store request classification categories as a single TEXT[] row."""
    run_migrations(get_connection)
    conn = get_connection()
    cur = conn.cursor()
    # Seed defaults if empty
    cur.execute("SELECT COUNT(*) FROM request_classifications")
    if cur.fetchone()[0] == 0:
        cur.execute(
            "INSERT INTO request_classifications (categories, updated_by) VALUES (%s, %s)",
            (DEFAULT_REQUEST_CLASSIFICATIONS, 'system')
        )
    conn.commit()
    cur.close()
    conn.close()

@st.cache_data(ttl=300)
def get_request_classifications():
    """Return the most recently updated list of categories from DB (fallback to defaults)."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT categories FROM request_classifications ORDER BY updated_at DESC, id DESC LIMIT 1"
        )
        row = cur.fetchone()
        cur.close(); conn.close()
        cats = list(row[0]) if row and row[0] else []
        # Ensure 'other' exists
        if 'other' not in cats:
            cats.append('other')
        # Deduplicate while preserving order
        seen = set(); ordered = []
        for c in cats:
            if c and c not in seen:
                ordered.append(c); seen.add(c)
        return ordered or DEFAULT_REQUEST_CLASSIFICATIONS
    except Exception:
        return DEFAULT_REQUEST_CLASSIFICATIONS

def save_request_classifications(categories, updated_by='admin'):
    """Persist categories (TEXT[]) to DB, keeping a single active row (update the oldest row)."""
    # Normalize list: strip, drop empties, dedupe, ensure 'other'
    norm = []
    seen = set()
    for c in categories or []:
        c2 = (c or '').strip()
        if not c2:
            continue
        if c2 not in seen:
            norm.append(c2); seen.add(c2)
    if 'other' not in seen:
        norm.append('other')

    conn = get_connection()
    cur = conn.cursor()
    # If a row exists, update it; else insert
    cur.execute("SELECT id FROM request_classifications ORDER BY id ASC LIMIT 1")
    row = cur.fetchone()
    if row:
        cur.execute(
            "UPDATE request_classifications SET categories=%s, updated_by=%s, updated_at=NOW() WHERE id=%s",
            (norm, updated_by, row[0])
        )
    else:
        cur.execute(
            "INSERT INTO request_classifications (categories, updated_by) VALUES (%s, %s)",
            (norm, updated_by)
        )
    conn.commit(); cur.close(); conn.close()
    # Invalidate cached reads
    try:
        st.cache_data.clear()
    except Exception:
        pass

def get_document_status_counts() -> dict[str, int]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT
            COUNT(*) AS total_docs,
            COUNT(*) FILTER (WHERE vector_file_id IS NOT NULL) AS vectorized_docs,
            COUNT(*) FILTER (WHERE vector_file_id IS NULL AND (no_upload IS FALSE OR no_upload IS NULL)) AS non_vectorized_docs,
            COUNT(*) FILTER (WHERE vector_file_id IS NULL AND old_file_id IS NULL AND (no_upload IS FALSE OR no_upload IS NULL)) AS new_unsynced_count,
            COUNT(*) FILTER (
                WHERE (vector_file_id IS NULL AND old_file_id IS NOT NULL AND (no_upload IS FALSE OR no_upload IS NULL))
                   OR (no_upload IS TRUE AND vector_file_id IS NOT NULL)
            ) AS pending_resync_count,
            COUNT(*) FILTER (WHERE no_upload IS TRUE) AS excluded_docs
        FROM documents
    """)
    row = cur.fetchone()
    cur.close(); conn.close()
    keys = ("total_docs","vectorized_docs","non_vectorized_docs","new_unsynced_count","pending_resync_count","excluded_docs")
    return dict(zip(keys, row))
