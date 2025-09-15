import psycopg2
import streamlit as st
import hashlib
from pathlib import Path
import json
from functools import lru_cache
import uuid

BASE_DIR = Path(__file__).parent.parent

LOGIN_SVG = (BASE_DIR / "assets" / "key.svg").read_text()

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
    conn = get_connection()
    cursor = conn.cursor()

    # Step 1: Create the documents table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            url TEXT UNIQUE NOT NULL,
            title TEXT,
            safe_title TEXT,
            crawl_date DATE,
            lang TEXT,
            summary TEXT,
            tags TEXT[], -- PostgreSQL supports arrays
            markdown_content TEXT,
            markdown_hash TEXT,
            recordset TEXT,
            vector_file_id VARCHAR(255),
            old_file_id VARCHAR(255),
            updated_at TIMESTAMP DEFAULT NOW(),
            page_type VARCHAR(50),
            no_upload BOOLEAN DEFAULT FALSE
        );
    """)
    
    # Step 2: Create a non-unique index on recordset for performance (optional but recommended)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_documents_recordset
        ON documents(recordset);
    """)

    conn.commit()
    cursor.close()
    conn.close()
    
    
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
                SELECT id, url, title, safe_title, crawl_date, lang, summary, tags, markdown_content, recordset, vector_file_id, page_type, no_upload
                FROM documents
                ORDER BY updated_at DESC
                LIMIT %s
            """, (limit,))
        else:
            cursor.execute("""
                SELECT id, url, title, safe_title, crawl_date, lang, summary, tags, markdown_content, recordset, vector_file_id, page_type, no_upload
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

# Function to create the log_table if it doesn't exist   
def create_log_table():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS log_table (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            session_id VARCHAR(36),
            user_input TEXT NOT NULL,
            assistant_response TEXT NOT NULL,
            error_code VARCHAR(10),
            citation_count INTEGER DEFAULT 0,
            citations JSONB,
            confidence DECIMAL(3,2) DEFAULT 0.0,
            request_classification VARCHAR(50),
            evaluation_notes TEXT,
            -- New usage/cost/model fields
            model VARCHAR(100),
            usage_input_tokens INTEGER,
            usage_output_tokens INTEGER,
            usage_total_tokens INTEGER,
            usage_reasoning_tokens INTEGER,
            api_cost_usd NUMERIC(12,6),
            -- Latency of main API call in milliseconds
            response_time_ms INTEGER
        );
    """)
    
    # Add new columns to existing table if they don't exist
    cursor.execute(
        """
        DO $$ 
        BEGIN
            BEGIN
                ALTER TABLE log_table ADD COLUMN IF NOT EXISTS confidence DECIMAL(3,2) DEFAULT 0.0;
            EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN
                ALTER TABLE log_table ADD COLUMN IF NOT EXISTS request_classification VARCHAR(50);
            EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN
                ALTER TABLE log_table ADD COLUMN IF NOT EXISTS evaluation_notes TEXT;
            EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN
                ALTER TABLE log_table ADD COLUMN IF NOT EXISTS session_id VARCHAR(36);
            EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN
                ALTER TABLE log_table ADD COLUMN IF NOT EXISTS model VARCHAR(100);
            EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN
                ALTER TABLE log_table ADD COLUMN IF NOT EXISTS usage_input_tokens INTEGER;
            EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN
                ALTER TABLE log_table ADD COLUMN IF NOT EXISTS usage_output_tokens INTEGER;
            EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN
                ALTER TABLE log_table ADD COLUMN IF NOT EXISTS usage_total_tokens INTEGER;
            EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN
                ALTER TABLE log_table ADD COLUMN IF NOT EXISTS usage_reasoning_tokens INTEGER;
            EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN
                ALTER TABLE log_table ADD COLUMN IF NOT EXISTS api_cost_usd NUMERIC(12,6);
            EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN
                ALTER TABLE log_table ADD COLUMN IF NOT EXISTS response_time_ms INTEGER;
            EXCEPTION WHEN duplicate_column THEN NULL; END;
        END $$;
        """
    )
    
    # Create index on session_id for efficient conversation queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_log_table_session_id ON log_table(session_id);
    """)
    
    conn.commit()
    cursor.close()
    conn.close()

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
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prompt_versions (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            prompt TEXT NOT NULL,
            edited_by TEXT,
            note TEXT
        );
    """)
    conn.commit()
    cursor.close()
    conn.close()

def create_url_configs_table():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS url_configs (
            id SERIAL PRIMARY KEY,
            url TEXT NOT NULL,
            recordset TEXT NOT NULL DEFAULT '',
            depth INTEGER NOT NULL DEFAULT 2,
            exclude_paths TEXT[] DEFAULT ARRAY['/en/', '/pl/', '/_ablage-alte-www/', '/site-euv/', '/site-zwe-ikm/'],
            include_lang_prefixes TEXT[] DEFAULT ARRAY['/de/'],
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """)
    conn.commit()
    cursor.close()
    conn.close()

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

def admin_authentication():
    """
    Authenticates the admin based on a password in st.secrets.
    Returns True if authenticated, False otherwise.
    """
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        #st.title("Admin Login")
        st.markdown(
            f"""
            <h1 style="display:flex; align-items:center; gap:.5rem; margin:0;">
                {LOGIN_SVG}
                Admin Login
            </h1>
            """,
            unsafe_allow_html=True
        )
        password = st.text_input("Admin Password", type="password")
        if password == st.secrets["ADMIN_PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        elif password:
            st.error("Incorrect password.")
        return False
    return True


# Function to render the sidebar with common elements
def render_sidebar(authenticated=False, show_debug=False):
    """
    Renders common sidebar elements.
    Args:
        authenticated: Whether user is authenticated
        show_debug: Whether to show debug controls (only for main chat page)
    Returns:
        debug_one: Debug state if show_debug=True, otherwise False
    """
    st.sidebar.page_link("app.py", label="Chat Assistant", icon=":material/chat_bubble:")
    
    # New chat
    if st.sidebar.button(
        "New chat",
        type="secondary",
        help="Start a fresh session",
        icon=":material/add_comment:",
        use_container_width=True,
        key="sidebar_new_chat_button",
    ):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        try:
            st.switch_page("app.py")
        except Exception:
            st.rerun()
    
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
        
        st.sidebar.button("Logout", on_click=lambda: st.session_state.update({"authenticated": False}), icon=":material/logout:")
        with st.sidebar.container(key="sidebar_bottom"):
            st.caption("Source code on [GitHub](https://github.com/jmiba/ai-service-chatbot)")
    else:
        with st.sidebar.container(key="sidebar_bottom"):
            st.page_link("pages/logs.py", label="Admin Login", icon=":material/key:")
            st.caption("Source code on [GitHub](https://github.com/jmiba/ai-service-chatbot)")
        
    
    return debug_one

# Functions to save a document to the knowledge base
def compute_sha256(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

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
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS llm_settings (
            id SERIAL PRIMARY KEY,
            model VARCHAR(100) NOT NULL DEFAULT 'gpt-4o-mini',
            parallel_tool_calls BOOLEAN DEFAULT TRUE,
            reasoning_effort VARCHAR(20) DEFAULT 'medium',
            text_verbosity VARCHAR(20) DEFAULT 'medium',
            updated_by VARCHAR(100),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)
    
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
    cursor.execute("""
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

def get_llm_settings():
    """Get current LLM settings from database (future-ready with reasoning effort and verbosity)"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT model, parallel_tool_calls, reasoning_effort, text_verbosity, updated_by, updated_at
        FROM llm_settings
        ORDER BY updated_at DESC
        LIMIT 1
    """)
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
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS filter_settings (
            id SERIAL PRIMARY KEY,
            web_search_enabled BOOLEAN DEFAULT TRUE,
            web_locale TEXT DEFAULT 'de-DE',
            web_domains TEXT[] DEFAULT ARRAY[]::TEXT[],
            web_domains_mode VARCHAR(16) DEFAULT 'include',
            web_userloc_type TEXT DEFAULT 'approximate',
            web_userloc_country TEXT,
            web_userloc_city TEXT,
            web_userloc_region TEXT,
            web_userloc_timezone TEXT,
            updated_by VARCHAR(100),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
    )
    # migrate columns if upgrading from older schema
    cursor.execute(
        """
        DO $$ BEGIN
            BEGIN ALTER TABLE filter_settings ADD COLUMN IF NOT EXISTS web_search_enabled BOOLEAN DEFAULT TRUE; EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN ALTER TABLE filter_settings ADD COLUMN IF NOT EXISTS web_locale TEXT DEFAULT 'de-DE'; EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN ALTER TABLE filter_settings ADD COLUMN IF NOT EXISTS web_domains TEXT[] DEFAULT ARRAY[]::TEXT[]; EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN ALTER TABLE filter_settings ADD COLUMN IF NOT EXISTS web_domains_mode VARCHAR(16) DEFAULT 'include'; EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN ALTER TABLE filter_settings ADD COLUMN IF NOT EXISTS web_userloc_type TEXT DEFAULT 'approximate'; EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN ALTER TABLE filter_settings ADD COLUMN IF NOT EXISTS web_userloc_country TEXT; EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN ALTER TABLE filter_settings ADD COLUMN IF NOT EXISTS web_userloc_city TEXT; EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN ALTER TABLE filter_settings ADD COLUMN IF NOT EXISTS web_userloc_region TEXT; EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN ALTER TABLE filter_settings ADD COLUMN IF NOT EXISTS web_userloc_timezone TEXT; EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN ALTER TABLE filter_settings ADD COLUMN IF NOT EXISTS updated_by VARCHAR(100); EXCEPTION WHEN duplicate_column THEN NULL; END;
            BEGIN ALTER TABLE filter_settings ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW(); EXCEPTION WHEN duplicate_column THEN NULL; END;
        END $$;
        """
    )
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
            updated_by = %s,
            updated_at = NOW()
        WHERE id = (SELECT MIN(id) FROM filter_settings)
        """,
        (enabled, web_locale, web_domains_norm, mode, ul_type, ul_country, ul_city, ul_region, ul_timezone, updated_by)
    )

    conn.commit(); cursor.close(); conn.close()


def get_filter_settings():
    """Get current web search filter settings from database"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT web_search_enabled, web_locale, web_domains, web_domains_mode,
               web_userloc_type, web_userloc_country, web_userloc_city,
               web_userloc_region, web_userloc_timezone,
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
            'updated_by': row[9],
            'updated_at': row[10]
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
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS request_classifications (
            id SERIAL PRIMARY KEY,
            categories TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
            updated_by TEXT,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
    )
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
    # Ensure table exists
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS request_classifications (
            id SERIAL PRIMARY KEY,
            categories TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
            updated_by TEXT,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
    )
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

