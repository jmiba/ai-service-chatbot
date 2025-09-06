import psycopg2
import streamlit as st
import hashlib

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
            deleted BOOLEAN DEFAULT FALSE
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
                SELECT id, url, title, safe_title, crawl_date, lang, summary, tags, markdown_content, recordset, vector_file_id, page_type
                FROM documents
                ORDER BY updated_at DESC
                LIMIT %s
            """, (limit,))
        else:
            cursor.execute("""
                SELECT id, url, title, safe_title, crawl_date, lang, summary, tags, markdown_content, recordset, vector_file_id, page_type
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
            evaluation_notes TEXT
        );
    """)
    
    # Add new columns to existing table if they don't exist
    cursor.execute("""
        DO $$ 
        BEGIN
            BEGIN
                ALTER TABLE log_table ADD COLUMN confidence DECIMAL(3,2) DEFAULT 0.0;
            EXCEPTION
                WHEN duplicate_column THEN NULL;
            END;
            
            BEGIN
                ALTER TABLE log_table ADD COLUMN request_classification VARCHAR(50);
            EXCEPTION
                WHEN duplicate_column THEN NULL;
            END;
            
            BEGIN
                ALTER TABLE log_table ADD COLUMN evaluation_notes TEXT;
            EXCEPTION
                WHEN duplicate_column THEN NULL;
            END;
            
            BEGIN
                ALTER TABLE log_table ADD COLUMN session_id VARCHAR(36);
            EXCEPTION
                WHEN duplicate_column THEN NULL;
            END;
        END $$;
    """)
    
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
        st.title("🔑 Admin Login")
        password = st.text_input("Admin Password", type="password")
        if password == st.secrets["ADMIN_PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        elif password:
            st.error("Incorrect password.")
        return False
    return True


# Function to render the sidebar with common elements
def render_sidebar(authenticated=False):
    """
    Renders common sidebar elements.
    """
    st.sidebar.page_link("app.py", label="💬 Chat Assistant")
    
    if authenticated:
        st.sidebar.success("Authenticated as admin.")
        st.sidebar.page_link("pages/admin.py", label="⚙️ LLM Settings")
        st.sidebar.page_link("pages/logs.py", label="📊 Logs & Analytics")
        st.sidebar.page_link("pages/scrape.py", label="🌐 Content Indexing")
        st.sidebar.page_link("pages/vectorize.py", label="🗄️ Vector Store Management")
        #st.sidebar.page_link("pages/manage_users.py", label="👥 Manage Users")
        
        st.sidebar.button("🔓 Logout", on_click=lambda: st.session_state.update({"authenticated": False}))
    else:
        st.sidebar.page_link("pages/admin.py", label="🔑 Admin Login")

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
    """Create table for storing response and content filter settings"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS filter_settings (
            id SERIAL PRIMARY KEY,
            confidence_threshold DECIMAL(3,2) DEFAULT 0.70,
            min_citations INTEGER DEFAULT 1,
            max_response_length INTEGER DEFAULT 2000,
            enable_fact_checking BOOLEAN DEFAULT TRUE,
            academic_integrity_check BOOLEAN DEFAULT TRUE,
            language_consistency BOOLEAN DEFAULT TRUE,
            topic_restriction VARCHAR(50) DEFAULT 'University-related only',
            inappropriate_content_filter BOOLEAN DEFAULT TRUE,
            user_type_adaptation VARCHAR(50) DEFAULT 'Standard',
            citation_style VARCHAR(50) DEFAULT 'Academic (APA-style)',
            enable_sentiment_analysis BOOLEAN DEFAULT FALSE,
            enable_keyword_blocking BOOLEAN DEFAULT FALSE,
            blocked_keywords TEXT DEFAULT '',
            enable_response_caching BOOLEAN DEFAULT TRUE,
            updated_by VARCHAR(100),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)
    
    # Insert default settings if table is empty
    cursor.execute("SELECT COUNT(*) FROM filter_settings")
    if cursor.fetchone()[0] == 0:
        cursor.execute("""
            INSERT INTO filter_settings (updated_by) VALUES (%s)
        """, ('system',))
    
    conn.commit()
    cursor.close()
    conn.close()

def save_filter_settings(settings, updated_by="admin"):
    """Save filter settings to database"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Update the single row (we only keep one active configuration)
    cursor.execute("""
        UPDATE filter_settings SET 
            confidence_threshold = %s,
            min_citations = %s,
            max_response_length = %s,
            enable_fact_checking = %s,
            academic_integrity_check = %s,
            language_consistency = %s,
            topic_restriction = %s,
            inappropriate_content_filter = %s,
            user_type_adaptation = %s,
            citation_style = %s,
            enable_sentiment_analysis = %s,
            enable_keyword_blocking = %s,
            blocked_keywords = %s,
            enable_response_caching = %s,
            updated_by = %s,
            updated_at = NOW()
        WHERE id = (SELECT MIN(id) FROM filter_settings)
    """, (
        settings.get('confidence_threshold', 0.70),
        settings.get('min_citations', 1),
        settings.get('max_response_length', 2000),
        settings.get('enable_fact_checking', True),
        settings.get('academic_integrity_check', True),
        settings.get('language_consistency', True),
        settings.get('topic_restriction', 'University-related only'),
        settings.get('inappropriate_content_filter', True),
        settings.get('user_type_adaptation', 'Standard'),
        settings.get('citation_style', 'Academic (APA-style)'),
        settings.get('enable_sentiment_analysis', False),
        settings.get('enable_keyword_blocking', False),
        settings.get('blocked_keywords', ''),
        settings.get('enable_response_caching', True),
        updated_by
    ))
    
    conn.commit()
    cursor.close()
    conn.close()

def get_filter_settings():
    """Get current filter settings from database"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT confidence_threshold, min_citations, max_response_length, 
               enable_fact_checking, academic_integrity_check, language_consistency,
               topic_restriction, inappropriate_content_filter, user_type_adaptation,
               citation_style, enable_sentiment_analysis, enable_keyword_blocking,
               blocked_keywords, enable_response_caching, updated_by, updated_at
        FROM filter_settings
        ORDER BY updated_at DESC
        LIMIT 1
    """)
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if result:
        return {
            'confidence_threshold': float(result[0]),
            'min_citations': result[1],
            'max_response_length': result[2],
            'enable_fact_checking': result[3],
            'academic_integrity_check': result[4],
            'language_consistency': result[5],
            'topic_restriction': result[6],
            'inappropriate_content_filter': result[7],
            'user_type_adaptation': result[8],
            'citation_style': result[9],
            'enable_sentiment_analysis': result[10],
            'enable_keyword_blocking': result[11],
            'blocked_keywords': result[12],
            'enable_response_caching': result[13],
            'updated_by': result[14],
            'updated_at': result[15]
        }
    else:
        # Return defaults if no settings found
        return {
            'confidence_threshold': 0.70,
            'min_citations': 1,
            'max_response_length': 2000,
            'enable_fact_checking': True,
            'academic_integrity_check': True,
            'language_consistency': True,
            'topic_restriction': 'University-related only',
            'inappropriate_content_filter': True,
            'user_type_adaptation': 'Standard',
            'citation_style': 'Academic (APA-style)',
            'enable_sentiment_analysis': False,
            'enable_keyword_blocking': False,
            'blocked_keywords': '',
            'enable_response_caching': True,
            'updated_by': 'system',
            'updated_at': None
        }
