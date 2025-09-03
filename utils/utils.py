import psycopg2
import streamlit as st
import hashlib

# Function to get a connection to the Postgres database
def get_connection():
    return psycopg2.connect(
        host=st.secrets["postgres"]["host"],
        port=st.secrets["postgres"]["port"],
        user=st.secrets["postgres"]["user"],
        password=st.secrets["postgres"]["password"],
        dbname=st.secrets["postgres"]["database"]
    )

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
        END $$;
    """)
    
    conn.commit()
    cursor.close()
    conn.close()
    
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
        st.title("🔒 Admin Login")
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
        st.sidebar.page_link("pages/admin.py", label="Edit Prompt")
        st.sidebar.page_link("pages/view_logs.py", label="View Logs")
        st.sidebar.page_link("pages/scrape.py", label="Scrape Web")
        st.sidebar.page_link("pages/vectorize.py", label="Vectorize Documents")
        #st.sidebar.page_link("pages/manage_users.py", label="👥 Manage Users")
        
        st.sidebar.button("🔓 Logout", on_click=lambda: st.session_state.update({"authenticated": False}))
    else:
        st.sidebar.page_link("pages/admin.py", label="🔒 Admin Login")

# Functions to save a document to the knowledge base
def compute_sha256(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()
