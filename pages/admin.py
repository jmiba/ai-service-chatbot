import streamlit as st
from utils import get_connection, get_latest_prompt, admin_authentication, render_sidebar, create_llm_settings_table, save_llm_settings, get_llm_settings, get_available_openai_models, supports_reasoning_effort, get_supported_verbosity_options, create_filter_settings_table, get_filter_settings, save_filter_settings, create_request_classifications_table, get_request_classifications, save_request_classifications
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

SETTINGS_SVG = (BASE_DIR / "assets" / "settings.svg").read_text()

# Must be the first Streamlit call
st.set_page_config(page_title="LLM & API Settings", layout="wide")

# Initialize LLM settings table
def _ensure_prompt_versions_table():
    try:
        conn = get_connection()
        cur = conn.cursor()
        # Minimal, portable-ish schema without PK (sufficient for history listing)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS prompt_versions (
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                prompt TEXT NOT NULL,
                edited_by VARCHAR(255),
                note TEXT
            )
        """)
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        st.error(f"Error ensuring prompt_versions table: {e}")

try:
    create_llm_settings_table()
    create_filter_settings_table()
    create_request_classifications_table()
    _ensure_prompt_versions_table()
except Exception as e:
    st.error(f"Error initializing settings tables: {e}")

# Helper to get admin email for audit fields
def _updated_by_default():
    return st.session_state.get("admin_email") or st.secrets.get("ADMIN_EMAIL") or "Admin"

def backup_prompt_to_db(current_prompt, edited_by=None, note=None):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO prompt_versions (prompt, edited_by, note)
            VALUES (%s, %s, %s)
        """, (current_prompt, edited_by, note))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        st.error(f"Error saving prompt: {e}")

def get_prompt_history(limit=10):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT timestamp, prompt, edited_by, note
        FROM prompt_versions
        ORDER BY timestamp DESC
        LIMIT %s
    """, (limit,))
    history = cursor.fetchall()
    cursor.close()
    conn.close()
    return history

@st.cache_data(ttl=900)
def _load_available_models_cached():
    return get_available_openai_models()

def load_css(file_path):
    with open(BASE_DIR / file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("css/styles.css")

# --- Authentication ---
authenticated = admin_authentication()
render_sidebar(authenticated)

# --- Admin content ---
if authenticated:
    st.markdown(
        f"""
        <h1 style="display:flex; align-items:center; gap:.5rem; margin:0;">
            {SETTINGS_SVG}
            LLM & API Settings
        </h1>
        """,
        unsafe_allow_html=True
    )
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["System Prompt", "Language Model", "Request Categories", "Filters"])
    
    with tab1:
        st.header("Edit System Prompt")

        current_prompt, current_note = get_latest_prompt()
        current_prompt = current_prompt or ""

        new_prompt = st.text_area("**System prompt:**", value=current_prompt, height=400)
        new_note = st.text_input("**Edit note (optional):**", value="")

        if st.button("Save Prompt", icon=":material/save:", type="primary"):
            backup_prompt_to_db(new_prompt, edited_by=(st.session_state.get("admin_email") or "Admin"), note=new_note)
            st.success("System prompt updated successfully.")

        st.caption(f"Length: {len(new_prompt)} characters")
        
        st.subheader("View Prompt History")

        with st.expander("Prompt history", icon=":material/history:", expanded=False):
            for ts, prompt, author, note in get_prompt_history():
                # Safe timestamp formatting
                try:
                    ts_str = ts.strftime('%Y-%m-%d %H:%M:%S') if ts else "unknown time"
                except Exception:
                    ts_str = str(ts) if ts is not None else "unknown time"
                st.markdown(f"**{ts_str}** by `{author or 'unknown'}` – {note or ''}")
                st.info(prompt)
                
    with tab2:
        st.header("Language Model Settings")
        
        # Get current settings from database
        current_settings = get_llm_settings()
        
        # Fetch available models dynamically from OpenAI API (cached)
        with st.spinner("Loading available models..."):
            available_models = _load_available_models_cached()
        
        # Show model count and refresh option
        col_refresh, col_count = st.columns([3, 1])
        with col_refresh:
            if st.button("Refresh Models", icon=":material/refresh:", help="Reload available models from OpenAI API"):
                st.cache_data.clear()
                st.rerun()
        with col_count:
            st.caption(f"📊 {len(available_models)} models available")
        
        if not available_models:
            st.error("No models available. Please check OpenAI credentials/network and try refreshing.")
            st.stop()
        
        # Wrap model settings in a form to reduce reruns
        with st.form("llm_settings_form", clear_on_submit=False):
            # Model selection
            model_index = 0
            try:
                model_index = available_models.index(current_settings['model'])
            except Exception:
                pass  # Use default if current model not in list
                
            model = st.selectbox(
                "Select Language Model", 
                options=available_models,
                index=min(model_index, max(0, len(available_models) - 1)),
                help="Choose from available OpenAI models (fetched dynamically from API). Models are prioritized by capability and cost-effectiveness."
            )
            
            # Parallel tool calls setting
            parallel_tool_calls = st.checkbox(
                "Enable Parallel Tool Calls",
                value=current_settings['parallel_tool_calls'],
                help="Allow multiple tools to be called simultaneously for faster responses"
            )
            
            # Model-aware reasoning effort setting (only for GPT-5 models)
            reasoning_effort = current_settings.get('reasoning_effort', 'medium')
            text_verbosity = current_settings.get('text_verbosity', 'medium')
            
            # Show advanced controls for GPT-5 models
            if supports_reasoning_effort(model):
                st.info("Advanced AI Model - This model supports reasoning effort and full verbosity control", icon=":material/rocket:")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    reasoning_effort = st.selectbox(
                        "Reasoning Effort",
                        options=["low", "medium", "high"],
                        index=["low", "medium", "high"].index(reasoning_effort) if reasoning_effort in ["low", "medium", "high"] else 1,
                        help="Controls how much the model thinks before responding"
                    )
                    
                    # Show reasoning effort explanation with latency warnings
                    effort_explanations = {
                        "low": "Fast reasoning - Quick responses (~25s), lower cost",
                        "medium": "Balanced reasoning - Good quality-speed balance (~35s)", 
                        "high": "Deep reasoning - Best quality, much slower responses (~85s)"
                    }
                    
                    if reasoning_effort == "high":
                        st.warning("High effort significantly increases response time (3-4x slower) and cost")
                    
                    st.caption(effort_explanations.get(reasoning_effort, ""))
                
                with col2:
                    supported_verbosity = get_supported_verbosity_options(model) or ["medium"]
                    text_verbosity = st.selectbox(
                        "Response Verbosity",
                        options=supported_verbosity,
                        index=supported_verbosity.index(text_verbosity) if text_verbosity in supported_verbosity else 0,
                        help="Controls how detailed and comprehensive responses are"
                    )
            else:
                st.info("Standard Model - Advanced controls available with GPT-5 models", icon=":material/info:")
                supported_verbosity = get_supported_verbosity_options(model) or ["medium"]
                if len(supported_verbosity) == 1:
                    st.caption("Response Verbosity: Medium (only option supported by this model)")
                    text_verbosity = "medium"
                else:
                    text_verbosity = st.selectbox(
                        "Response Verbosity",
                        options=supported_verbosity,
                        index=supported_verbosity.index(text_verbosity) if text_verbosity in supported_verbosity else 0,
                        help="Controls response detail level"
                    )
            
            # Settings preview
            st.subheader("Current Configuration")
            config_info = f"""
            **Model**: {model}  
            **Parallel Tool Calls**: {'Enabled' if parallel_tool_calls else 'Disabled'}
            **Response Verbosity**: {text_verbosity.title()}
            """
            
            if supports_reasoning_effort(model):
                config_info += f"\n**Reasoning Effort**: {reasoning_effort.title()}"
            
            st.info(config_info)
            
            # Save button inside the form
            col_save, col_info = st.columns([1, 2])
            with col_save:
                submitted_llm = st.form_submit_button("Save LLM Settings", icon=":material/save:" ,type="primary")
            with col_info:
                if current_settings['updated_at']:
                    st.caption(f"Last updated: {current_settings['updated_at'].strftime('%Y-%m-%d %H:%M:%S')} by {current_settings['updated_by']}")
                else:
                    st.caption("Using default settings")
            
            if submitted_llm:
                try:
                    save_llm_settings(
                        model=model,
                        parallel_tool_calls=parallel_tool_calls,
                        reasoning_effort=reasoning_effort,
                        text_verbosity=text_verbosity,
                        updated_by=(st.session_state.get("admin_email") or "Admin")
                    )
                    st.success("Language model settings saved successfully!", icon=":material/check_circle:")
                    st.rerun()  # Refresh to show updated settings
                except Exception as e:
                    st.error(f"Error saving settings: {str(e)}", icon=":material/error:")
        
    with tab3:
        st.subheader("Request Categorization")
        st.caption("These categories are used for the classification of interactions and analytics. One per line. The fall-back catergory 'other' will be added automatically.")
        try:
            current_cats = get_request_classifications()
        except Exception:
            current_cats = []
        cats_text = "\n".join(current_cats)
        with st.form("request_classifications_form"):
            new_cats_text = st.text_area("Categories", value=cats_text, height=400, help="Enter one category per line")
            submitted_cats = st.form_submit_button("Save Categories", icon=":material/save:", type="primary")
        if submitted_cats:
            new_cats = [ln.strip() for ln in (new_cats_text or '').splitlines() if ln.strip()]
            try:
                save_request_classifications(new_cats, updated_by=(st.session_state.get("admin_email") or "Admin"))
                st.success("Request classifications saved.")
            except Exception as e:
                st.error(f"Error saving categories: {e}")

    with tab4:
        st.header("Web Search Filters")
        st.info("Configure domain allow-list and user location for the OpenAI web search tool")
        
        # Load current filter settings
        try:
            current_filter_settings = get_filter_settings()
        except Exception as e:
            st.error(f"Error loading filter settings: {e}")
            current_filter_settings = {}
        
        with st.form("filter_settings_form", clear_on_submit=False):
            enabled = st.toggle(
                "Enable Web Search",
                value=current_filter_settings.get('web_search_enabled', True),
                help="Turn web search tool on/off"
            )

            # Allowed domains (empty means web search will not run)
            domains_text = st.text_area(
                "Allowed Domains (one per line or comma-separated)",
                value="\n".join(current_filter_settings.get('web_domains', []) or []),
                height=120,
                help="Provide one or more domains (e.g., europa-uni.de). Leave empty to disable web search at runtime."
            )

            # User Location
            st.subheader("User Location")
            col_ul1, col_ul2, col_ul3 = st.columns([1,1,2])
            with col_ul1:
                ul_type = st.selectbox(
                    "Type",
                    options=["approximate","precise"],
                    index=["approximate","precise"].index(current_filter_settings.get('web_userloc_type','approximate')),
                    help="How precise the user location is"
                )
            with col_ul2:
                ul_country = st.text_input(
                    "Country",
                    value=current_filter_settings.get('web_userloc_country',''),
                    placeholder="US, DE, ..."
                )
            with col_ul3:
                ul_city = st.text_input(
                    "City",
                    value=current_filter_settings.get('web_userloc_city',''),
                    placeholder="New York, Berlin, ..."
                )

            # Additional optional fields
            col_ul4, col_ul5 = st.columns([1,2])
            with col_ul4:
                ul_region = st.text_input(
                    "Region (optional)",
                    value=current_filter_settings.get('web_userloc_region',''),
                    placeholder="Brandenburg, CA, ..."
                )
            with col_ul5:
                ul_timezone = st.text_input(
                    "Timezone (optional)",
                    value=current_filter_settings.get('web_userloc_timezone',''),
                    placeholder="Europe/Berlin, America/New_York"
                )

            submitted_filters = st.form_submit_button("Save Web Search Settings", type="primary", icon=":material/save:")
            if submitted_filters:
                # basic validations
                if ul_country and len(ul_country.strip()) not in (0,2):
                    st.error("Country should be a 2-letter code (ISO 3166-1 alpha-2), e.g., 'US', 'DE'.")
                    st.stop()

                settings = {
                    'web_search_enabled': enabled,
                    'web_domains': domains_text,
                    'web_userloc_type': ul_type,
                    'web_userloc_country': (ul_country or '').upper(),
                    'web_userloc_city': ul_city,
                    'web_userloc_region': ul_region,
                    'web_userloc_timezone': ul_timezone,
                }
                try:
                    save_filter_settings(settings, updated_by=(st.session_state.get("admin_email") or "admin@viadrina.de"))
                    st.success("Web search settings saved!", icon=":material/check_circle:")
                except Exception as e:
                    st.error(f"Error saving settings: {e}", icon=":material/error:")

        st.caption("These settings affect only the web_search tool; file_search remains unchanged.")
        
        # Reset button outside the form
        if st.button("Reset to Defaults", icon=":material/cached:",):
            st.rerun()

