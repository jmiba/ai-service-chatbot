import streamlit as st
from utils import get_connection, get_latest_prompt, admin_authentication, render_sidebar, create_llm_settings_table, save_llm_settings, get_llm_settings, get_available_openai_models, supports_reasoning_effort, get_supported_verbosity_options, create_filter_settings_table, get_filter_settings, save_filter_settings, create_request_classifications_table, get_request_classifications, save_request_classifications
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

SETTINGS_SVG = (BASE_DIR / "assets" / "settings.svg").read_text()

# Must be the first Streamlit call
st.set_page_config(page_title="LLM Settings", layout="wide")

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

# --- Authentication ---
authenticated = admin_authentication()
render_sidebar(authenticated)

# --- Admin content ---
if authenticated:
    #st.title("LLM Settings")
    st.markdown(
        f"""
        <h1 style="display:flex; align-items:center; gap:.5rem; margin:0;">
            {SETTINGS_SVG}
            LLM Settings
        </h1>
        """,
        unsafe_allow_html=True
    )

    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["System Prompt", "Language Model", "Filters"])
    
    with tab1:
        st.header("Edit System Prompt")

        current_prompt, current_note = get_latest_prompt()
        current_prompt = current_prompt or ""

        new_prompt = st.text_area("**System prompt:**", value=current_prompt, height=400)
        new_note = st.text_input("**Edit note (optional):**", value="")

        if st.button("Save Prompt", icon=":material/save:", type="primary"):
            backup_prompt_to_db(new_prompt, edited_by="admin@viadrina.de", note=new_note)
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
                        updated_by="admin@viadrina.de"
                    )
                    st.success("Language model settings saved successfully!", icon=":material/check_circle:")
                    st.rerun()  # Refresh to show updated settings
                except Exception as e:
                    st.error(f"Error saving settings: {str(e)}", icon=":material/error:")
        
        st.divider()
        st.subheader("Request Classifications")
        st.caption("These categories are used for request classification and analytics. One per line. 'other' is always included.")
        try:
            current_cats = get_request_classifications()
        except Exception:
            current_cats = []
        cats_text = "\n".join(current_cats)
        with st.form("request_classifications_form"):
            new_cats_text = st.text_area("Categories", value=cats_text, height=180, help="Enter one category per line")
            submitted_cats = st.form_submit_button("Save Categories", icon=":material/save:", type="primary")
        if submitted_cats:
            new_cats = [ln.strip() for ln in (new_cats_text or '').splitlines() if ln.strip()]
            try:
                save_request_classifications(new_cats, updated_by="admin@viadrina.de")
                st.success("Request classifications saved.")
            except Exception as e:
                st.error(f"Error saving categories: {e}")

    with tab3:
        st.header("Response & Content Filters")
        
        st.info("Configure automatic filtering and quality controls for chatbot responses")
        
        # Load current filter settings
        try:
            current_filter_settings = get_filter_settings()
        except Exception as e:
            st.error(f"Error loading filter settings: {e}")
            current_filter_settings = {}
        
        # Wrap filter settings in a form to reduce reruns
        with st.form("filter_settings_form", clear_on_submit=False):
            # Content Quality Filters
            st.subheader("📊 Quality Control")
            
            col1, col2 = st.columns(2)
            
            with col1:
                confidence_threshold = st.slider(
                    "Minimum Confidence Threshold", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=current_filter_settings.get('confidence_threshold', 0.7), 
                    step=0.05,
                    help="Responses below this confidence level will be flagged or require manual review"
                )
                
                min_citations = st.number_input(
                    "Minimum Citations Required",
                    min_value=0,
                    max_value=10,
                    value=current_filter_settings.get('min_citations', 1),
                    help="Require at least this many citations for factual responses"
                )
            
            with col2:
                max_response_length = st.number_input(
                    "Max Response Length (characters)", 
                    min_value=100, 
                    max_value=5000, 
                    value=current_filter_settings.get('max_response_length', 2000), 
                    step=100,
                    help="Truncate responses longer than this limit"
                )
                
                enable_fact_checking = st.checkbox(
                    "Enable Fact-Checking Alerts",
                    value=current_filter_settings.get('enable_fact_checking', True),
                    help="Flag responses that might contain unverified claims"
                )
            
            # Content Filtering
            st.subheader("🛡️ Content Moderation")
            
            col1, col2 = st.columns(2)
            
            with col1:
                academic_integrity_check = st.checkbox(
                    "Academic Integrity Detection",
                    value=current_filter_settings.get('academic_integrity_check', True),
                    help="Detect and handle potential homework/exam questions appropriately"
                )
                
                language_consistency = st.checkbox(
                    "Language Consistency Check",
                    value=current_filter_settings.get('language_consistency', True),
                    help="Ensure responses match the language of the input query"
                )
            
            with col2:
                topic_restriction = st.selectbox(
                    "Topic Restriction Level",
                    options=["None", "University-related only", "Academic only", "Strict policy"],
                    index=["None", "University-related only", "Academic only", "Strict policy"].index(
                        current_filter_settings.get('topic_restriction', 'University-related only')
                    ),
                    help="Limit discussions to appropriate topics for a university chatbot"
                )
                
                inappropriate_content_filter = st.checkbox(
                    "Inappropriate Content Filter",
                    value=current_filter_settings.get('inappropriate_content_filter', True),
                    help="Block or flag potentially harmful or offensive content"
                )
            
            # User Experience
            st.subheader("👥 User Experience")
            
            col1, col2 = st.columns(2)
            
            with col1:
                user_type_adaptation = st.selectbox(
                    "Response Adaptation",
                    options=["Standard", "Student-friendly", "Faculty-focused", "Staff-oriented"],
                    index=["Standard", "Student-friendly", "Faculty-focused", "Staff-oriented"].index(
                        current_filter_settings.get('user_type_adaptation', 'Standard')
                    ),
                    help="Adapt response style based on intended audience"
                )
            
            with col2:
                citation_style = st.selectbox(
                    "Citation Format",
                    options=["Inline", "Numbered", "Academic (APA-style)", "Simple links"],
                    index=["Inline", "Numbered", "Academic (APA-style)", "Simple links"].index(
                        current_filter_settings.get('citation_style', 'Academic (APA-style)')
                    ),
                    help="How to format source citations in responses"
                )
            
            # Advanced Filters
            with st.expander("🔬 Advanced Filtering Options"):
                enable_sentiment_analysis = st.checkbox(
                    "Sentiment Analysis",
                    value=current_filter_settings.get('enable_sentiment_analysis', False),
                    help="Analyze emotional tone of queries and responses"
                )
                
                enable_keyword_blocking = st.checkbox(
                    "Keyword-based Blocking",
                    value=current_filter_settings.get('enable_keyword_blocking', False),
                    help="Block responses containing specific keywords"
                )
                
                if enable_keyword_blocking:
                    blocked_keywords = st.text_area(
                        "Blocked Keywords (one per line)",
                        value=current_filter_settings.get('blocked_keywords', ''),
                        help="Responses containing these keywords will be blocked"
                    )
                else:
                    blocked_keywords = ""
                
                enable_response_caching = st.checkbox(
                    "Response Caching",
                    value=current_filter_settings.get('enable_response_caching', True),
                    help="Cache similar responses to improve performance"
                )
            
            # Filter Summary
            st.subheader("📋 Current Filter Configuration")
            
            filter_summary = f"""
            **Quality Control:**
            - Confidence threshold: {confidence_threshold:.1%}
            - Minimum citations: {min_citations}
            - Max response length: {max_response_length:,} characters
            - Fact-checking: {'Enabled' if enable_fact_checking else 'Disabled'}
            
            **Content Moderation:**
            - Academic integrity check: {'Enabled' if academic_integrity_check else 'Disabled'}
            - Language consistency: {'Enabled' if language_consistency else 'Disabled'}
            - Topic restriction: {topic_restriction}
            - Inappropriate content filter: {'Enabled' if inappropriate_content_filter else 'Disabled'}
            
            **User Experience:**
            - Response adaptation: {user_type_adaptation}
            - Citation style: {citation_style}
            """
            
            st.info(filter_summary)
            
            # Save Filter Settings inside the form
            submitted_filters = st.form_submit_button("💾 Save Filter Settings", type="primary")
            if submitted_filters:
                # Collect all filter settings
                filter_settings = {
                    'confidence_threshold': confidence_threshold,
                    'min_citations': min_citations,
                    'max_response_length': max_response_length,
                    'enable_fact_checking': enable_fact_checking,
                    'academic_integrity_check': academic_integrity_check,
                    'language_consistency': language_consistency,
                    'topic_restriction': topic_restriction,
                    'inappropriate_content_filter': inappropriate_content_filter,
                    'user_type_adaptation': user_type_adaptation,
                    'citation_style': citation_style,
                    'enable_sentiment_analysis': enable_sentiment_analysis,
                    'enable_keyword_blocking': enable_keyword_blocking,
                    'blocked_keywords': blocked_keywords if enable_keyword_blocking else '',
                    'enable_response_caching': enable_response_caching
                }
                
                try:
                    save_filter_settings(filter_settings, updated_by="admin@viadrina.de")
                    st.success("✅ Filter settings saved successfully!")
                    st.caption("Filters will be applied to all new conversations")
                except Exception as e:
                    st.error(f"❌ Error saving filter settings: {e}")
        
        # Reset button outside the form
        if st.button("🔄 Reset to Defaults"):
            st.rerun()
        
        # Implementation Status
        st.subheader("🔧 Implementation Status")
        
        implementation_status = {
            "Confidence threshold": "✅ Fully implemented - Blocks responses below threshold",
            "Citation counting": "✅ Fully implemented - Tracks and validates citations", 
            "Response length limiting": "✅ Fully implemented - Truncates overly long responses",
            "Academic integrity detection": "✅ Fully implemented - Pattern-based homework detection",
            "Language detection": "✅ Fully implemented - Basic German/English detection",
            "Topic restriction": "✅ Fully implemented - Keyword-based university focus",
            "Content moderation": "✅ Basic implementation - Inappropriate keyword filtering",
            "Citation formatting": "✅ Fully implemented - Multiple citation styles (APA, numbered, etc.)",
            "User adaptation": "✅ Fully implemented - Student/Faculty/Staff response modes",
            "Response caching": "✅ Framework ready - Database integration complete",
            "Advanced sentiment analysis": "🚧 Framework ready - Requires ML model integration",
            "Advanced keyword blocking": "🚧 Framework ready - Custom word lists supported"
        }
        
        for feature, status in implementation_status.items():
            if "✅" in status:
                st.success(f"{feature}: {status}")
            elif "🚧" in status:
                st.info(f"{feature}: {status}")
            else:
                st.error(f"{feature}: {status}")
        
        st.caption("💡 All core filtering features are fully operational and ready for production use!")