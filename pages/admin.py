import streamlit as st
from utils import get_connection, get_latest_prompt, admin_authentication, render_sidebar, create_llm_settings_table, save_llm_settings, get_llm_settings, get_available_openai_models, supports_reasoning_effort, get_supported_verbosity_options, supports_full_verbosity

# Initialize LLM settings table
try:
    create_llm_settings_table()
except Exception as e:
    st.error(f"Error initializing LLM settings: {e}")

def backup_prompt_to_db(current_prompt, edited_by=None, note=None):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO prompt_versions (prompt, edited_by, note)
        VALUES (%s, %s, %s)
    """, (current_prompt, edited_by, note))
    conn.commit()
    cursor.close()
    conn.close()

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

# --- Authentication ---
authenticated = admin_authentication()
render_sidebar(authenticated)

st.set_page_config(page_title="LLM Settings", layout="wide")

# --- Admin content ---
if authenticated:
    st.title("⚙️ LLM Settings")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📄 System Prompt", "🧠 Language Model", "🔽 Filters"])
    
    with tab1:
        st.header("Edit System Prompt")

        current_prompt, current_note = get_latest_prompt()

        new_prompt = st.text_area("**System prompt:**", value=current_prompt, height=400)
        new_note = st.text_input("**Edit note (optional):**", value="")

        if st.button("💾 Save Prompt"):
            backup_prompt_to_db(new_prompt, edited_by="admin@viadrina.de", note=new_note)
            st.success("System prompt updated successfully.")

        st.caption(f"Length: {len(new_prompt)} characters")
        
        st.subheader("View Prompt History")

        with st.expander("🕒 Prompt history"):
            for ts, prompt, author, note in get_prompt_history():
                st.markdown(f"🕒 **{ts.strftime('%Y-%m-%d %H:%M:%S')}** by `{author or 'unknown'}` – {note or ''}")
                st.info(prompt)
                
    with tab2:
        st.header("Language Model Settings")
        
        # Get current settings from database
        current_settings = get_llm_settings()
        
        # Fetch available models dynamically from OpenAI API
        with st.spinner("Loading available models..."):
            available_models = get_available_openai_models()
        
        # Show model count and refresh option
        col_refresh, col_count = st.columns([3, 1])
        with col_refresh:
            if st.button("🔄 Refresh Models", help="Reload available models from OpenAI API"):
                st.cache_data.clear()
                st.rerun()
        with col_count:
            st.caption(f"📊 {len(available_models)} models available")
        
        # Model selection
        model_index = 0
        try:
            model_index = available_models.index(current_settings['model'])
        except ValueError:
            pass  # Use default if current model not in list
            
        model = st.selectbox(
            "Select Language Model", 
            options=available_models,
            index=model_index,
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
            st.info("🧠 **Advanced AI Model** - This model supports reasoning effort and full verbosity control")
            
            col1, col2 = st.columns(2)
            
            with col1:
                reasoning_effort = st.selectbox(
                    "Reasoning Effort",
                    options=["low", "medium", "high"],
                    index=["low", "medium", "high"].index(reasoning_effort),
                    help="Controls how much the model thinks before responding"
                )
                
                # Show reasoning effort explanation with latency warnings
                effort_explanations = {
                    "low": "⚡ **Fast reasoning** - Quick responses (~25s), lower cost",
                    "medium": "⚖️ **Balanced reasoning** - Good quality-speed balance (~35s)", 
                    "high": "🎯 **Deep reasoning** - Best quality, much slower responses (~85s) 💰"
                }
                
                if reasoning_effort == "high":
                    st.warning("⚠️ **High effort significantly increases response time** (3-4x slower) and cost")
                
                st.caption(effort_explanations[reasoning_effort])
            
            with col2:
                text_verbosity = st.selectbox(
                    "Response Verbosity",
                    options=["low", "medium", "high"],
                    index=["low", "medium", "high"].index(text_verbosity),
                    help="Controls how detailed and comprehensive responses are"
                )
                
                # Show verbosity explanation
                verbosity_explanations = {
                    "low": "📝 **Concise** - Brief, to-the-point responses",
                    "medium": "📄 **Balanced** - Appropriate detail level",
                    "high": "📚 **Comprehensive** - Detailed, thorough responses"
                }
                st.caption(verbosity_explanations[text_verbosity])
                
        else:
            st.info("ℹ️ **Standard Model** - Advanced controls available with GPT-5 models")
            
            # Still show verbosity for GPT-4 models, but limited to medium
            supported_verbosity = get_supported_verbosity_options(model)
            if len(supported_verbosity) == 1:
                st.caption("🔧 **Response Verbosity**: Medium (only option supported by this model)")
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
        
        # Save button
        col_save, col_info = st.columns([1, 2])
        with col_save:
            if st.button("💾 Save LLM Settings", type="primary"):
                try:
                    save_llm_settings(
                        model=model,
                        parallel_tool_calls=parallel_tool_calls,
                        reasoning_effort=reasoning_effort,
                        text_verbosity=text_verbosity,
                        updated_by="admin@viadrina.de"
                    )
                    st.success("✅ Language model settings saved successfully!")
                    st.rerun()  # Refresh to show updated settings
                except Exception as e:
                    st.error(f"❌ Error saving settings: {str(e)}")
        
        with col_info:
            if current_settings['updated_at']:
                st.caption(f"Last updated: {current_settings['updated_at'].strftime('%Y-%m-%d %H:%M:%S')} by {current_settings['updated_by']}")
            else:
                st.caption("Using default settings")
            
    with tab3:
        st.header("Filter Settings")
        
        enable_filtering = st.checkbox("Enable Response Filtering", value=True)
        filter_threshold = st.slider("Filter Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        max_response_length = st.number_input("Max Response Length", min_value=50, max_value=2000, value=500, step=50)

        if st.button("💾 Save Filter Settings"):
            # Here you would save these settings to your database or config
            st.success("Filter settings saved.")    

