import streamlit as st
from psycopg2.extras import DictCursor
from datetime import datetime, timedelta
import json
from pathlib import Path
from utils import get_connection, admin_authentication, render_sidebar

BASE_DIR = Path(__file__).parent.parent

LOG_SVG = (BASE_DIR / "assets" / "search_activity.svg").read_text()

authenticated = admin_authentication()
render_sidebar(authenticated)

st.set_page_config(page_title="Logging & Analytics", layout="wide")

# --- Read logs from DB ---
def read_logs(limit=200, error_code=None):
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=DictCursor)

    if error_code and error_code != "All":
        query = "SELECT * FROM log_table WHERE error_code = %s ORDER BY timestamp DESC LIMIT %s"
        cursor.execute(query, (error_code, limit))
    else:
        query = "SELECT * FROM log_table ORDER BY timestamp DESC LIMIT %s"
        cursor.execute(query, (limit,))

    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

# --- Delete all logs ---
def delete_logs():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM log_table")
    conn.commit()
    cursor.close()
    conn.close()

if authenticated:
    #st.title("📊 Logging & Analytics")
    st.markdown(
        f"""
        <h1 style="display:flex; align-items:center; gap:.5rem; margin:0;">
            {LOG_SVG}
            Logging & Analytics
        </h1>
        """,
        unsafe_allow_html=True
    )
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Interaction Logs", "Session Analytics", "System Metrics"])
    
    with tab1:
        # Admin-only content
        filter_code = st.selectbox("Filter by error code", options=["All", "E00", "E01", "E02"])
        logs = read_logs(limit=200, error_code=filter_code)
        st.header("View Interaction Logs")

        st.markdown(f"### Showing {len(logs)} log entries")

        for entry in logs:
            st.markdown(f"**{entry['timestamp']}**, **Code:** `{entry.get('error_code') or "OK"}`, **Citations:** {entry.get('citation_count', 0)}", unsafe_allow_html=False)
            with st.expander("View Details", expanded=False):
                st.info(f"{entry['user_input']}", icon=":material/face:")
                if entry.get('error_code') == "E03":
                    st.error(f"{entry['assistant_response']}", icon=":material/robot_2:")
                elif entry.get('error_code') == "E02":
                    st.warning(f"{entry['assistant_response']}", icon=":material/robot_2:")
                else:
                    st.success(f"{entry['assistant_response']}", icon=":material/robot_2:")
                if entry.get('citations'):
                    citations = entry.get('citations')
                    
                    # Ensure citations is parsed into a dictionary
                    if isinstance(citations, str):
                        try:
                            citations = json.loads(citations)  # Parse stringified JSON into a dictionary
                        except json.JSONDecodeError as e:
                            st.error(f"Failed to parse citations: {e}")
                            citations = {}  # Set citations to an empty dictionary if parsing fails
                    
                    markdown_list = ""
                    for citation in citations.values():  # Iterate over the values of the dictionary
                        title = citation.get("title", "Untitled")
                        url = citation.get("url", "#")
                        file_id = citation.get("file_id", "Unknown")
                        file_name = citation.get("file_name", "Unknown")
                        markdown_list += f"* [{title}]({url}) (Vector Store File: {file_name}, ID: `{file_id}`)\n"
                    st.success(markdown_list)

            st.markdown("---")

        if st.button("Delete All Logs", icon=":material/delete_forever:"):
            delete_logs()
            st.success("All logs have been deleted.")

        if st.button("Logout", icon=":material/logout:"):
            st.session_state["authenticated"] = False
            st.rerun()
        
    with tab2:
        st.subheader("Session-Based Analytics")
        
        # Session analytics using the new session_id column
        conn = get_connection()
        cursor = conn.cursor()
        
        # Session statistics
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT session_id) AS total_sessions,
                COUNT(*) AS total_interactions,
                COUNT(*)::numeric / NULLIF(COUNT(DISTINCT session_id), 0) AS avg_interactions_per_session
            FROM log_table
            WHERE session_id IS NOT NULL
        """)
        
        stats = cursor.fetchone()
        if stats:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Sessions", stats[0] or 0)
            with col2:
                st.metric("Total Interactions", stats[1] or 0)
            with col3:
                st.metric("Avg Interactions/Session", f"{stats[2]:.1f}" if stats[2] else "0.0")
        
        cursor.close()
        conn.close()
        
    with tab3:
        st.subheader("System Performance Metrics")
        # Add system metrics here
        st.info("System metrics dashboard - coming soon!")