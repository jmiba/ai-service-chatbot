import streamlit as st
from psycopg2.extras import DictCursor
from datetime import datetime
import json
from utils import get_connection, admin_authentication, render_sidebar

authenticated = admin_authentication()
render_sidebar(authenticated)

st.set_page_config(page_title="Interaction Logs", layout="wide")
st.title("📄 View Interaction Logs")

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
    # Admin-only content
    filter_code = st.selectbox("Filter by error code", options=["All", "E00", "E01", "E02"])
    logs = read_logs(limit=200, error_code=filter_code)

    st.markdown(f"### Showing {len(logs)} log entries")

    for entry in logs:
        st.markdown(f"🕒 **{entry['timestamp']}**, 🔖 **Code:** `{entry.get('error_code') or "OK"}`, 📚 **Citations:** {entry.get('citation_count', 0)}", unsafe_allow_html=False)
        with st.expander("View Details", expanded=False):
            st.info(f"{entry['user_input']}")
            if entry.get('error_code') == "E01":
                st.error(f"{entry['assistant_response']}")
            elif entry.get('error_code') == "E02":
                st.warning(f"{entry['assistant_response']}")
            else:
                st.success(f"{entry['assistant_response']}")
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

    if st.button("🗑️ Delete All Logs"):
        delete_logs()
        st.success("All logs have been deleted.")

    if st.button("🔓 Logout"):
        st.session_state["authenticated"] = False
        st.rerun()
