import streamlit as st
import json
from pathlib import Path

st.set_page_config(page_title="Admin Logs", layout="wide")
st.title("📄 View Interaction Logs")

# --- Simple Auth Function ---
def authenticate():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if st.session_state["authenticated"]:
        return True

# Custom navigation menu using page_link
st.sidebar.page_link("app.py", label="💬 Chat Assistant")

# --- If not authenticated, show login ---
if not authenticate():
    st.sidebar.page_link("pages/admin.py", label="🔒 Admin")
    password = st.text_input("Admin Password", type="password")
    if password == st.secrets["ADMIN_PASSWORD"]:
        st.session_state.authenticated = True
        st.rerun()
    elif password:
        st.error("Incorrect password.")

# --- Only show content after auth ---
if authenticate():

    def read_logs(limit=200):
        log_path = Path("logs/interaction_log.jsonl")
        if not log_path.exists():
            return []

        with log_path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f.readlines()[-limit:]]
    st.sidebar.success("Authenticated as admin.")
    st.sidebar.page_link("pages/view_logs.py", label="📄 View Logs")
    #st.sidebar.page_link("pages/manage_users.py", label="👥 Manage Users")

    logs = read_logs()

    filter_code = st.selectbox("Filter by error code", options=["All", "E01", "E02"])
    if filter_code != "All":
        logs = [log for log in logs if log.get("error_code") == filter_code]

    st.markdown(f"### Showing {len(logs)} log entries")

    for entry in logs:
        st.markdown(f"""
⏱️ **{entry['timestamp']}**  
❓ **User:**  
{entry['user_input']}  
  
🤖 **Assistant:**  
{entry['assistant_response']}  
  
🔖 **Code:** `{entry.get('error_code') or "OK"}`  
  
📚 **Citations:** {entry.get('citation_count', 0)}  

---
    """, unsafe_allow_html=False)

    if st.button("🗑️ Delete All Logs"):
        Path("logs/interaction_log.jsonl").unlink(missing_ok=True)
        st.success("All logs have been deleted.")

    if st.button("🔓 Logout"):
        st.session_state["authenticated"] = False
        st.rerun()
