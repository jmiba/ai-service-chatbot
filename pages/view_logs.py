import streamlit as st
import json
from pathlib import Path

st.set_page_config(page_title="Admin Logs", layout="wide")
st.title("🔐 Admin Panel – Interaction Logs")

# --- Simple Auth Function ---
def authenticate():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if st.session_state["authenticated"]:
        return True

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        correct_username = st.secrets["admin"]["username"]
        correct_password = st.secrets["admin"]["password"]

        if username == correct_username and password == correct_password:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Invalid credentials.")
    return False

# Custom navigation menu using page_link
st.sidebar.page_link("app.py", label="💬 Chat Assistant")
st.sidebar.page_link("pages/view_logs.py", label="🔒 Admin")  

# --- Only show content after auth ---
if authenticate():

    def read_logs(limit=200):
        log_path = Path("logs/interaction_log.jsonl")
        if not log_path.exists():
            return []

        with log_path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f.readlines()[-limit:]]

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
        st.experimental_rerun()
