import streamlit as st
from utils import get_connection, get_latest_prompt, admin_authentication, render_sidebar

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

# --- Streamlit page setup ---
st.set_page_config(page_title="Admin Login", layout="wide")

# --- Authentication ---
authenticated = admin_authentication()

# --- Sidebar ---
render_sidebar(authenticated)

# --- Admin content ---
if authenticated:
    st.set_page_config(page_title="System Prompt", layout="wide")
    st.title("Edit System Prompt")

    current_prompt, current_note = get_latest_prompt()

    new_prompt = st.text_area("**Edit the system prompt:**", value=current_prompt, height=400)
    new_note = st.text_input("**Edit note (optional):**", value="")

    if st.button("💾 Save Prompt"):
        backup_prompt_to_db(new_prompt, edited_by="admin@viadrina.de", note=new_note)
        st.success("System prompt updated successfully.")

    st.caption(f"Length: {len(new_prompt)} characters")

    with st.expander("🕒 Prompt history"):
        for ts, prompt, author, note in get_prompt_history():
            st.markdown(f"🕒 **{ts.strftime('%Y-%m-%d %H:%M:%S')}** by `{author or 'unknown'}` – {note or ''}")
            st.info(prompt)

