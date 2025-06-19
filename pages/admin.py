import streamlit as st

st.set_page_config(page_title="Admin Login", layout="wide")

st.sidebar.page_link("app.py", label="💬 Chat Assistant")

# Authentication check
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Admin Login")
    st.sidebar.page_link("pages/admin.py", label="🔒 Admin")  
    password = st.text_input("Admin Password", type="password")
    if password == st.secrets["ADMIN_PASSWORD"]:
        st.session_state.authenticated = True
        st.rerun()
    elif password:
        st.error("Incorrect password.")
else:
    st.title("🔒 Admin Logout")
    st.sidebar.success("Authenticated as admin.")
    st.sidebar.page_link("pages/view_logs.py", label="📄 View Logs")
    #st.sidebar.page_link("pages/manage_users.py", label="👥 Manage Users")

    st.button("🔓 Logout", on_click=lambda: st.session_state.update({"authenticated": False}))
