import streamlit as st
from utils import (
    create_knowledge_base_table,
    get_document_by_identifier,
    load_css,
)

st.set_page_config(page_title="Document Viewer", layout="wide", initial_sidebar_state="collapsed")

# Ensure schema exists before querying
try:
    create_knowledge_base_table()
except Exception:
    pass

query_params = st.query_params
file_id = query_params.get("file_id")
doc_id_param = query_params.get("doc_id")
doc_id = None
if doc_id_param:
    try:
        doc_id = int(doc_id_param)
    except (TypeError, ValueError):
        doc_id = None
       
load_css()


st.markdown(
    """
    <style>
        [data-testid="stSidebar"] { display: none !important; }
        [data-testid="stSidebarNav"] { display: none !important; }
        [data-testid="collapsedControl"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

document = None
if file_id or doc_id is not None:
    try:
        document = get_document_by_identifier(doc_id=doc_id, file_id=file_id)
    except Exception as exc:
        st.error(f"Failed to load document: {exc}")
else:
    st.info("Open this viewer via a reference link in the chat to inspect a document.")

if document is None:
    st.warning("Document not found or no identifier supplied.")
    st.stop()

st.title(document["title"])
if document.get("recordset"):
    st.caption(f"Recordset: {document['recordset']}")

meta_cols = st.columns(2)
with meta_cols[0]:
    if document.get("summary"):
        st.info(f"**Summary:** {document['summary']}", icon=":material/summarize:")
with meta_cols[1]:
    tags = document.get("tags") or []
    if tags:
        st.info("**Tags:**\n\n" + ", ".join(tags), icon=":material/sell:")

st.divider()
content = document.get("markdown")
if content:
    st.markdown(content)
else:
    st.info("This document does not have markdown content stored.")
