import streamlit as st
from psycopg2.extras import DictCursor
from datetime import datetime, timedelta
import json
from pathlib import Path
from utils import get_connection, admin_authentication, render_sidebar

BASE_DIR = Path(__file__).parent.parent

LOG_SVG = (BASE_DIR / "assets" / "search_activity.svg").read_text()

# Helfer: Datums-/Zeitformatierung
def fmt_dt(value, fmt="%Y-%m-%d %H:%M:%S") -> str:
    if value is None:
        return "—"
    # Already a datetime
    if isinstance(value, datetime):
        try:
            return value.strftime(fmt)
        except Exception:
            return str(value)
    # Try to parse ISO strings
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return "—"
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
            return dt.strftime(fmt)
        except Exception:
            return value
    # Fallback
    try:
        return str(value)
    except Exception:
        return "—"

authenticated = admin_authentication()
render_sidebar(authenticated)

st.set_page_config(page_title="Logging & Analytics", layout="wide")

# --- Read logs from DB ---
def read_logs(limit=200, error_code=None, session_id=None):
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=DictCursor)

    # Build query based on filters
    if error_code and error_code != "All":
        if session_id and session_id != "All":
            query = """
                SELECT * FROM log_table 
                WHERE error_code = %s AND session_id = %s 
                ORDER BY timestamp DESC LIMIT %s
            """
            cursor.execute(query, (error_code, session_id, limit))
        else:
            query = "SELECT * FROM log_table WHERE error_code = %s ORDER BY timestamp DESC LIMIT %s"
            cursor.execute(query, (error_code, limit))
    else:
        if session_id and session_id != "All":
            query = "SELECT * FROM log_table WHERE session_id = %s ORDER BY timestamp DESC LIMIT %s"
            cursor.execute(query, (session_id, limit))
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

# --- Sessions overview (aggregated) ---
def get_session_overview(limit=500):
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=DictCursor)
    cursor.execute(
        """
        SELECT 
            session_id,
            COUNT(*) AS interactions,
            MIN(timestamp) AS first_seen,
            MAX(timestamp) AS last_seen,
            SUM(CASE WHEN error_code = 'E03' THEN 1 ELSE 0 END) AS errors
        FROM log_table
        WHERE session_id IS NOT NULL
        GROUP BY session_id
        ORDER BY last_seen DESC
        LIMIT %s
        """,
        (limit,)
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

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

        # New: Filter by session_id and group view
        session_rows = get_session_overview(limit=1000)
        session_options = ["All"] + session_rows
        def _format_session(o):
            if o == "All":
                return "All sessions"
            try:
                last = fmt_dt(o["last_seen"], "%Y-%m-%d %H:%M")
                return f"{o['session_id']} ({o['interactions']} msgs • last {last})"
            except Exception:
                return str(o)
        selected_session = st.selectbox("Filter by session", options=session_options, format_func=_format_session)
        selected_session_id = None if selected_session == "All" else selected_session["session_id"]

        group_by_session = st.toggle("Group by session", value=False, help="Show logs grouped by session with a per-session expander")

        logs = read_logs(limit=200, error_code=filter_code, session_id=selected_session_id)
        st.header("View Interaction Logs")
        st.markdown(f"### Showing {len(logs)} log entries")

        if group_by_session:
            # Group logs by session_id (None gets a label)
            groups = {}
            for entry in logs:
                sid = entry.get("session_id") or "(no session)"
                groups.setdefault(sid, []).append(entry)
            # Order groups by last activity desc
            ordered = sorted(groups.items(), key=lambda kv: max(e["timestamp"] for e in kv[1]), reverse=True)

            for sid, entries in ordered:
                entries_sorted = sorted(entries, key=lambda e: e["timestamp"])  # chronological within session
                n = len(entries_sorted)
                first_seen = entries_sorted[0]["timestamp"] if n else None
                last_seen = entries_sorted[-1]["timestamp"] if n else None
                error_count = sum(1 for e in entries_sorted if (e.get("error_code") == "E03"))
                header = f"Session {sid} — {n} interactions, {fmt_dt(first_seen, '%Y-%m-%d %H:%M')} → {fmt_dt(last_seen, '%Y-%m-%d %H:%M')}, errors: {error_count}"
                with st.expander(header, expanded=False):
                    for entry in entries_sorted:
                        st.markdown(f"**{fmt_dt(entry['timestamp'])}**, **Code:** `{entry.get('error_code') or 'OK'}`, **Citations:** {entry.get('citation_count', 0)}", unsafe_allow_html=False)
                        with st.container():
                            st.info(f"{entry['user_input']}", icon=":material/face:")
                            if entry.get('error_code') == "E03":
                                st.error(f"{entry['assistant_response']}", icon=":material/robot_2:")
                            elif entry.get('error_code') == "E02":
                                st.warning(f"{entry['assistant_response']}", icon=":material/robot_2:")
                            else:
                                st.success(f"{entry['assistant_response']}", icon=":material/robot_2:")
                            if entry.get('citations'):
                                citations = entry.get('citations')
                                if isinstance(citations, str):
                                    try:
                                        citations = json.loads(citations)
                                    except json.JSONDecodeError as e:
                                        st.error(f"Failed to parse citations: {e}")
                                        citations = {}
                                markdown_list = ""
                                for citation in citations.values():
                                    title = citation.get("title", "Untitled")
                                    url = citation.get("url", "#")
                                    file_id = citation.get("file_id", "Unknown")
                                    file_name = citation.get("file_name", "Unknown")
                                    markdown_list += f"* [{title}]({url}) (Vector Store File: {file_name}, ID: `{file_id}`)\n"
                                st.success(markdown_list)
                    st.markdown("---")
        else:
            # Flat list (existing behavior)
            for entry in logs:
                st.markdown(f"**{fmt_dt(entry['timestamp'])}**, **Code:** `{entry.get('error_code') or 'OK'}`, **Citations:** {entry.get('citation_count', 0)}", unsafe_allow_html=False)
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
                        if isinstance(citations, str):
                            try:
                                citations = json.loads(citations)
                            except json.JSONDecodeError as e:
                                st.error(f"Failed to parse citations: {e}")
                                citations = {}
                        markdown_list = ""
                        for citation in citations.values():
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
        
        # Session statistics (unchanged metrics)
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
        
        # Sessions overview table
        cursor.close()
        conn.close()
        st.markdown("### Sessions Overview")
        rows = get_session_overview(limit=500)
        if rows:
            table = [
                {
                    "session_id": r["session_id"],
                    "interactions": r["interactions"],
                    "first_seen": fmt_dt(r["first_seen"]),
                    "last_seen": fmt_dt(r["last_seen"]),
                    "errors": r["errors"],
                }
                for r in rows
            ]
            st.dataframe(table, use_container_width=True, hide_index=True)

            # Drill-down per session
            selected = st.selectbox(
                "Drill-down: select a session",
                options=["None"] + rows,
                format_func=lambda o: "None" if o == "None" else o["session_id"],
            )
            if selected != "None":
                sid = selected["session_id"]
                st.markdown(f"#### Session {sid} — timeline")
                detailed = read_logs(limit=1000, session_id=sid)
                for entry in sorted(detailed, key=lambda e: e["timestamp"]):
                    st.markdown(f"**{fmt_dt(entry['timestamp'])}**, **Code:** `{entry.get('error_code') or 'OK'}`")
                    with st.container():
                        st.info(f"{entry['user_input']}", icon=":material/face:")
                        st.success(f"{entry['assistant_response']}", icon=":material/robot_2:")
                
        
    with tab3:
        st.subheader("System Performance Metrics")
        # Add system metrics here
        st.info("System metrics dashboard - coming soon!")