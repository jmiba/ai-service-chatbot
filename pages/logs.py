import streamlit as st
from psycopg2.extras import DictCursor
from datetime import datetime, timedelta
import json
from pathlib import Path
from utils import get_connection, admin_authentication, render_sidebar
import base64
import pandas as pd
import math

BASE_DIR = Path(__file__).parent.parent
ICON_PATH = (BASE_DIR / "assets" / "search_activity.png")

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

authenticated = admin_authentication(return_to="/pages/logs")
render_sidebar(authenticated)

st.set_page_config(page_title="Logging & Analytics", layout="wide")

# --- Read logs from DB ---
def read_logs(limit=200, error_code=None, session_id=None, request_classification=None):
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=DictCursor)

    # Build dynamic WHERE clause
    clauses = []
    params = []

    if error_code and error_code != "All":
        clauses.append("error_code = %s")
        params.append(error_code)

    if session_id and session_id != "All":
        clauses.append("session_id = %s")
        params.append(session_id)

    if request_classification and request_classification != "All":
        if request_classification == "(unclassified)":
            clauses.append("request_classification IS NULL")
        else:
            clauses.append("request_classification = %s")
            params.append(request_classification)

    where_sql = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    query = f"SELECT * FROM log_table{where_sql} ORDER BY timestamp DESC LIMIT %s"
    params.append(limit)

    cursor.execute(query, tuple(params))

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


def render_log_details(entry: dict | None) -> None:
    """Display metadata and content for a single log entry."""
    if not entry:
        st.info("Select a log entry to inspect.")
        return

    timestamp = fmt_dt(entry.get("timestamp"))
    session_id = entry.get("session_id") or "(no session)"
    error_code = entry.get("error_code") or "OK"
    topic = entry.get("request_classification") or "(unclassified)"
    citations_count = entry.get("citation_count", 0)
    response_time = entry.get("response_time_ms")
    api_cost = entry.get("api_cost_usd")

    meta_lines = [
        f"**Timestamp:** {timestamp}",
        f"**Session:** `{session_id}`",
        f"**Error code:** `{error_code}`",
        f"**Topic:** `{topic}`",
        f"**Citations:** {citations_count}",
    ]
    if response_time is not None:
        meta_lines.append(f"**Response time:** {response_time} ms")
    if api_cost is not None:
        meta_lines.append(f"**API cost:** ${api_cost:.6f}")

    st.markdown("<br />".join(meta_lines), unsafe_allow_html=True)

    user_text = entry.get("user_input") or "(no user input captured)"
    st.info(user_text, icon=":material/face:")

    assistant_text = entry.get("assistant_response") or "(no response captured)"
    code = entry.get("error_code")
    if code == "E03":
        st.error(assistant_text, icon=":material/robot_2:")
    elif code == "E02":
        st.warning(assistant_text, icon=":material/robot_2:")
    else:
        st.success(assistant_text, icon=":material/robot_2:")

    citations = entry.get("citations")
    if citations:
        if isinstance(citations, str):
            try:
                citations = json.loads(citations)
            except json.JSONDecodeError:
                citations = {}
        if isinstance(citations, dict):
            lines = []
            for item in citations.values():
                title = item.get("title", "Untitled")
                url = item.get("url", "#")
                file_id = item.get("file_id", "Unknown")
                file_name = item.get("file_name", "Unknown")
                lines.append(f"* [{title}]({url}) • `{file_name}` (`{file_id}`)")
            if lines:
                st.markdown("\n".join(lines))


if authenticated:
    if ICON_PATH.exists():
        encoded_icon = base64.b64encode(ICON_PATH.read_bytes()).decode("utf-8")
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:.75rem;">
                <img src="data:image/png;base64,{encoded_icon}" width="48" height="48"/>
                <h1 style="margin:0;">Logging & Analytics</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.header("Logging & Analytics")
        
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Interaction Logs", "Session Analytics", "System Metrics"])
    
    with tab1:
        st.header("View Interaction Logs")
        # Admin-only content
        filter_code = st.selectbox("Filter by code", options=["All", "E00", "E01", "E02", "E03"])

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

        # New: Filter by topic (request_classification)
        try:
            conn_fc = get_connection()
            cur_fc = conn_fc.cursor(cursor_factory=DictCursor)
            cur_fc.execute("""
                SELECT DISTINCT COALESCE(request_classification, '(unclassified)') AS topic
                FROM log_table
                ORDER BY topic
            """)
            topic_rows = cur_fc.fetchall()
            cur_fc.close(); conn_fc.close()
            topic_options = ["All"] + [r["topic"] for r in topic_rows]
        except Exception:
            topic_options = ["All"]
        selected_topic = st.selectbox("Filter by topic", options=topic_options)
        selected_topic_val = None if selected_topic == "All" else selected_topic

        group_by_session = st.toggle(
            "Group by session",
            value=False,
            help="Show logs grouped by session with a per-session expander",
        )

        logs = read_logs(
            limit=200,
            error_code=filter_code,
            session_id=selected_session_id,
            request_classification=selected_topic_val,
        )
        log_entries = [dict(row) for row in logs]

        if not log_entries:
            st.markdown("---")
            st.info("No log entries match the current filters.")
        elif group_by_session:
            groups: dict[str, list[dict]] = {}
            for entry in log_entries:
                sid = entry.get("session_id") or "(no session)"
                groups.setdefault(sid, []).append(entry)

            ordered = sorted(
                groups.items(),
                key=lambda kv: max(e["timestamp"] for e in kv[1]),
                reverse=True,
            )

            session_payloads: list[dict] = []
            for sid, entries in ordered:
                entries_sorted = sorted(entries, key=lambda e: e["timestamp"])
                n = len(entries_sorted)
                first_seen = entries_sorted[0]["timestamp"] if n else None
                last_seen = entries_sorted[-1]["timestamp"] if n else None
                error_count = sum(1 for e in entries_sorted if (e.get("error_code") == "E03"))
                session_payloads.append(
                    {
                        "session_id": sid,
                        "interactions": n,
                        "first_seen": fmt_dt(first_seen, "%Y-%m-%d %H:%M"),
                        "last_seen": fmt_dt(last_seen, "%Y-%m-%d %H:%M"),
                        "errors": error_count,
                        "entries": entries_sorted,
                    }
                )

            if not session_payloads:
                st.info("No sessions found for the current filters.")
            else:
                total_sessions = len(session_payloads)
                #total_errors = sum(p["errors"] for p in session_payloads)

                st.session_state.setdefault("logs_session_page_size", 25)
                st.session_state.setdefault("logs_session_page", 1)
                st.session_state.setdefault("logs_session_selected_idx", 0)

                page_size_options = [10, 25, 50, 100]
                current_page_size = st.session_state["logs_session_page_size"]
                if current_page_size not in page_size_options:
                    current_page_size = 25
                    st.session_state["logs_session_page_size"] = current_page_size

                page_size = st.selectbox(
                    "Sessions per page",
                    options=page_size_options,
                    index=page_size_options.index(current_page_size),
                    key="logs_session_page_size_select",
                )
                if page_size != st.session_state["logs_session_page_size"]:
                    st.session_state["logs_session_page_size"] = page_size
                    st.session_state["logs_session_page"] = 1
                    
                st.markdown("---")
                st.markdown(f"### {total_sessions} sessions")

                total_pages = max(
                    1,
                    math.ceil(total_sessions / st.session_state["logs_session_page_size"]),
                )
                current_page = st.session_state.get("logs_session_page", 1)
                if current_page < 1 or current_page > total_pages:
                    current_page = 1
                    st.session_state["logs_session_page"] = 1

                start_idx = (current_page - 1) * st.session_state["logs_session_page_size"]
                end_idx = min(start_idx + st.session_state["logs_session_page_size"], total_sessions)
                page_payloads = session_payloads[start_idx:end_idx]

                paginator_col1, paginator_col2, paginator_col3 = st.columns([1, 2, 1])
                with paginator_col1:
                    if st.button(
                        "◀ Previous",
                        disabled=current_page <= 1,
                        key="logs_session_prev_page",
                    ):
                        st.session_state["logs_session_page"] = max(1, current_page - 1)
                        st.rerun()
                with paginator_col2:
                    st.markdown(
                        f"**Page {current_page} of {total_pages}** &nbsp;&nbsp;"
                        f"Showing sessions {start_idx + 1}–{end_idx} of {total_sessions}"
                    )
                with paginator_col3:
                    if st.button(
                        "Next ▶",
                        disabled=current_page >= total_pages,
                        key="logs_session_next_page",
                    ):
                        st.session_state["logs_session_page"] = min(
                            total_pages, current_page + 1
                        )
                        st.rerun()

                session_df = pd.DataFrame(
                    [
                        {
                            "Session": payload["session_id"],
                            "Interactions": payload["interactions"],
                            "First Seen": payload["first_seen"],
                            "Last Seen": payload["last_seen"],
                            "Errors": payload["errors"],
                        }
                        for payload in page_payloads
                    ]
                )

                selection_enabled = True
                try:
                    table_event = st.dataframe(
                        session_df,
                        hide_index=True,
                        width='stretch',
                        selection_mode="single-row",
                        on_select="rerun",
                        key="logs_session_table",
                    )
                except TypeError:
                    selection_enabled = False
                    table_event = st.dataframe(
                        session_df,
                        hide_index=True,
                        width='stretch',
                        key="logs_session_table",
                    )

                selected_session_idx = st.session_state.get(
                    "logs_session_selected_idx", 0
                )
                if selection_enabled:
                    selected_rows: list[int] = []
                    event_selection = getattr(table_event, "selection", None)
                    if isinstance(event_selection, dict):
                        selected_rows = event_selection.get("rows", []) or []

                    if selected_rows:
                        row_idx = selected_rows[0]
                        selected_session_idx = start_idx + row_idx
                        st.session_state["logs_session_selected_idx"] = selected_session_idx

                if not (0 <= selected_session_idx < total_sessions):
                    selected_session_idx = start_idx if page_payloads else 0
                    st.session_state["logs_session_selected_idx"] = selected_session_idx

                if not selection_enabled and page_payloads:
                    fallback_labels = {
                        f"{payload['session_id']} · {payload['interactions']} msgs · last {payload['last_seen']}": start_idx
                        + idx
                        for idx, payload in enumerate(page_payloads)
                    }
                    fallback_default = next(
                        (
                            label
                            for label, idx in fallback_labels.items()
                            if idx == selected_session_idx
                        ),
                        None,
                    )
                    fallback_options = list(fallback_labels.keys())
                    fallback_index = (
                        fallback_options.index(fallback_default)
                        if fallback_default in fallback_options
                        else 0
                    )
                    selected_label = st.radio(
                        "Select session",
                        options=fallback_options,
                        index=fallback_index,
                        key="logs_session_fallback",
                    )
                    selected_session_idx = fallback_labels[selected_label]
                    st.session_state["logs_session_selected_idx"] = selected_session_idx

                if page_payloads:
                    selected_payload = session_payloads[selected_session_idx]
                    st.markdown(
                        f"#### Session {selected_payload['session_id']} "
                        f"— {selected_payload['interactions']} interactions"
                    )
                    st.caption(
                        f"{selected_payload['first_seen']} → {selected_payload['last_seen']} • "
                        f"Errors: {selected_payload['errors']}"
                    )

                    for idx, entry in enumerate(selected_payload["entries"]):
                        st.markdown(
                            f"**{fmt_dt(entry['timestamp'])}**, **Code:** `{entry.get('error_code') or 'OK'}`"
                            f", **Citations:** {entry.get('citation_count', 0)}, **Topic:** "
                            f"`{entry.get('request_classification') or '(unclassified)'}`",
                            unsafe_allow_html=False,
                        )
                        render_log_details(entry)
                        if idx < len(selected_payload["entries"]) - 1:
                            st.markdown("---")
        else:
            st.session_state.setdefault("logs_page_size", 25)
            st.session_state.setdefault("logs_page", 1)
            st.session_state.setdefault("logs_selected_index", 0)

            page_size_options = [10, 25, 50, 100]
            current_page_size = st.session_state["logs_page_size"]
            if current_page_size not in page_size_options:
                current_page_size = 25
                st.session_state["logs_page_size"] = current_page_size

            page_size = st.selectbox(
                "Rows per page",
                options=page_size_options,
                index=page_size_options.index(current_page_size),
                key="logs_page_size_select",
            )
            
            st.markdown("---")
            st.markdown(f"### {len(log_entries)} interactions")
            if page_size != st.session_state["logs_page_size"]:
                st.session_state["logs_page_size"] = page_size
                st.session_state["logs_page"] = 1

            total_entries = len(log_entries)
            total_pages = max(1, -(-total_entries // st.session_state["logs_page_size"]))
            current_page = st.session_state.get("logs_page", 1)
            if current_page < 1 or current_page > total_pages:
                current_page = 1
                st.session_state["logs_page"] = 1

            start_idx = (current_page - 1) * st.session_state["logs_page_size"]
            end_idx = min(start_idx + st.session_state["logs_page_size"], total_entries)
            page_entries = log_entries[start_idx:end_idx]

            paginator_col1, paginator_col2, paginator_col3 = st.columns([1, 2, 1])
            with paginator_col1:
                if st.button("◀ Previous", disabled=current_page <= 1, key="logs_prev_page"):
                    st.session_state["logs_page"] = max(1, current_page - 1)
                    st.rerun()
            with paginator_col2:
                st.markdown(
                    f"**Page {current_page} of {total_pages}** &nbsp;&nbsp;"
                    f"Showing entries {start_idx + 1}–{end_idx} of {total_entries}"
                )
            with paginator_col3:
                if st.button("Next ▶", disabled=current_page >= total_pages, key="logs_next_page"):
                    st.session_state["logs_page"] = min(total_pages, current_page + 1)
                    st.rerun()

            table_rows = []
            for entry in page_entries:
                table_rows.append(
                    {
                        "Timestamp": fmt_dt(entry.get("timestamp")),
                        "Session": entry.get("session_id") or "(no session)",
                        "Error": entry.get("error_code") or "OK",
                        "Topic": entry.get("request_classification") or "(unclassified)",
                        "Citations": entry.get("citation_count", 0),
                    }
                )

            table_df = pd.DataFrame(table_rows)
            selection_enabled = True
            try:
                table_event = st.dataframe(
                    table_df,
                    hide_index=True,
                    width='stretch',
                    selection_mode="single-row",
                    on_select="rerun",
                    key="logs_table",
                )
            except TypeError:
                selection_enabled = False
                table_event = st.dataframe(
                    table_df,
                    hide_index=True,
                    width='stretch',
                    key="logs_table",
                )

            selected_global_index = st.session_state.get("logs_selected_index", 0)
            if selection_enabled:
                selected_rows: list[int] = []
                event_selection = getattr(table_event, "selection", None)
                if isinstance(event_selection, dict):
                    selected_rows = event_selection.get("rows", []) or []

                if selected_rows:
                    row_idx = selected_rows[0]
                    selected_global_index = start_idx + row_idx
                    st.session_state["logs_selected_index"] = selected_global_index

            if not (0 <= selected_global_index < total_entries):
                selected_global_index = start_idx if page_entries else 0
                st.session_state["logs_selected_index"] = selected_global_index

            if not selection_enabled and page_entries:
                fallback_labels = {
                    f"{fmt_dt(entry.get('timestamp'))} · {entry.get('error_code') or 'OK'} · "
                    f"{entry.get('request_classification') or '(unclassified)'}": start_idx + idx
                    for idx, entry in enumerate(page_entries)
                }
                fallback_default = next(
                    (
                        label
                        for label, idx in fallback_labels.items()
                        if idx == selected_global_index
                    ),
                    None,
                )
                fallback_options = list(fallback_labels.keys())
                fallback_index = (
                    fallback_options.index(fallback_default)
                    if fallback_default in fallback_options
                    else 0
                )
                selected_label = st.radio(
                    "Select entry to inspect",
                    options=fallback_options,
                    index=fallback_index,
                    key="logs_table_fallback",
                )
                selected_global_index = fallback_labels[selected_label]
                st.session_state["logs_selected_index"] = selected_global_index

            selected_entry = log_entries[selected_global_index] if page_entries else None
            detail_container = st.container()
            with detail_container:
                render_log_details(selected_entry)

        if st.button("Delete All Logs", icon=":material/delete_forever:"):
            delete_logs()
            st.success("All logs have been deleted.")

        if st.button("Logout", icon=":material/logout:"):
            st.session_state["authenticated"] = False
            st.rerun()
        
    with tab2:
        st.subheader("Session-Based Analytics")

        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT 
                COUNT(DISTINCT session_id) AS total_sessions,
                COUNT(*) AS total_interactions,
                COUNT(*)::numeric / NULLIF(COUNT(DISTINCT session_id), 0) AS avg_interactions_per_session
            FROM log_table
            WHERE session_id IS NOT NULL
            """
        )

        stats = cursor.fetchone()
        if stats:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Sessions", stats[0] or 0)
            with col2:
                st.metric("Total Interactions", stats[1] or 0)
            with col3:
                st.metric(
                    "Avg Interactions/Session",
                    f"{stats[2]:.1f}" if stats[2] else "0.0",
                )

        try:
            from psycopg2.extras import DictCursor as _DictCursor

            cursor.close()
            cursor = conn.cursor(cursor_factory=_DictCursor)
            cursor.execute(
                """
                SELECT COALESCE(request_classification, '(unclassified)') AS topic,
                       COUNT(*)::int AS count
                FROM log_table
                GROUP BY topic
                ORDER BY count DESC
                """
            )
            topic_rows = cursor.fetchall()
        except Exception as exc:
            topic_rows = []
            st.warning(f"Could not load topics: {exc}")
        finally:
            cursor.close()
            conn.close()

        st.markdown("### Topics")
        if topic_rows:
            try:
                import altair as alt

                records: list[dict] = []
                for row in topic_rows:
                    try:
                        topic_val = row["topic"]
                    except Exception:
                        topic_val = row[0]
                    if not isinstance(topic_val, str):
                        topic_val = str(topic_val) if topic_val is not None else "(unclassified)"

                    try:
                        count_val = int(row["count"])
                    except Exception:
                        count_val = int(row[1]) if len(row) > 1 else 0
                    records.append({"topic": topic_val, "count": count_val})

                df_topics = pd.DataFrame.from_records(records)
                df_topics = df_topics[df_topics["count"] > 0]
                if df_topics.empty:
                    st.info("No topic data available yet.")
                else:
                    pie = (
                        alt.Chart(df_topics)
                        .mark_arc()
                        .encode(
                            theta=alt.Theta(field="count", type="quantitative"),
                            color=alt.Color(field="topic", type="nominal", legend=alt.Legend(title="Topic")),
                            tooltip=[
                                alt.Tooltip("topic:N", title="Topic"),
                                alt.Tooltip("count:Q", title="Count"),
                            ],
                        )
                        .properties(height=320)
                    )
                    st.altair_chart(pie)
            except Exception as exc:
                st.warning("Falling back to table view due to a charting error.")
                try:
                    st.exception(exc)
                except Exception:
                    pass
                st.table(df_topics if "df_topics" in locals() else topic_rows)
        else:
            st.info("No topic data available yet.")

        st.markdown("### Topics — Interactions and Sessions")
        try:
            from psycopg2.extras import DictCursor as _DictCursor

            conn2 = get_connection()
            cur2 = conn2.cursor(cursor_factory=_DictCursor)
            cur2.execute(
                """
                SELECT COALESCE(request_classification, '(unclassified)') AS topic,
                       COUNT(*)::int AS interactions,
                       COUNT(DISTINCT session_id)::int AS sessions_with_topic
                FROM log_table
                GROUP BY topic
                ORDER BY interactions DESC
                """
            )
            agg_rows = cur2.fetchall()
            cur2.execute(
                """
                SELECT COUNT(*)::int AS total_interactions,
                       COUNT(DISTINCT session_id)::int AS total_sessions
                FROM log_table
                WHERE session_id IS NOT NULL
                """
            )
            totals = cur2.fetchone() or {"total_interactions": 0, "total_sessions": 0}
            cur2.close()
            conn2.close()

            df = pd.DataFrame(agg_rows)
            if not {"topic", "interactions", "sessions_with_topic"}.issubset(df.columns):
                if len(df.columns) >= 3:
                    df.columns = [
                        "topic",
                        "interactions",
                        "sessions_with_topic",
                        *list(df.columns[3:]),
                    ]
                else:
                    st.info("No topic data available yet.")
                    df = pd.DataFrame(columns=["topic", "interactions", "sessions_with_topic"])

            if df.empty:
                st.info("No topic data available yet.")
            else:
                total_interactions = (
                    totals.get("total_interactions", 0)
                    if isinstance(totals, dict)
                    else totals[0]
                )
                total_sessions = (
                    totals.get("total_sessions", 0)
                    if isinstance(totals, dict)
                    else totals[1]
                )

                if total_interactions == 0 or total_sessions == 0:
                    st.info("No topic data available yet.")
                else:
                    df["share_of_interactions_%"] = (
                        df["interactions"] / total_interactions
                    ) * 100
                    df["share_of_sessions_%"] = (
                        df["sessions_with_topic"] / total_sessions
                    ) * 100
                    df["interactions_per_session"] = df["interactions"] / df[
                        "sessions_with_topic"
                    ].replace(0, pd.NA)

                    display_df = df[
                        [
                            "topic",
                            "interactions",
                            "sessions_with_topic",
                            "share_of_interactions_%",
                            "share_of_sessions_%",
                            "interactions_per_session",
                        ]
                    ].copy()
                    display_df.rename(
                        columns={
                            "topic": "Topic",
                            "interactions": "Interactions",
                            "sessions_with_topic": "Sessions with topic",
                            "share_of_interactions_%": "Share of interactions (%)",
                            "share_of_sessions_%": "Share of sessions (%)",
                            "interactions_per_session": "Interactions/session",
                        },
                        inplace=True,
                    )
                    display_df["Share of interactions (%)"] = display_df[
                        "Share of interactions (%)"
                    ].round(1)
                    display_df["Share of sessions (%)"] = display_df[
                        "Share of sessions (%)"
                    ].round(1)
                    display_df["Interactions/session"] = display_df[
                        "Interactions/session"
                    ].round(2)
                    st.dataframe(display_df, width="stretch", hide_index=True)
        except Exception as exc:
            st.warning(f"Could not build topics table: {exc}")

    with tab3:
        st.subheader("System Performance Metrics")
        st.caption("OpenAI usage, tokens, estimated costs, and latency (from log_table)")

        # Date range filter
        today = datetime.now().date()
        default_start = today - timedelta(days=30)
        col_a, col_b = st.columns(2)
        with col_a:
            start_date = st.date_input("Start date", value=default_start)
        with col_b:
            end_date = st.date_input("End date", value=today)

        conn = get_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)

        # Pull usage/cost rows
        cursor.execute(
            """
            SELECT timestamp::date AS day,
                   model,
                   COALESCE(usage_input_tokens, 0) AS input_tokens,
                   COALESCE(usage_output_tokens, 0) AS output_tokens,
                   COALESCE(usage_total_tokens, 0) AS total_tokens,
                   COALESCE(usage_reasoning_tokens, 0) AS reasoning_tokens,
                   COALESCE(api_cost_usd, 0.0) AS cost_usd,
                   response_time_ms
            FROM log_table
            WHERE timestamp::date BETWEEN %s AND %s
        """,
            (start_date, end_date)
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        # Summaries
        total_in = sum(r["input_tokens"] for r in rows)
        total_out = sum(r["output_tokens"] for r in rows)
        total_reason = sum(r["reasoning_tokens"] for r in rows)
        total_cost = sum(float(r["cost_usd"]) for r in rows)
        latencies = [int(r["response_time_ms"]) for r in rows if r.get("response_time_ms") is not None]
        avg_latency = (sum(latencies) / len(latencies) / 1000) if latencies else 0.0

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Input tokens", f"{total_in:,}")
        c2.metric("Output tokens", f"{total_out:,}")
        c3.metric("Reasoning tokens", f"{total_reason:,}")
        c4.metric("Estimated cost (USD)", f"${total_cost:,.2f}")
        c5.metric("Avg. latency (s)", f"{avg_latency:,.1f}")

        # Prepare daily aggregates
        from collections import defaultdict
        daily_tokens = defaultdict(lambda: {"input":0, "output":0, "total":0, "reason":0, "cost":0.0, "lat":[]})
        for r in rows:
            d = r["day"]
            daily = daily_tokens[d]
            daily["input"] += r["input_tokens"]
            daily["output"] += r["output_tokens"]
            daily["total"] += r["total_tokens"]
            daily["reason"] += r["reasoning_tokens"]
            daily["cost"] += float(r["cost_usd"])
            if r.get("response_time_ms") is not None:
                daily["lat"].append(int(r["response_time_ms"]))

        if daily_tokens:
            # Build chart-friendly data
            days_sorted = sorted(daily_tokens.keys())
            chart_data = {
                "day": days_sorted,
                "input_tokens": [daily_tokens[d]["input"] for d in days_sorted],
                "output_tokens": [daily_tokens[d]["output"] for d in days_sorted],
                "total_tokens": [daily_tokens[d]["total"] for d in days_sorted],
                "cost_usd": [daily_tokens[d]["cost"] for d in days_sorted],
                "avg_latency_ms": [ (sum(daily_tokens[d]["lat"])/len(daily_tokens[d]["lat"])) if daily_tokens[d]["lat"] else 0 for d in days_sorted ],
            }
            st.line_chart(chart_data, x="day", y=["input_tokens","output_tokens","total_tokens"]) 
            st.area_chart({"day": chart_data["day"], "cost_usd": chart_data["cost_usd"]}, x="day", y="cost_usd")
            st.line_chart({"day": chart_data["day"], "avg_latency_ms": chart_data["avg_latency_ms"]}, x="day", y="avg_latency_ms")

        # Breakdown by model (add latency)
        model_agg = defaultdict(lambda: {"input":0, "output":0, "total":0, "reason":0, "cost":0.0, "latencies":[]})
        for r in rows:
            m = r["model"] or "(unknown)"
            model_agg[m]["input"] += r["input_tokens"]
            model_agg[m]["output"] += r["output_tokens"]
            model_agg[m]["total"] += r["total_tokens"]
            model_agg[m]["reason"] += r["reasoning_tokens"]
            model_agg[m]["cost"] += float(r["cost_usd"])
            if r.get("response_time_ms") is not None:
                model_agg[m]["latencies"].append(int(r["response_time_ms"]))

        if model_agg:
            st.markdown("### By model")
            table = [
                {
                    "Model": m,
                    "Input tokens": v["input"],
                    "Output tokens": v["output"],
                    "Total tokens": v["total"],
                    "Reasoning tokens": v["reason"],
                    "Costs (USD)": round(v["cost"], 3),
                    "Avg. latency (s)": round((sum(v["latencies"])/len(v["latencies"])/1000) if v["latencies"] else 0.0, 1)
                }
                for m, v in sorted(model_agg.items(), key=lambda kv: kv[1]["cost"], reverse=True)
            ]
            st.dataframe(table, width="stretch", hide_index=True)

        st.info("Costs are estimated from stored usage and pricing in config/pricing.json.")
