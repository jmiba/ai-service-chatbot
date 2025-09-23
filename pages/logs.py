import streamlit as st
from psycopg2.extras import DictCursor
from datetime import datetime, timedelta
import json
from pathlib import Path
from utils import get_connection, admin_authentication, render_sidebar
import base64

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

        group_by_session = st.toggle("Group by session", value=False, help="Show logs grouped by session with a per-session expander")

        logs = read_logs(limit=200, error_code=filter_code, session_id=selected_session_id, request_classification=selected_topic_val)
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
                        st.markdown(f"**{fmt_dt(entry['timestamp'])}**, **Code:** `{entry.get('error_code') or 'OK'}`, **Citations:** {entry.get('citation_count', 0)}, **Topic:** `{entry.get('request_classification') or '(unclassified)'}`", unsafe_allow_html=False)
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
                with st.expander(f"**{fmt_dt(entry['timestamp'])}**, **Code:** `{entry.get('error_code') or 'OK'}`, **Citations:** {entry.get('citation_count', 0)}, **Topic:** `{entry.get('request_classification') or '(unclassified)'}`", expanded=False):
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
        
        # --- Topics distribution (request_classification) ---
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
        except Exception as e:
            topic_rows = []
            st.warning(f"Could not load topics: {e}")
        finally:
            cursor.close()
            conn.close()

        st.markdown("### Topics")
        if topic_rows:
            try:
                import pandas as pd
                import altair as alt
                # Normalize rows to a clean list of dicts and coerce types
                records = []
                for row in topic_rows:
                    # row can be a DictRow; prefer key access, fall back to positional
                    try:
                        topic_val = row["topic"]
                    except Exception:
                        try:
                            topic_val = row[0]
                        except Exception:
                            topic_val = "(unclassified)"
                    if not isinstance(topic_val, str):
                        topic_val = str(topic_val) if topic_val is not None else "(unclassified)"

                    try:
                        count_val = int(row["count"])  # ensure int
                    except Exception:
                        try:
                            count_val = int(row[1])
                        except Exception:
                            count_val = 0
                    records.append({"topic": topic_val, "count": count_val})

                df_topics = pd.DataFrame.from_records(records)
                # Drop zero or negative counts just in case
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
                                alt.Tooltip("count:Q", title="Count")
                            ],
                        )
                        .properties(height=320)
                    )
                    st.altair_chart(pie, use_container_width=True)
            except Exception as e:
                st.warning("Falling back to table view due to a charting error.")
                try:
                    st.exception(e)
                except Exception:
                    pass
                # Fallback: show as simple table if chart fails
                try:
                    st.table(df_topics if 'df_topics' in locals() else topic_rows)
                except Exception:
                    st.write(topic_rows)
        else:
            st.info("No topic data available yet.")

        # --- Topics table: interactions vs sessions ---
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
            # Totals (for shares)
            cur2.execute(
                """
                SELECT COUNT(*)::int AS total_interactions,
                       COUNT(DISTINCT session_id)::int AS total_sessions
                FROM log_table
                WHERE session_id IS NOT NULL
                """
            )
            totals = cur2.fetchone() or {"total_interactions": 0, "total_sessions": 0}
            cur2.close(); conn2.close()

            import pandas as pd
            df = pd.DataFrame(agg_rows)
            if not {'topic','interactions','sessions_with_topic'}.issubset(df.columns):
                # handle tuple rows
                if len(df.columns) >= 3:
                    df.columns = ['topic','interactions','sessions_with_topic'] + list(df.columns[3:])
                else:
                    st.info("No topic data available yet.")
                
            if not df.empty:
                ti = int(totals.get('total_interactions', 0) or 0)
                ts = int(totals.get('total_sessions', 0) or 0)
                # Compute shares
                df['share_of_interactions_%'] = df['interactions'].apply(lambda x: (x/ti*100.0) if ti else 0.0)
                df['share_of_sessions_%'] = df['sessions_with_topic'].apply(lambda x: (x/ts*100.0) if ts else 0.0)
                df['interactions_per_session'] = df.apply(lambda r: (r['interactions']/r['sessions_with_topic']) if r['sessions_with_topic'] else 0.0, axis=1)
                # Format for display
                display_df = df[['topic','interactions','sessions_with_topic','share_of_interactions_%','share_of_sessions_%','interactions_per_session']].copy()
                display_df.rename(columns={
                    'topic':'Topic',
                    'interactions':'Interactions',
                    'sessions_with_topic':'Sessions with topic',
                    'share_of_interactions_%':'Share of interactions (%)',
                    'share_of_sessions_%':'Share of sessions (%)',
                    'interactions_per_session':'Interactions/session'
                }, inplace=True)
                # Round numeric columns
                display_df['Share of interactions (%)'] = display_df['Share of interactions (%)'].round(1)
                display_df['Share of sessions (%)'] = display_df['Share of sessions (%)'].round(1)
                display_df['Interactions/session'] = display_df['Interactions/session'].round(2)
                st.dataframe(display_df, width="stretch", hide_index=True)
            else:
                st.info("No topic data available yet.")
        except Exception as e:
            st.warning(f"Could not build topics table: {e}")
        
        # Sessions overview table
        st.markdown("### Sessions Overview")
        rows = get_session_overview(limit=500)
        if rows:
            table = [
                {
                    "Session ID": r["session_id"],
                    "Interactions": r["interactions"],
                    "Start": fmt_dt(r["first_seen"]),
                    "Stop": fmt_dt(r["last_seen"]),
                    "Errors": r["errors"],
                }
                for r in rows
            ]
            st.dataframe(table, width="stretch", hide_index=True)

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
                    st.markdown(f"**{fmt_dt(entry['timestamp'])}**, **Code:** `{entry.get('error_code') or 'OK'}`, **Topic:** `{entry.get('request_classification') or '(unclassified)'}`")
                    with st.container():
                        st.info(f"{entry['user_input']}", icon=":material/face:")
                        st.success(f"{entry['assistant_response']}", icon=":material/robot_2:")
                
        
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
            st.line_chart(chart_data, x="day", y=["input_tokens","output_tokens","total_tokens"], use_container_width=True)
            st.area_chart({"day": chart_data["day"], "cost_usd": chart_data["cost_usd"]}, x="day", y="cost_usd", use_container_width=True)
            st.line_chart({"day": chart_data["day"], "avg_latency_ms": chart_data["avg_latency_ms"]}, x="day", y="avg_latency_ms", use_container_width=True)

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
