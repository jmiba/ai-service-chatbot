import base64
import json
import math
import re
import time as _time
from datetime import datetime, timezone, timedelta, time
from collections import defaultdict
from zoneinfo import ZoneInfo
import uuid

import pandas as pd
import streamlit as st
import html

from scrape.core import BASE_DIR, summarize_and_tag_tooluse, save_document_to_db
from scrape.config import load_summarize_prompts
from utils import (
    get_connection,
    get_kb_entries,
    get_kb_entry_by_id,
    create_knowledge_base_table,
    admin_authentication,
    render_sidebar,
    compute_sha256,
    create_url_configs_table,
    save_url_configs,
    load_url_configs,
    initialize_default_url_configs,
    launch_cli_job,
    CLIJob,
    CLIJobError,
    show_blocking_overlay,
    hide_blocking_overlay,
    get_document_metrics,
    is_job_locked,
    release_job_lock,
    render_log_output,
    mark_vector_store_dirty,
    read_scraper_schedule,
    write_scraper_schedule,
    normalize_tags_for_storage,
    normalize_existing_document_tags,
)

ICON_PATH = BASE_DIR / "assets" / "home_storage.png"

# -----------------------------
# Auth / sidebar
# -----------------------------
authenticated = admin_authentication()
render_sidebar(authenticated)

def ensure_url_config_ids(configs: list[dict]) -> None:
    """Assign stable IDs so Streamlit widget keys stay aligned after reordering."""
    for config in configs:
        config.setdefault("_id", uuid.uuid4().hex)


def rerun_app():
    """Trigger a Streamlit rerun compatible with recent and legacy versions."""
    try:
        st.rerun()
    except AttributeError:
        try:
            st.experimental_rerun()
        except AttributeError:
            pass
        
def render_kb_entry_details(entry: tuple):
    """Render a single knowledge base entry and associated actions."""

    (
        entry_id,
        url,
        title,
        safe_title,
        crawl_date,
        lang,
        summary,
        tags,
        markdown,
        recordset,
        vector_file_id,
        page_type,
        no_upload,
        is_stale,
    ) = entry

    tags = tags or []
    tags_str = " ".join(f"#{tag}" for tag in tags) if tags else "—"
    vector_status = (
        "✅ Vectorized"
        if vector_file_id and not no_upload
        else "🚫 Excluded" if no_upload else "⏳ Waiting for sync"
    )

    st.subheader(title or f"(no title) · ID {entry_id}")

    meta_lines = []
    if recordset:
        meta_lines.append(f"**Recordset:** {html.escape(str(recordset)) if recordset else '—'}")
    meta_lines.append(f"**Vector status:** {vector_status}")
    if vector_file_id:
        meta_lines.append(f"**Vector file:** `{html.escape(str(vector_file_id))}`")
    if crawl_date:
        meta_lines.append(f"**Last crawl:** {crawl_date}")
    meta_lines.append(f"**Language:** {html.escape(str(lang)) if lang else 'unknown'}")
    meta_lines.append(f"**Page type:** {html.escape(str(page_type)) if page_type else '—'}")
    meta_lines.append(f"**Tags:** {html.escape(tags_str)}")
    st.markdown("<br />".join(meta_lines), unsafe_allow_html=True)

    if url:
        st.caption(f"🔗 {url}")

    if is_stale:
        st.warning("This page was not encountered in the latest crawl.", icon=":material/report:")

    summary_placeholder = st.empty()
    if summary:
        summary_placeholder.markdown(f"**Summary:** {summary}")
    else:
        summary_placeholder.markdown("**Summary:** (none)")

    with st.expander("Markdown Preview", expanded=False):
        preview_placeholder = st.empty()
        if markdown:
            preview_placeholder.code(markdown, language="markdown")
        else:
            preview_placeholder.info("(no content)")

    cols = st.columns([1, 1, 1])
    is_internal = recordset == "Internal documents"
    is_editing_internal = is_internal and st.session_state.get("internal_edit_id") == entry_id

    with cols[0]:
        if is_internal:
            edit_label = "Close" if is_editing_internal else "Edit"
            edit_icon = ":material/close:" if is_editing_internal else ":material/edit:"
            if st.button(edit_label, key=f"edit_toggle_{entry_id}", icon=edit_icon, type="secondary"):
                st.session_state["internal_edit_id"] = None if is_editing_internal else entry_id
                rerun_app()

    with cols[1]:
        toggle_label = "Include in vector store" if no_upload else "Exclude from vector store"
        toggle_icon = ":material/check_circle:" if no_upload else ":material/block:"
        if st.button(toggle_label, key=f"toggle_upload_{entry_id}", icon=toggle_icon, type="secondary"):
            try:
                conn = get_connection()
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE documents SET no_upload = %s WHERE id = %s",
                        (not no_upload, entry_id),
                    )
                    conn.commit()
                conn.close()
                if no_upload:
                    st.success(f"Record {entry_id} included in vector store.")
                else:
                    st.success(f"Record {entry_id} excluded from vector store.")
                mark_vector_store_dirty()
                rerun_app()
            except Exception as exc:
                st.error(f"Failed to update vector store inclusion for record {entry_id}: {exc}")

    with cols[2]:
        if st.button(
            "Delete from knowledge base",
            key=f"delete_button_{entry_id}",
            icon=":material/delete:",
            type="secondary",
        ):
            try:
                conn = get_connection()
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM documents WHERE id = %s", (entry_id,))
                    conn.commit()
                conn.close()
                st.success(f"Record {entry_id} deleted successfully.")
                mark_vector_store_dirty()
                rerun_app()
            except Exception as exc:
                st.error(f"Failed to delete record {entry_id}: {exc}")

    if is_internal and is_editing_internal:
        st.divider()
        st.markdown("**Edit Internal Document**")
        options = ["text", "links", "other"]
        default_page_type = page_type if page_type in options else "text"
        identifier_value = (
            url.split("internal://internal-documents/")[-1]
            if url and url.startswith("internal://internal-documents/")
            else ""
        )

        with st.form(f"edit_internal_{entry_id}"):
            new_title = st.text_input("Title", value=title or "", key=f"edit_title_{entry_id}")
            new_safe_title = st.text_input(
                "Safe Title", value=safe_title or "", key=f"edit_safe_title_{entry_id}"
            )
            new_identifier = st.text_input(
                "Document identifier",
                value=identifier_value,
                help=(
                    "Stored as internal://internal-documents/<identifier>. "
                    "Leave blank to keep the current value."
                ),
                key=f"edit_identifier_{entry_id}",
            )
            new_markdown = st.text_area(
                "Markdown", value=markdown or "", height=220, key=f"edit_markdown_{entry_id}"
            )
            new_summary = st.text_area(
                "Summary", value=summary or "", height=120, key=f"edit_summary_{entry_id}"
            )
            new_tags_input = st.text_input(
                "Tags (comma separated)",
                value=", ".join(tags or []),
                key=f"edit_tags_{entry_id}",
            )
            new_lang = st.text_input(
                "Language", value=lang or "unknown", key=f"edit_lang_{entry_id}"
            )
            new_page_type = st.selectbox(
                "Page type",
                options=options,
                index=options.index(default_page_type),
                key=f"edit_page_type_{entry_id}",
            )
            new_no_upload = st.checkbox(
                "Exclude from vector store",
                value=bool(no_upload),
                key=f"edit_no_upload_{entry_id}",
            )
            submit_edit = st.form_submit_button("Save changes", type="primary")
            cancel_edit = st.form_submit_button("Cancel", type="secondary")

        if cancel_edit:
            st.session_state["internal_edit_id"] = None
            rerun_app()

        if submit_edit:
            if not new_title.strip():
                st.error("Title cannot be empty.")
            elif not new_markdown.strip():
                st.error("Markdown content cannot be empty.")
            else:
                identifier_final = new_identifier.strip() or identifier_value or re.sub(
                    r"_+",
                    "_",
                    "".join(
                        c if c.isalnum() else "_"
                        for c in (new_title.strip() or "untitled")
                    ),
                )[:64]
                new_url = f"internal://internal-documents/{identifier_final}"
                safe_title_final = new_safe_title.strip() or re.sub(
                    r"_+",
                    "_",
                    "".join(
                        c if c.isalnum() else "_"
                        for c in (new_title.strip() or "untitled")
                    ),
                )[:64]
                tags_list_new = normalize_tags_for_storage(new_tags_input)
                resync_needed = bool(
                    vector_file_id
                    and not new_no_upload
                    and new_markdown.strip() != (markdown or "").strip()
                )
                try:
                    conn = get_connection()
                    with conn:
                        with conn.cursor() as cur:
                            cur.execute(
                                """
                                UPDATE documents
                                SET url = %s,
                                    title = %s,
                                    safe_title = %s,
                                    summary = %s,
                                    tags = %s,
                                    markdown_content = %s,
                                    markdown_hash = %s,
                                    lang = %s,
                                    page_type = %s,
                                    no_upload = %s,
                                    vector_file_id = CASE WHEN %s THEN NULL ELSE vector_file_id END,
                                    old_file_id = CASE WHEN %s THEN vector_file_id ELSE old_file_id END,
                                    updated_at = NOW()
                                WHERE id = %s
                                """,
                                (
                                    new_url,
                                    new_title.strip(),
                                    safe_title_final,
                                    new_summary.strip(),
                                    tags_list_new,
                                    new_markdown,
                                    compute_sha256(new_markdown),
                                    new_lang.strip() or "unknown",
                                    new_page_type,
                                    new_no_upload,
                                    resync_needed,
                                    resync_needed,
                                    entry_id,
                                ),
                            )
                    st.session_state["internal_edit_success"] = {
                        "title": new_title.strip(),
                        "url": new_url,
                    }
                    st.session_state["internal_edit_id"] = None
                    mark_vector_store_dirty()
                    rerun_app()
                except Exception as exc:
                    st.error(f"Failed to update document: {exc}")

# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    # Ensure table exists before any DB operations
    create_knowledge_base_table()

    st.set_page_config(page_title="Content Indexing", layout="wide")

    if ICON_PATH.exists():
        encoded_icon = base64.b64encode(ICON_PATH.read_bytes()).decode("utf-8")
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:.75rem;">
                <img src="data:image/png;base64,{encoded_icon}" width="48" height="48"/>
                <h1 style="margin:0;">Content Indexing</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.header("Content Indexing")
    
    # Create tabs for different views
    tab1, tab_manual, tab2 = st.tabs(["Knowledge Base", "Manual Entry", "Indexing Tool"])

    kb_entries_light: list[tuple] | None = None
    kb_entries_full: list[tuple] | None = None

    def load_entries(*, include_markdown: bool = False) -> list[tuple]:
        """Fetch knowledge base rows, caching lightweight and full variants."""
        nonlocal kb_entries_light, kb_entries_full
        cache = kb_entries_full if include_markdown else kb_entries_light
        if cache is not None:
            return cache

        data = sorted(
            get_kb_entries(include_markdown=include_markdown),
            key=lambda entry: len(entry[1] or ""),
        )
        if include_markdown:
            kb_entries_full = data
        else:
            kb_entries_light = data
        return data
    
    with tab_manual:
        st.header("Add Knowledge Base Entry Manually")
        st.caption("Create documents directly in the knowledge base. Entries are stored under **Internal documents** and summarized automatically.")

        success_payload = st.session_state.pop("manual_entry_success", None)
        if success_payload:
            st.success(
                f"Saved **{success_payload['title']}** as `{success_payload['url']}`. Summary, tags, and language were generated automatically."
            )

        st.info(
            """**Markdown formatting tips**

- Start with a concise heading, e.g. `# Resetting Passwords`.
- Keep paragraphs or bullet points focused on one concept.
- Use Q&A blocks only when they mirror how teammates will ask questions.
- Show examples in fenced code blocks (```bash ...```); use bold or block quotes for callouts.
- Keep each chunk short (2–4 brief paragraphs) so embeddings capture a single intent.
- Include inline metadata when useful, e.g. `**Audience:** Support`, `**Updated:** 2024-09-01`.
"""
        )

        default_date = datetime.now(timezone.utc).date()
        manual_form_defaults = {
            "manual_title": "",
            "manual_safe_title": "",
            "manual_identifier": "",
            "manual_markdown": "",
            "manual_crawl_date": default_date,
            "manual_page_type": "text",
            "manual_no_upload": False,
            "manual_form_submitting": False,
        }

        if st.session_state.pop("manual_form_reset_pending", False):
            st.session_state.update(manual_form_defaults)

        for key, value in manual_form_defaults.items():
            st.session_state.setdefault(key, value)

        status_placeholder = st.empty()
        if st.session_state["manual_form_submitting"]:
            status_placeholder.info("Saving manual entry…")

        with st.form("manual_kb_entry"):
            manual_title = st.text_input("Title", key="manual_title")
            manual_safe_title = st.text_input(
                "Safe Title (optional)",
                key="manual_safe_title",
                help="Used for generated filenames. Leave blank to auto-generate from the title."
            )
            manual_identifier = st.text_input(
                "Document identifier",
                key="manual_identifier",
                help="Optional slug. Stored as internal://internal-documents/<identifier>. Leave blank to derive from the title.",
            )
            manual_markdown = st.text_area("Markdown Content", height=260, key="manual_markdown")
            manual_crawl_date = st.date_input(
                "Crawl date",
                key="manual_crawl_date"
            )
            options_page_type = ["text", "links", "other"]
            manual_page_type = st.selectbox(
                "Page type",
                options=options_page_type,
                index=options_page_type.index(st.session_state["manual_page_type"]),
                key="manual_page_type"
            )
            manual_no_upload = st.checkbox(
                "Exclude from vector store",
                value=st.session_state["manual_no_upload"],
                key="manual_no_upload"
            )
            submitted_manual = st.form_submit_button(
                "Save entry",
                disabled=st.session_state["manual_form_submitting"],
                icon=":material/save:"
            )

        if submitted_manual:
            st.session_state["manual_form_submitting"] = True
            status_placeholder.info("Saving manual entry…")
            errors: list[str] = []
            manual_markdown_clean = manual_markdown.strip()
            if not manual_markdown_clean:
                errors.append("Markdown content cannot be empty.")
            manual_title_clean = manual_title.strip() or "Untitled"

            if errors:
                st.session_state["manual_form_submitting"] = False
                status_placeholder.empty()
                for err in errors:
                    st.error(err)
            else:
                with st.spinner("Saving manual entry…"):
                    safe_title_manual = manual_safe_title.strip()
                    if not safe_title_manual:
                        safe_title_manual = re.sub(
                            r"_+",
                            "_",
                            "".join(c if c.isalnum() else "_" for c in manual_title_clean or "untitled"),
                        )[:64]

                    identifier = manual_identifier.strip() or safe_title_manual or "internal_doc"
                    normalized_manual_url = f"internal://internal-documents/{identifier}"

                    llm_result = None
                    try:
                        llm_result = summarize_and_tag_tooluse(manual_markdown_clean)
                    except Exception as exc:
                        st.session_state["manual_form_submitting"] = False
                        status_placeholder.empty()
                        st.error(f"LLM summarization failed: {exc}")

                    if not llm_result:
                        st.session_state["manual_form_submitting"] = False
                        status_placeholder.empty()
                        st.error("Could not generate summary and metadata; entry not saved.")
                    else:
                        summary_text = llm_result.get("summary", "")
                        tags_value = llm_result.get("tags", [])
                        if isinstance(tags_value, str):
                            try:
                                tags_value = json.loads(tags_value)
                            except Exception:
                                tags_value = [t.strip().strip("'\" ") for t in tags_value.strip("[]").split(",") if t.strip()]
                        tags_list = normalize_tags_for_storage(tags_value if isinstance(tags_value, list) else tags_value)
                        language = llm_result.get("detected_language") or "unknown"

                        conn = None
                        try:
                            conn = get_connection()
                            # Look up the source_config_id for "internal://" (Internal documents)
                            with conn.cursor() as cur:
                                cur.execute("SELECT id FROM url_configs WHERE url = 'internal://'")
                                row = cur.fetchone()
                                internal_config_id = row[0] if row else None
                            save_document_to_db(
                                conn,
                                normalized_manual_url,
                                manual_title_clean,
                                safe_title_manual,
                                manual_crawl_date.isoformat() if hasattr(manual_crawl_date, "isoformat") else str(manual_crawl_date),
                                language,
                                summary_text,
                                tags_list,
                                manual_markdown_clean,
                                compute_sha256(manual_markdown_clean),
                                "Internal documents",
                                manual_page_type,
                                manual_no_upload,
                                source_config_id=internal_config_id,
                            )
                        except Exception as exc:
                            st.session_state["manual_form_submitting"] = False
                            status_placeholder.empty()
                            st.error(f"Failed to save manual entry: {exc}")
                        else:
                            with st.expander("Generated metadata", expanded=True):
                                st.write(f"**Language:** {language}")
                                st.write(f"**Summary:** {summary_text or '(empty)'}")
                                st.write(f"**Tags:** {', '.join(tags_list) if tags_list else '(none)'}")
                            st.session_state["manual_entry_success"] = {
                                "title": manual_title_clean,
                                "url": normalized_manual_url,
                            }
                            st.session_state["manual_form_submitting"] = False
                            status_placeholder.empty()
                            st.session_state["manual_form_reset_pending"] = True
                            rerun_app()
                        finally:
                            if conn:
                                conn.close()

    with tab2:
        st.header("Scrape webpages")
    
        st.info("**Path-Based Scraping**: The scraper will only follow links that are within the same path as the starting URL. "
                "For example, if you start with `/suche/index.html`, it will only scrape pages under `/suche/` "
                "and its subdirectories.", icon=":material/info:")

        # Initialize database table for URL configs
        try:
            create_url_configs_table()
            initialize_default_url_configs()
        except Exception as e:
            st.error(f"Failed to initialize URL configurations table: {e}")

        # Initialize URL configs in session state from database
        # Always reload after order changes to ensure consistency
        if "url_configs" not in st.session_state or st.session_state.get("_reload_url_configs"):
            try:
                st.session_state.url_configs = load_url_configs()
                st.session_state._reload_url_configs = False
                if not st.session_state.url_configs:
                    # If no configs in DB, start with empty list
                    st.session_state.url_configs = []
            except Exception as e:
                st.error(f"Failed to load URL configurations: {e}")
                st.session_state.url_configs = []
        ensure_url_config_ids(st.session_state.url_configs)

        # Status Dashboard - Give users immediate overview of system state
        st.markdown("---")
        st.subheader("System Status")
        try:
            metrics = get_document_metrics()
        except Exception as exc:
            st.error(f"Could not load system status: {exc}")
        else:
            total_pages = metrics["total_pages"]
            pending_sync = metrics["pending_sync"]
            stale_pages = metrics["stale_pages"]
            total_configs = len(st.session_state.url_configs)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📄 Total Pages", total_pages, border=True)
            with col2:
                sync_color = "🟢" if pending_sync == 0 else "🟡"
                st.metric(f"{sync_color} Pending Sync", pending_sync, border=True)
            with col3:
                config_color = "🟢" if total_configs > 0 else "🔴"
                st.metric(f"{config_color} URL Configurations", total_configs, border=True)
            with col4:
                if stale_pages:
                    st.metric("🕰️ Stale Pages", stale_pages, border=True)
                elif pending_sync > 0:
                    st.info(
                        f"⏳ {pending_sync} awaiting vector sync.\n"
                        "Run the Vectorize page to process them.",
                        icon=":material/refresh:",
                    )
                else:
                    st.success("All synced", icon=":material/check_circle:")

            if total_pages == 0:
                st.info("**Welcome!** Add your first URL configuration below to start indexing content.", icon=":material/rocket_launch:")
            elif pending_sync > 0:
                st.warning(f"**{pending_sync} pages** are waiting for vector store synchronization.", icon=":material/sync_problem:")
            elif stale_pages > 0:
                st.warning(f"**{stale_pages} pages** look stale based on the latest crawl.", icon=":material/auto_delete:")
            else:
                st.success(f"**System healthy** - All {total_pages} pages are indexed and synchronized.", icon=":material/check_circle:")

        # Crawl settings (global for a run)
        st.markdown("---")
        st.subheader("URL Configurations")

        recordset_counts = defaultdict(list)
        for cfg in st.session_state.url_configs:
            rs_name = (cfg.get("recordset") or "").strip()
            if rs_name:
                recordset_counts[rs_name].append(cfg.get("url") or "No URL")
        duplicate_recordsets = {name: urls for name, urls in recordset_counts.items() if len(urls) > 1}
        if duplicate_recordsets:
            dup_list = ", ".join(f"'{name}' ({len(urls)} configs)" for name, urls in duplicate_recordsets.items())
            st.warning(
                f"Recordset names must be unique per configuration. Resolve duplicates: {dup_list}.",
                icon=":material/error:",
            )

        # Create a placeholder for status messages
        message_placeholder = st.empty()


        # Render each URL config with improved layout
        for i, config in enumerate(st.session_state.url_configs):
            config_id = config.setdefault("_id", uuid.uuid4().hex)
            # Create a container for each configuration with better visual separation
            with st.expander(f"URL Configuration {i+1}" + (f" - {config.get('url', 'No URL set')[:50]}..." if config.get('url') else ""), expanded=False, icon=":material/web_asset:"):
                # Reordering controls
                move_cols = st.columns([0.2, 0.2, 0.2, 0.4])
                with move_cols[0]:
                    move_to_start_disabled = i == 0
                    if st.button(
                        f"Move config {i+1} to start",
                        icon=":material/vertical_align_top:",
                        key=f"move_start_{config_id}",
                        type="secondary",
                        disabled=move_to_start_disabled,
                    ):
                        config_entry = st.session_state.url_configs.pop(i)
                        st.session_state.url_configs.insert(0, config_entry)
                        try:
                            save_url_configs(st.session_state.url_configs)
                            st.session_state._reload_url_configs = True
                            st.success(f"Configuration {i+1} moved to beginning!", icon=":material/check_circle:")
                        except Exception as e:
                            st.error(f"Failed to save configuration order: {e}")
                        st.rerun()
                with move_cols[1]:
                    move_up_disabled = i == 0
                    if st.button(
                        f"Move config {i+1} up",
                        icon=":material/arrow_upward:",
                        key=f"move_up_{config_id}",
                        type="secondary",
                        disabled=move_up_disabled,
                    ):
                        st.session_state.url_configs[i - 1], st.session_state.url_configs[i] = (
                            st.session_state.url_configs[i],
                            st.session_state.url_configs[i - 1],
                        )
                        try:
                            save_url_configs(st.session_state.url_configs)
                            st.session_state._reload_url_configs = True
                            st.success(f"Configuration {i+1} moved up!", icon=":material/check_circle:")
                        except Exception as e:
                            st.error(f"Failed to save configuration order: {e}")
                        st.rerun()
                with move_cols[2]:
                    move_down_disabled = i == len(st.session_state.url_configs) - 1
                    if st.button(
                        f"Move config {i+1} down",
                        icon=":material/arrow_downward:",
                        key=f"move_down_{config_id}",
                        type="secondary",
                        disabled=move_down_disabled,
                    ):
                        st.session_state.url_configs[i + 1], st.session_state.url_configs[i] = (
                            st.session_state.url_configs[i],
                            st.session_state.url_configs[i + 1],
                        )
                        try:
                            save_url_configs(st.session_state.url_configs)
                            st.session_state._reload_url_configs = True
                            st.success(f"Configuration {i+1} moved down!", icon=":material/check_circle:")
                        except Exception as e:
                            st.error(f"Failed to save configuration order: {e}")
                        st.rerun()
                with move_cols[3]:
                    move_to_end_disabled = i == len(st.session_state.url_configs) - 1
                    if st.button(
                        f"Move config {i+1} to end",
                        icon=":material/vertical_align_bottom:",
                        key=f"move_end_{config_id}",
                        type="secondary",
                        disabled=move_to_end_disabled,
                    ):
                        config_entry = st.session_state.url_configs.pop(i)
                        st.session_state.url_configs.append(config_entry)
                        try:
                            save_url_configs(st.session_state.url_configs)
                            st.session_state._reload_url_configs = True
                            st.success(f"Configuration {i+1} moved to end!", icon=":material/check_circle:")
                        except Exception as e:
                            st.error(f"Failed to save configuration order: {e}")
                        st.rerun()

                # Use columns for better layout
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.session_state.url_configs[i]["url"] = st.text_input(
                        f"Start URL",
                        value=config["url"],
                        key=f"start_url_{config_id}",
                        help="The scraper will only follow links within the same path as this starting URL",
                        placeholder="https://example.com/path/to/start"
                    )

                with col2:
                    st.session_state.url_configs[i]["depth"] = st.number_input(
                        f"Max scraping depth",
                        min_value=0,
                        max_value=15,
                        value=config["depth"],
                        step=1,
                        key=f"depth_{config_id}",
                        help="How many levels deep to follow links"
                    )

                st.session_state.url_configs[i]["recordset"] = st.text_input(
                    f"Recordset name",
                    value=config.get("recordset") or "",
                    key=f"recordset_name_{config_id}",
                    help="Recordsets group the documents produced by this configuration. Provide a unique name per configuration."
                )

                # Path filters with better layout
                col1, col2 = st.columns(2)
                
                with col1:
                    exclude_paths_str = st.text_area(
                        f"Exclude paths (comma-separated)",
                        value=", ".join(config.get("exclude_paths", [])),
                        key=f"exclude_paths_{config_id}",
                        height=100,
                        help="Paths to exclude from scraping (supports '*' wildcards, e.g., /en, /site-*, /admin)",
                        placeholder="/en, /pl, /site-*, /_ablage-alte-www"
                    )
                    st.session_state.url_configs[i]["exclude_paths"] = [path.strip() for path in exclude_paths_str.split(",") if path.strip()]

                with col2:
                    include_prefixes_str = st.text_area(
                        f"Restrict to prefixes or paths (comma-separated)",
                        value=", ".join(config.get("include_lang_prefixes", [])),
                        key=f"include_lang_prefixes_{config_id}",
                        height=100,
                        help="Only include paths starting with these prefixes (supports '*' wildcards, e.g., /de, /fr, /research-*)",
                        placeholder="/de, /fr, /research-*"
                    )
                    st.session_state.url_configs[i]["include_lang_prefixes"] = [prefix.strip() for prefix in include_prefixes_str.split(",") if prefix.strip()]

                # Save and Delete buttons for this configuration
                st.markdown("---")
                col_save, col_delete, col_spacer = st.columns([1, 1, 2])

                with col_save:
                    if st.button(f"Save config {i+1}", icon=":material/save:", key=f"save_config_{config_id}", type="primary"):
                        try:
                            save_url_configs(st.session_state.url_configs)
                            st.success(f"Configuration {i+1} saved!", icon=":material/check_circle:")
                        except Exception as e:
                            st.error(f"Failed to save configuration {i+1}: {e}")

                with col_delete:
                    if st.button(f"Delete config {i+1}", icon=":material/delete:", key=f"delete_config_{config_id}", type="secondary"):
                        # Remove this specific configuration
                        st.session_state.url_configs.pop(i)
                        try:
                            save_url_configs(st.session_state.url_configs)
                            st.success(f"Configuration {i+1} deleted and saved!", icon=":material/check_circle:")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to save after deletion: {e}")
        # Show number of configurations
        # Quick Add Single URL - Simple interface for basic use cases
        with st.expander("**Add URL Configuration**", expanded=True, icon=":material/add_circle:"):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                quick_url = st.text_input(
                    "URL to scrape", 
                    placeholder="https://example.com/path/to/content",
                    help="Enter any URL - the scraper will only follow links within the same path",
                    key="quick_url_input"
                )
            with col2:
                quick_template = st.selectbox(
                    "Template",
                    ["Default", "Blog", "Documentation", "News Site"],
                    help="Choose a pre-configured template for common website types"
                )
            with col3:
                st.markdown("<br>", unsafe_allow_html=True)  # Align button with input
                if st.button("Add URL", disabled=not quick_url, type="primary", icon=":material/add_circle:"):
                    # Create configuration based on template
                    template_configs = {
                        "Default": {
                            "depth": 2,
                            "exclude_paths": ["/en/", "/pl/", "/_ablage-alte-www/", "/site-euv/", "/site-zwe-ikm/"],
                            "include_lang_prefixes": []
                        },
                        "Blog": {
                            "depth": 2,
                            "exclude_paths": ["/tag/", "/category/", "/author/", "/page/"],
                            "include_lang_prefixes": []
                        },
                        "Documentation": {
                            "depth": 4,
                            "exclude_paths": ["/api/", "/_internal/", "/admin/"],
                            "include_lang_prefixes": []
                        },
                        "News Site": {
                            "depth": 2,
                            "exclude_paths": ["/archive/", "/tag/", "/category/"],
                            "include_lang_prefixes": []
                        }
                    }
                    
                    config = template_configs[quick_template]
                    st.session_state.url_configs.append({
                        "id": None,
                        "url": quick_url,
                        "recordset": f"Quick_{quick_template}_{len(st.session_state.url_configs)+1}",
                        "depth": config["depth"],
                        "exclude_paths": config["exclude_paths"],
                        "include_lang_prefixes": config["include_lang_prefixes"],
                        "_id": uuid.uuid4().hex,
                    })
                    
                    try:
                        save_url_configs(st.session_state.url_configs)
                        st.success(f"Quick configuration added! Using {quick_template} template.", icon=":material/check_circle:")
                        st.info("**Next step**: Scroll down to the 'Start Indexing' section to begin scraping!", icon=":material/info:")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to save quick configuration: {e}")
        
        # Global configuration management
        st.markdown("---")
        st.markdown("### Configuration Management")
        st.info("**Tip**: Individual configurations auto-save when you use their save buttons. Use the buttons below for bulk operations.", icon=":material/info:")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Save All Changes**")
            st.caption("Save all configuration changes to the database")
            if st.button("Save all configurations", type="primary", icon=":material/save:"):
                try:
                    save_url_configs(st.session_state.url_configs)
                    message_placeholder.success("All configurations saved to database!", icon=":material/check_circle:")
                except Exception as e:
                    message_placeholder.error(f"Failed to save configurations: {e}")
        
        with col2:
            st.markdown("**Reset to Database**")
            st.caption("Discard unsaved changes and reload from database")
            if st.button("Reset to saved", type="secondary", icon=":material/restore:"):
                try:
                    st.session_state.url_configs = load_url_configs()
                    ensure_url_config_ids(st.session_state.url_configs)
                    message_placeholder.success(f"Reset to saved state: {len(st.session_state.url_configs)} configurations loaded!", icon=":material/check_circle:")
                    st.rerun()
                except Exception as e:
                    message_placeholder.error(f"Failed to reload configurations: {e}")
                    
        if st.session_state.url_configs:
            st.success(f"**{len(st.session_state.url_configs)} configuration(s)** ready for indexing", icon=":material/check_circle:")
        else:
            st.info("**No configurations yet** - Add your first URL configuration above", icon=":material/info:")


        st.markdown("---")
        if "scraper_schedule" not in st.session_state:
            st.session_state["scraper_schedule"] = read_scraper_schedule()

        st.header("Scheduled Re-Sraping")

        def _parse_ts(value):
            if not value:
                return None
            try:
                parsed = datetime.fromisoformat(value)
            except ValueError:
                return None
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed

        def _parse_run_times_input(raw: str) -> list[str]:
            normalized: list[str] = []
            seen: set[str] = set()
            for token in raw.split(","):
                token = token.strip()
                if not token:
                    continue
                parts = token.split(":")
                if len(parts) != 2:
                    continue
                try:
                    hour = int(parts[0])
                    minute = int(parts[1])
                except ValueError:
                    continue
                if not (0 <= hour < 24 and 0 <= minute < 60):
                    continue
                value = f"{hour:02d}:{minute:02d}"
                if value not in seen:
                    seen.add(value)
                    normalized.append(value)
            return sorted(normalized)

        def _parse_run_times_utc(values: list[str] | None) -> list[time]:
            result: list[time] = []
            if not values:
                return result
            for token in values:
                try:
                    hour, minute = (int(part) for part in token.split(":", 1))
                except ValueError:
                    continue
                if not (0 <= hour < 24 and 0 <= minute < 60):
                    continue
                result.append(time(hour=hour, minute=minute, tzinfo=timezone.utc))
            return sorted(result)

        def _resolve_timezone(name: str | None) -> ZoneInfo:
            try:
                return ZoneInfo((name or "UTC").strip() or "UTC")
            except Exception:
                return ZoneInfo("UTC")

        def _run_times_utc_to_local(times_utc: list[str], tz: ZoneInfo) -> list[str]:
            if not times_utc:
                return []
            today = datetime.now(tz).date()
            converted: list[str] = []
            seen: set[str] = set()
            for token in times_utc:
                try:
                    hour, minute = (int(part) for part in token.split(":", 1))
                except ValueError:
                    continue
                utc_dt = datetime.combine(today, time(hour=hour, minute=minute, tzinfo=timezone.utc))
                local_dt = utc_dt.astimezone(tz)
                value = local_dt.strftime("%H:%M")
                if value not in seen:
                    seen.add(value)
                    converted.append(value)
            return sorted(converted)

        def _run_times_local_to_utc(times_local: list[str], tz: ZoneInfo) -> list[str]:
            if not times_local:
                return []
            today = datetime.now(tz).date()
            converted: list[str] = []
            seen: set[str] = set()
            for token in times_local:
                try:
                    hour, minute = (int(part) for part in token.split(":", 1))
                except ValueError:
                    continue
                local_dt = datetime.combine(today, time(hour=hour, minute=minute, tzinfo=tz))
                utc_dt = local_dt.astimezone(timezone.utc)
                value = utc_dt.strftime("%H:%M")
                if value not in seen:
                    seen.add(value)
                    converted.append(value)
            return sorted(converted)

        def _next_run_time(last_run: datetime | None, run_times: list[time], now: datetime) -> datetime | None:
            if not run_times:
                return None
            reference = last_run or now
            if reference.tzinfo is None:
                reference = reference.replace(tzinfo=timezone.utc)
            start_date = (reference - timedelta(days=1)).date()
            end_date = reference.date() + timedelta(days=3)
            candidates: list[datetime] = []
            current = start_date
            while current <= end_date:
                for rt in run_times:
                    candidates.append(datetime.combine(current, rt))
                current += timedelta(days=1)
            candidates.sort()
            for slot in candidates:
                if slot > reference:
                    return slot
            future_date = end_date + timedelta(days=1)
            return datetime.combine(future_date, run_times[0])

        schedule_state = st.session_state["scraper_schedule"]
        timezone_name = schedule_state.get("timezone", "UTC") or "UTC"
        schedule_tz = _resolve_timezone(timezone_name)
        stored_run_times = schedule_state.get("run_times", [])
        if schedule_state.get("run_times_are_utc", False):
            local_run_times = _run_times_utc_to_local(stored_run_times, schedule_tz)
        else:
            local_run_times = stored_run_times

        last_run_dt = _parse_ts(schedule_state.get("last_run_at"))
        now_utc = datetime.now(timezone.utc)
        run_times_utc = _parse_run_times_utc(_run_times_local_to_utc(local_run_times, schedule_tz))
        next_run_dt = None
        if run_times_utc:
            next_run_dt = _next_run_time(last_run_dt, run_times_utc, now_utc)
        elif last_run_dt:
            next_run_dt = last_run_dt + timedelta(hours=schedule_state.get("interval_hours", 12))
        else:
            next_run_dt = now_utc + timedelta(hours=schedule_state.get("interval_hours", 12))

        with st.expander("Configure scheduled CLI runner", expanded=False):
            st.caption(
                "These settings are read by `scripts/run_cli_if_due.py`. Trigger it from your host cron or systemd "
                "timer using the command below so heavy scraping/vectorization runs outside the UI container."
            )

            enabled = st.checkbox("Enable scheduled runs", value=bool(schedule_state.get("enabled", False)))
            timezone_input = st.text_input(
                "Schedule time zone (IANA name)",
                value=timezone_name,
                help="Daily times below are interpreted in this time zone (e.g., Europe/Berlin for CET/CEST).",
            )
            selected_timezone_name = timezone_input.strip() or "UTC"
            selected_timezone = _resolve_timezone(selected_timezone_name)
            run_times_input = st.text_input(
                "Daily start times (HH:MM, comma separated)",
                value=", ".join(local_run_times),
                help="Exact times when the CLI runner should start (24h format) in the selected time zone.",
            )
            parsed_run_times = _parse_run_times_input(run_times_input)
            interval_hours = st.number_input(
                "Interval (hours)",
                min_value=0.5,
                max_value=168.0,
                value=float(schedule_state.get("interval_hours", 12.0)),
                step=0.5,
                help="Fallback interval used only when no explicit start times are configured.",
            )

            mode_label_map = {
                "scrape": "Scrape only",
                "vectorize": "Vector sync only",
                "sync": "Scrape + vector sync",
                "all": "Scrape + vector sync + cleanup (default)",
                "cleanup": "Vector cleanup (orphans only)",
            }
            inverse_mode_map = {v: k for k, v in mode_label_map.items()}
            default_mode_label = mode_label_map.get(schedule_state.get("mode", "all"), mode_label_map["all"])
            mode_options = list(mode_label_map.values())
            try:
                default_mode_index = mode_options.index(default_mode_label)
            except ValueError:
                default_mode_index = mode_options.index("Scrape + vector sync + cleanup (default)")
            selected_mode_label = st.selectbox(
                "Scheduled job mode",
                options=mode_options,
                index=default_mode_index,
            )

            scheduled_budget = st.number_input(
                "Crawl budget for scheduled run",
                min_value=1000,
                max_value=20000,
                step=1000,
                value=int(schedule_state.get("crawl_budget", 5000)),
            )

            scheduled_keep_query = st.text_input(
                "Whitelist query keys (comma-separated)",
                value=schedule_state.get("keep_query", ""),
            )

            scheduled_dry_run = st.checkbox(
                "Dry run (skip DB writes/LLM)",
                value=bool(schedule_state.get("dry_run", False)),
            )

            if last_run_dt:
                last_local = last_run_dt.astimezone(selected_timezone)
                next_local = next_run_dt.astimezone(selected_timezone) if next_run_dt else None
                st.info(
                    f"Last scheduled run: {last_local.strftime('%Y-%m-%d, **%H:%M:%S** %Z')} | "
                    + (f"Next due at {next_local.strftime('%Y-%m-%d, **%H:%M:%S** %Z')}" if next_local else ""),
                    icon=":material/history:",
                )
            else:
                st.caption("Scheduler has not run yet. Use the cron command below to trigger it.")

            if parsed_run_times:
                st.caption(
                    f"Configured daily times (local {selected_timezone.key}): {', '.join(parsed_run_times)}"
                )
            else:
                st.caption(
                    "No explicit times set. The helper falls back to the interval above, so it runs every "
                    f"{interval_hours:g} hours when triggered."
                )

            previous_enabled = bool(schedule_state.get("enabled", False))

            if st.button("Save scheduler settings", icon=":material/save:"):
                updated_schedule = {
                    "enabled": enabled,
                    "interval_hours": interval_hours,
                    "run_times": parsed_run_times,
                    "run_times_are_utc": False,
                    "mode": inverse_mode_map[selected_mode_label],
                    "crawl_budget": scheduled_budget,
                    "keep_query": scheduled_keep_query,
                    "dry_run": scheduled_dry_run,
                    "last_run_at": schedule_state.get("last_run_at"),
                    "timezone": selected_timezone_name,
                }

                if enabled and not previous_enabled and not updated_schedule["last_run_at"]:
                    updated_schedule["last_run_at"] = datetime.now(timezone.utc).isoformat()

                schedule_state = write_scraper_schedule(updated_schedule)
                st.session_state["scraper_schedule"] = schedule_state
                st.success("Scheduler settings saved.", icon=":material/check_circle:")
                _time.sleep(2)
                rerun_app()

            st.markdown("**Cron/systemd command examples**")
            st.code(
                ".venv/bin/python scripts/run_cli_if_due.py  # bare-metal Python\n"
                "docker compose run --rm --entrypoint python cli-runner scripts/run_cli_if_due.py\n"
                "podman ccompose run --rm --entrypoint python cli-runner scripts/run_cli_if_due.py",
                language="bash",
            )
            st.caption(
                "Invoke the command above from your preferred scheduler. The helper script exits immediately when the "
                "interval has not elapsed, and runs `cli_scrape.py` (in the selected mode) when due."
            )

        st.markdown("## Manual Indexing (override)")
        st.info(
            "Use the manual controls below when you need to trigger an immediate run outside the configured schedule.",
            icon=":material/schedule:",
        )

        with st.expander("Show manual indexing controls", expanded=False):
            st.markdown("**Indexing will:** Discover and crawl pages, process content with LLM, save to knowledge base, show real-time progress")

            if not st.session_state.url_configs:
                st.warning(
                    "**No URL configurations found.** Add at least one entry above to run scraping jobs. "
                    "Vector cleanup-only runs remain available.",
                    icon=":material/warning:",
                )

            st.markdown("### Crawler settings")
            colA, colB = st.columns(2)
            with colA:
                max_urls_per_run = st.number_input(
                    "Max URLs per run (crawl budget)",
                    min_value=1000,
                    max_value=20000,
                    value=5000,
                    step=1000,
                )
            with colB:
                keep_query_str = st.text_input(
                    "Whitelist query keys (comma-separated)",
                    value="",
                    help="By default the crawler normalizes links by stripping all query strings (so /page?id=123 and /page?id=456 collapse to the same URL and you don’t crawl endless variants). If you enter a list here—e.g. page,lang—those keys are preserved while all others are dropped. Use it when certain query params are meaningful (pagination, language, etc.) and you need the crawler to treat ?page=2 or ?lang=en as distinct pages, while still ignoring the rest (like tracking tags).",
                )

            manual_dry_run = st.checkbox(
                "Dry run (no DB writes, no LLM calls)",
                value=True,
                help="When enabled, the crawler won't write to the database or call the LLM. It will only traverse and show which URLs would be processed.",
            )

            if "scrape_cli_job" not in st.session_state:
                st.session_state["scrape_cli_job"] = None
            if "scrape_cli_message" not in st.session_state:
                st.session_state["scrape_cli_message"] = None
            if "scrape_cli_last_return" not in st.session_state:
                st.session_state["scrape_cli_last_return"] = None

            job_mode_options = [
                "Scrape only",
                "Scrape + vector sync",
                "Scrape + vector sync + cleanup",
                "Vector cleanup only",
            ]
            default_job_mode = "Scrape + vector sync + cleanup"
            default_index = job_mode_options.index(default_job_mode)
            job_mode = st.radio(
                "Job mode",
                options=job_mode_options,
                index=default_index if not manual_dry_run else 0,
                help="Choose whether to run vector store synchronization (optionally with cleanup) after scraping, or just purge orphaned vector files.",
            )
            if manual_dry_run and job_mode != "Scrape only":
                st.info("Dry run enabled — skipping vector store synchronization.", icon=":material/info:")
                job_mode = "Scrape only"

            cli_job: CLIJob | None = st.session_state.get("scrape_cli_job")
            job_running = cli_job is not None and cli_job.is_running()

            lock_check_error = None
            external_job_running = False
            try:
                lock_held = is_job_locked("scrape_job")
                external_job_running = lock_held and not job_running
            except RuntimeError as exc:
                lock_check_error = str(exc)

            if lock_check_error:
                st.warning(f"Could not verify indexing job status: {lock_check_error}", icon=":material/warning:")
            elif external_job_running:
                st.warning(
                    "Another indexing/vectorization job is currently running (scheduled cron or different session). "
                    "Please wait for it to finish before starting a new run.",
                    icon=":material/pending_actions:",
                )

            start_button_disabled = job_running or external_job_running

            if st.button(
                "**Start Indexing All URLs**",
                type="primary",
                width="stretch",
                disabled=start_button_disabled,
                icon=":material/rocket_launch:",
            ):
                if job_running:
                    st.warning("An indexing job is already running.", icon=":material/info:")
                elif external_job_running:
                    st.warning(
                        "Another indexing/vectorization job is active. Please retry after it finishes.",
                        icon=":material/pending_actions:",
                    )
                elif (
                    job_mode != "Vector cleanup only"
                    and not any(config.get('url', '').strip() for config in st.session_state.url_configs)
                ):
                    st.error("No valid URLs found in configurations. Please add at least one URL.", icon=":material/error:")
                else:
                    cli_args = ["--budget", str(max_urls_per_run)]
                    if keep_query_str:
                        cli_args.extend(["--keep-query", keep_query_str])
                    if manual_dry_run:
                        cli_args.append("--dry-run")

                    if job_mode == "Vector cleanup only":
                        cli_mode = "cleanup"
                    elif job_mode == "Scrape + vector sync + cleanup" and not manual_dry_run:
                        cli_mode = "all"
                    elif job_mode == "Scrape + vector sync" and not manual_dry_run:
                        cli_mode = "sync"
                    else:
                        cli_mode = "scrape"

                    overlay = show_blocking_overlay()
                    rerun_needed = False
                    try:
                        try:
                            save_url_configs(st.session_state.url_configs)
                        except Exception as exc:
                            st.error(f"Failed to save configurations before starting job: {exc}")
                            return

                        try:
                            job = launch_cli_job(mode=cli_mode, args=cli_args)
                        except CLIJobError as exc:
                            st.error(f"Failed to start scraping job: {exc}", icon=":material/error:")
                        else:
                            st.session_state["scrape_cli_job"] = job
                            st.session_state["scrape_cli_message"] = "Scrape job running via cli_scrape.py."
                            st.session_state["scrape_cli_last_return"] = None
                            rerun_needed = True
                    finally:
                        hide_blocking_overlay(overlay)
                    if rerun_needed:
                        rerun_app()

            if st.session_state["scrape_cli_message"]:
                st.info(st.session_state["scrape_cli_message"], icon=":material/center_focus_weak:")

            if cli_job:
                st.markdown("#### CLI Scrape Log")
                log_lines = "\n".join(list(cli_job.logs))

                render_log_output(log_lines, element_id="scrape-log")

                job_running = cli_job.is_running()
                if job_running:
                    st.info(
                        f"Scrape job in progress (PID {cli_job.process.pid}). Use the controls below to refresh or cancel.",
                        icon=":material/hourglass:",
                    )
                    refresh_col, cancel_col = st.columns(2)
                    with refresh_col:
                        st.button("Refresh status", key="refresh_scrape_job", icon=":material/refresh:")
                    with cancel_col:
                        if st.button("  Cancel job  ", key="cancel_scrape_job", icon=":material/cancel:"):
                            cli_job.terminate()
                            try:
                                release_job_lock("scrape_job")
                            except Exception:
                                pass
                            st.session_state["scrape_cli_message"] = "Cancellation requested. Check the log for confirmation."
                            rerun_app()
                else:
                    if st.session_state["scrape_cli_last_return"] is None:
                        st.session_state["scrape_cli_last_return"] = cli_job.returncode()

                if not job_running:
                    if st.session_state["scrape_cli_last_return"] is None:
                        st.session_state["scrape_cli_last_return"] = cli_job.returncode()
                    exit_code = st.session_state["scrape_cli_last_return"] or 0
                    if exit_code == 0:
                        st.session_state["scrape_cli_message"] = None
                        st.success("Scrape job completed successfully.", icon=":material/check_circle:")
                        try:
                            release_job_lock("scrape_job")
                        except Exception:
                            pass
                    else:
                        st.session_state["scrape_cli_message"] = None
                        st.error(f"Scrape job finished with exit code {exit_code}.", icon=":material/error:")
                        try:
                            release_job_lock("scrape_job")
                        except Exception:
                            pass

                    st.button("Refresh status", key="refresh_scrape_job_done", icon=":material/refresh:")
                    if st.button("Clear log", key="clear_scrape_job", icon=":material/delete:"):
                        st.session_state["scrape_cli_job"] = None
                        st.session_state["scrape_cli_message"] = None
                        st.session_state["scrape_cli_last_return"] = None
                        rerun_app()

        # --- Show current knowledge base entries ---
        # Fetch entries for filters and display
        entries = load_entries(include_markdown=False)

        def _normalize_tags(raw) -> list[str]:
            """Mirror the shared normalization used for storage."""
            return normalize_tags_for_storage(raw)

        recordsets = sorted(set(entry[9] for entry in entries if entry[9] is not None))
        page_types = sorted(set(entry[11] for entry in entries if entry[11] is not None))
        st.session_state.setdefault("kb_tag_filter", [])
        filters_should_expand = bool(st.session_state.get("kb_tag_filter"))

    with tab1:
        st.header("Browse Knowledge Base")
        st.markdown("*View, search, and manage your indexed content*")

        edit_success = st.session_state.pop("internal_edit_success", None)
        if edit_success:
            st.success(
                f"Updated **{edit_success['title']}** (`{edit_success['url']}`) successfully."
            )

        # Add summary of pages waiting for vector sync
        pending_vector_sync = len([entry for entry in entries if entry[10] is None and entry[12] is not True])
        excluded_needing_cleanup = len([entry for entry in entries if entry[10] is not None and entry[12] is True])
        stale_documents_total = len([entry for entry in entries if len(entry) > 13 and entry[13]])
        
        summary_messages = []
        if pending_vector_sync > 0:
            summary_messages.append(
                f"**{pending_vector_sync} page(s)** are waiting for vector store synchronization. "
                "These pages have been processed by LLM but haven't been vectorized yet."
            )
        if excluded_needing_cleanup > 0:
            summary_messages.append(
                f"**{excluded_needing_cleanup} excluded page(s)** still have vector files and need cleanup (run Vectorize)."
            )
        if stale_documents_total > 0:
            summary_messages.append(
                f"**{stale_documents_total} page(s)** are currently marked as stale (missing in the latest crawl)."
            )

        if summary_messages:
            st.warning("\n\n".join(summary_messages), icon=":material/warning:")
        else:
            st.info("All pages are synchronized, no stale pages detected, and no excluded files need cleanup.", icon=":material/check_circle:")

        with st.expander("Show filters", expanded=filters_should_expand):
            selected_recordset = st.selectbox(
                "Filter by recordset",
                options=["All"] + recordsets,
                index=0
            )
            selected_page_type = st.selectbox(
                "Filter by page type",
                options=["All"] + page_types,
                index=0
            )
            selected_vector_status = st.selectbox(
                "Filter by vectorization status",
                options=["All", "Non-vectorized", "Vectorized (synced)"],
                index=0,
                help="Filter entries based on whether they have been vectorized and synced to the vector store"
            )
            # New: filter by exclusion from vector store (no_upload)
            selected_exclusion_status = st.selectbox(
                "Filter by vector store exclusion",
                options=["All", "Excluded", "Included"],
                index=0,
                help="Filter entries that are excluded from vectorization"
            )
            selected_stale_status = st.selectbox(
                "Filter by stale status",
                options=["All", "Fresh", "Stale"],
                index=0,
                help="Filter entries based on whether they were missing in the latest crawl"
            )
            tag_filter_placeholder = st.container()

            search_query = st.text_input(
                "Search text",
                value="",
                placeholder="Filter by URL, title, or content…",
                help="Case-insensitive search across URL, title, and markdown content",
            )

        search_query_clean = search_query.strip()
        entries = load_entries(include_markdown=bool(search_query_clean))

        filtered = entries
        if selected_recordset != "All":
            filtered = [entry for entry in filtered if entry[9] == selected_recordset]
        if selected_page_type != "All":
            filtered = [entry for entry in filtered if entry[11] == selected_page_type]
        if selected_vector_status != "All":
            if selected_vector_status == "Non-vectorized":
                filtered = [entry for entry in filtered if entry[10] is None]
            elif selected_vector_status == "Vectorized (synced)":
                filtered = [entry for entry in filtered if entry[10] is not None]
        if selected_exclusion_status != "All":
            if selected_exclusion_status == "Excluded":
                filtered = [entry for entry in filtered if len(entry) > 12 and bool(entry[12])]
            else:
                filtered = [entry for entry in filtered if len(entry) > 12 and not bool(entry[12])]
        if selected_stale_status != "All":
            if selected_stale_status == "Stale":
                filtered = [entry for entry in filtered if len(entry) > 13 and bool(entry[13])]
            else:
                filtered = [entry for entry in filtered if len(entry) > 13 and not bool(entry[13])]
        if search_query_clean:
            needle = search_query_clean.lower()
            def _matches(entry: tuple) -> bool:
                url = (entry[1] or "").lower()
                title = (entry[2] or "").lower()
                content = (entry[8] or "").lower()
                return needle in url or needle in title or needle in content

            filtered = [entry for entry in filtered if _matches(entry)]

        filtered_before_tags = filtered

        def _collect_tags(entry_list: list[tuple]) -> set[str]:
            tag_pool: set[str] = set()
            for entry in entry_list:
                tag_pool.update(_normalize_tags(entry[7]))
            return tag_pool

        current_tag_selection = st.session_state.get("kb_tag_filter", [])

        if filtered_before_tags:
            if current_tag_selection:
                selected_set = set(current_tag_selection)
                compatible_entries = [
                    entry
                    for entry in filtered_before_tags
                    if selected_set <= set(_normalize_tags(entry[7]))
                ]
                available_tag_options = sorted(
                    set(current_tag_selection) | _collect_tags(compatible_entries)
                )
            else:
                available_tag_options = sorted(_collect_tags(filtered_before_tags))
        else:
            available_tag_options = sorted(set(current_tag_selection))

        with tag_filter_placeholder:
            selected_tags = st.multiselect(
                "Filter by tags",
                options=available_tag_options,
                default=current_tag_selection,
                key="kb_tag_filter",
                help="Show entries that contain all selected tags (AND filter).",
                placeholder="Select one or more tags" if available_tag_options else "No tags available",
                disabled=not available_tag_options,
            )

        if selected_tags:
            selected_set = set(selected_tags)
            filtered = [
                entry for entry in filtered_before_tags if selected_set <= set(_normalize_tags(entry[7]))
            ]
        else:
            filtered = filtered_before_tags

        filter_signature = (
            selected_recordset,
            selected_page_type,
            selected_vector_status,
            selected_exclusion_status,
            selected_stale_status,
            tuple(sorted(selected_tags)),
            search_query_clean,
        )
        if st.session_state.get("kb_filter_signature") != filter_signature:
            st.session_state["kb_filter_signature"] = filter_signature
            st.session_state["kb_page"] = 1

        st.session_state.setdefault("kb_page", 1)
        st.session_state.setdefault("kb_page_size", 25)

        page_size_options = [10, 25, 50, 100]
        default_page_size = st.session_state["kb_page_size"]
        if default_page_size not in page_size_options:
            default_page_size = 25
            st.session_state["kb_page_size"] = default_page_size

        page_size = st.selectbox(
            "Rows per page",
            options=page_size_options,
            index=page_size_options.index(default_page_size),
            help="Number of knowledge base entries shown per page",
        )
        if st.session_state["kb_page_size"] != page_size:
            st.session_state["kb_page_size"] = page_size
            st.session_state["kb_page"] = 1

        filtered_count = len(filtered)
        page_entries: list = []
        current_page = st.session_state.get("kb_page", 1)
        start_idx = 0
        end_idx = 0
        if filtered_count > 0:
            total_pages = math.ceil(filtered_count / st.session_state["kb_page_size"])
            current_page = max(1, min(current_page, total_pages))
            st.session_state["kb_page"] = current_page
            start_idx = (current_page - 1) * st.session_state["kb_page_size"]
            end_idx = start_idx + st.session_state["kb_page_size"]
            page_entries = filtered[start_idx:end_idx]
        else:
            total_pages = 1
            st.session_state["kb_page"] = 1

        try:
            if not filtered:
                st.info("No entries found in the knowledge base.")
            else:
                total_entries = len(entries)
                filtered_entries = filtered_count
                non_vectorized_total = len([entry for entry in entries if entry[10] is None and entry[12] is not True])
                vectorized_total = len([entry for entry in entries if entry[10] is not None])
                excluded = len([entry for entry in entries if len(entry) > 12 and bool(entry[12])])
                stale_total = stale_documents_total

                col1, col2, col3, col4, col5, col6 = st.columns(6)
                with col1:
                    st.metric("Total Entries", total_entries, border=True)
                with col2:
                    st.metric("Filtered Results", filtered_entries, border=True)
                with col3:
                    st.metric("Vectorized", vectorized_total, border=True)
                with col4:
                    st.metric("Excluded from Vector Store", excluded, border=True)
                with col5:
                    st.metric("Non-vectorized", non_vectorized_total, border=True)
                with col6:
                    st.metric("Stale Pages", stale_total, border=True)

                st.markdown("---")

                if filtered_entries > 0:
                    display_start = start_idx + 1
                    display_end = min(end_idx, filtered_entries)
                    pagination_col1, pagination_col2, pagination_col3 = st.columns([1, 2, 1])
                    with pagination_col1:
                        if st.button("◀ Previous", disabled=current_page <= 1, key="kb_prev_page"):
                            st.session_state["kb_page"] = max(1, current_page - 1)
                            rerun_app()
                    with pagination_col2:
                        st.markdown(
                            f"**Page {current_page} of {total_pages}** &nbsp;&nbsp;"
                            f"Showing entries {display_start}–{display_end} of {filtered_entries}"
                        )
                    with pagination_col3:
                        if st.button("Next ▶", disabled=current_page >= total_pages, key="kb_next_page"):
                            st.session_state["kb_page"] = min(total_pages, current_page + 1)
                            rerun_app()

                page_entries = page_entries or []

                st.session_state.setdefault("kb_selected_id", None)

                if page_entries:
                    table_rows = []
                    for entry in page_entries:
                        (
                            entry_id,
                            url,
                            title,
                            safe_title,
                            crawl_date,
                            lang,
                            summary,
                            tags,
                            markdown,
                            recordset,
                            vector_file_id,
                            page_type,
                            no_upload,
                            is_stale,
                        ) = entry

                        vector_state = (
                            "Excluded"
                            if no_upload
                            else "Vectorized" if vector_file_id else "Pending"
                        )

                        table_rows.append(
                            {
                                "ID": entry_id,
                                "Title": title or "(no title)",
                                "Recordset": recordset or "—",
                                "Lang": lang or "—",
                                "Vector": vector_state,
                                "Stale": "Yes" if is_stale else "No",
                                "Last Crawl": str(crawl_date) if crawl_date else "—",
                            }
                        )

                    table_df = pd.DataFrame(table_rows)
                    # Calculate height based on number of rows (approx 35px per row + 60px header/padding)
                    table_height = min(len(table_df), st.session_state["kb_page_size"]) * 35 + 60

                    current_ids = [entry[0] for entry in page_entries]
                    stored_id = st.session_state.get("kb_selected_id")
                    if stored_id not in current_ids:
                        stored_id = None
                        st.session_state["kb_selected_id"] = None

                    if stored_id is None and page_entries:
                        stored_id = current_ids[0]
                        st.session_state["kb_selected_id"] = stored_id

                    selection_enabled = True

                    try:
                        table_event = st.dataframe(
                            table_df,
                            hide_index=True,
                            width='stretch',
                            height=table_height,
                            selection_mode="single-row",
                            on_select="rerun",
                            key="kb_table",
                        )
                    except TypeError:
                        selection_enabled = False
                        table_event = st.dataframe(
                            table_df,
                            hide_index=True,
                            width='stretch',
                            height=table_height,
                            key="kb_table",
                        )

                    selected_entry_id = stored_id
                    if selection_enabled:
                        selected_rows: list[int] = []
                        event_selection = getattr(table_event, "selection", None)
                        if isinstance(event_selection, dict):
                            selected_rows = event_selection.get("rows", []) or []

                        if selected_rows:
                            row_idx = selected_rows[0]
                            if 0 <= row_idx < len(page_entries):
                                candidate_id = page_entries[row_idx][0]
                                if candidate_id != st.session_state.get("kb_selected_id"):
                                    st.session_state["kb_selected_id"] = candidate_id
                                    st.session_state.pop("internal_edit_id", None)
                                selected_entry_id = candidate_id

                    if selected_entry_id is None and page_entries:
                        selected_entry_id = page_entries[0][0]
                        st.session_state["kb_selected_id"] = selected_entry_id

                    if not selection_enabled:
                        st.caption(
                            "Streamlit does not support data frame row selection in this environment. "
                            "Use the fallback chooser below."
                        )
                        fallback_labels = {
                            f"[ID {entry[0]}] {entry[2] or '(no title)'}": entry[0]
                            for entry in page_entries
                        }
                        fallback_default = next(
                            (
                                label
                                for label, val in fallback_labels.items()
                                if val == selected_entry_id
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
                            key="kb_entry_fallback",
                        )
                        selected_entry_id = fallback_labels.get(selected_label)
                        st.session_state["kb_selected_id"] = selected_entry_id

                    if selected_entry_id:
                        st.caption(f"Click a table row to inspect (ID {selected_entry_id}).")
                        st.markdown("---")
                        selected_entry = next(
                            (entry for entry in page_entries if entry[0] == selected_entry_id),
                            None,
                        )
                        if selected_entry:
                            if selected_entry[8] is None:
                                refreshed_entry = get_kb_entry_by_id(selected_entry[0], include_markdown=True)
                                if refreshed_entry:
                                    selected_entry = refreshed_entry
                            render_kb_entry_details(selected_entry)
                else:
                    st.info("No entries available on this page.")

        except Exception as e:
            st.error(f"Failed to load entries: {e}")
        
        # ADMIN utilities
        st.subheader("Knowledge Base Administration")
        with st.expander("Admin utilities", expanded=False):
            st.caption("Maintenance helpers for operators.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button(
                    "Normalize all document tags",
                    type="secondary",
                    icon=":material/tag:",
                    key="normalize_tags_button",
                ):
                    with st.spinner("Normalizing stored tags…"):
                        try:
                            result = normalize_existing_document_tags()
                        except Exception as exc:
                            st.error(f"Tag normalization failed: {exc}")
                        else:
                            st.success(
                                f"Processed {result.get('processed', 0)} document(s); "
                                f"updated {result.get('updated', 0)} row(s).",
                                icon=":material/check_circle:",
                            )
            with col2:
                # --- Delete all entries button ---
                ids_to_delete = [entry[0] for entry in filtered]
                if not ids_to_delete:
                    st.info("No records match the current filters.")
                else:
                    if st.button(
                        f"Delete {len(ids_to_delete)} filtered record(s)",
                        type="secondary",
                        icon=":material/delete_forever:",
                    ):
                        try:
                            conn = get_connection()
                            with conn:
                                with conn.cursor() as cur:
                                    cur.execute("DELETE FROM documents WHERE id = ANY(%s)", (ids_to_delete,))
                            st.success(f"Deleted {len(ids_to_delete)} record(s) matching the current filters.")
                            mark_vector_store_dirty()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to delete filtered documents: {e}")

if authenticated:
    main()
