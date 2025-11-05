import os
import datetime
import asyncio
from pathlib import Path
import time
import base64
from typing import Any

from openai import OpenAI, AsyncOpenAI
import streamlit as st

from utils import (
    get_connection,
    admin_authentication,
    render_sidebar,
    get_document_status_counts,
    is_job_locked,
    launch_cli_job,
    CLIJob,
    CLIJobError,
    read_vector_status,
    write_vector_status,
    show_blocking_overlay,
    hide_blocking_overlay,
    render_log_output,
)

try:
    from streamlit.runtime import exists as streamlit_runtime_exists  # type: ignore
except (ImportError, ModuleNotFoundError):
    streamlit_runtime_exists = None  # type: ignore

try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore
except (ImportError, ModuleNotFoundError):
    def get_script_run_ctx():  # type: ignore
        return None

if streamlit_runtime_exists:
    try:
        if streamlit_runtime_exists():
            _script_ctx = get_script_run_ctx()
        else:
            _script_ctx = None
    except Exception:
        _script_ctx = None
else:
    _script_ctx = None

HAS_STREAMLIT_CONTEXT = _script_ctx is not None


def rerun_app() -> None:
    """Trigger a Streamlit rerun compatible with recent and legacy versions."""
    try:
        st.rerun()
    except AttributeError:
        try:
            st.experimental_rerun()
        except AttributeError:
            pass


def _get_required_secret(key: str) -> str:
    try:
        value = st.secrets[key]
    except KeyError as exc:
        raise RuntimeError(
            f"Missing Streamlit secret '{key}'. Provide it via STREAMLIT_SECRETS_JSON or secrets.toml."
        ) from exc

    if not value:
        raise RuntimeError(f"Streamlit secret '{key}' is set but empty.")

    return value

VECTORIZE_CONFIG_ERROR: str | None = None
try:
    OPENAI_API_KEY = _get_required_secret("OPENAI_API_KEY")
    VECTOR_STORE_ID = _get_required_secret("VECTOR_STORE_ID")
except RuntimeError as exc:
    VECTORIZE_CONFIG_ERROR = str(exc)
    OPENAI_API_KEY = ""
    VECTOR_STORE_ID = ""

BASE_DIR = Path(__file__).parent.parent
ICON_PATH = BASE_DIR / "assets" / "owl.png"

client = OpenAI(api_key=OPENAI_API_KEY) if not VECTORIZE_CONFIG_ERROR else None


def _get_openai_client() -> OpenAI:
    if client is None:
        raise RuntimeError(VECTORIZE_CONFIG_ERROR or "OpenAI client is not configured.")
    return client


def _create_async_openai_client() -> AsyncOpenAI:
    if VECTORIZE_CONFIG_ERROR:
        raise RuntimeError(VECTORIZE_CONFIG_ERROR)
    if not OPENAI_API_KEY:
        raise RuntimeError("OpenAI async client is not configured.")
    return AsyncOpenAI(api_key=OPENAI_API_KEY)


def _run_async_task(coro):
    """Run an async coroutine, even if called from a running event loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():  # Streamlit may already own an event loop
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()

    runner = asyncio.Runner()
    try:
        return runner.run(coro)
    finally:
        runner.close()

# Cache for vector store files (expires after 5 minutes)
_vs_files_cache = {"data": None, "timestamp": 0, "ttl": 300}

def get_cached_vector_store_files(vector_store_id: str, force_refresh: bool = False):
    """
    Cache vector store files for 5 minutes to avoid repeated expensive API calls.
    """
    current_time = time.time()
    cache = _vs_files_cache
    
    if force_refresh or cache["data"] is None or (current_time - cache["timestamp"]) > cache["ttl"]:
        print("🔄 Refreshing vector store files cache...")
        cache["data"] = list_all_files_in_vector_store(vector_store_id)
        cache["timestamp"] = current_time
        print(f"📦 Cached {len(cache['data'])} vector store files")
    else:
        print(f"⚡ Using cached vector store files ({len(cache['data'])} files)")
    
    return cache["data"]

def list_all_files_in_vector_store(vector_store_id: str):
    """
    Retrieve *all* files in the vector store by following pagination.
    """
    openai_client = _get_openai_client()
    try:
        all_files = []
        after = None
        while True:
            resp = openai_client.vector_stores.files.list(
                vector_store_id=vector_store_id,
                after=after  # page cursor
            )
            all_files.extend(resp.data)
            if not getattr(resp, "has_more", False):
                break
            after = getattr(resp, "last_id", None)
            if not after:  # safety
                break
        return all_files
    except Exception as e:
        print(f"❌ Failed to list files in vector store: {e}")
        return []
    
def compute_vector_store_sets(vector_store_id: str, use_cache: bool = True):
    """
    Returns three sets of file IDs:
      - current_ids: files referenced as vector_file_id (the live ones)
      - pending_replacement_ids: files referenced as old_file_id (to delete after successful replacement)
      - true_orphan_ids: files in VS that are not in DB at all
    """
    # 1) Vector store files (use cache for performance!)
    vs_files = get_cached_vector_store_files(vector_store_id, force_refresh=not use_cache)
    vs_ids = {f.id for f in vs_files}

    # 2) DB references
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT vector_file_id FROM documents WHERE vector_file_id IS NOT NULL")
    current_ids = {row[0] for row in cur.fetchall()}

    cur.execute("SELECT old_file_id FROM documents WHERE old_file_id IS NOT NULL")
    pending_replacement_ids = {row[0] for row in cur.fetchall()}

    cur.close()
    conn.close()

    # 3) True orphans are in VS but not referenced in DB at all
    referenced_anywhere = current_ids | pending_replacement_ids
    true_orphan_ids = vs_ids - referenced_anywhere

    return current_ids, pending_replacement_ids & vs_ids, true_orphan_ids

def read_last_vector_sync_timestamp():
    try:
        with open("last_vector_sync.txt", "r") as f:
            ts_str = f.read().strip()
            return datetime.datetime.fromisoformat(ts_str)
    except FileNotFoundError:
        return None  # no previous sync

# def count_new_unsynced_docs() -> int:
#     """
#     Count docs that were added by scraping and have never been synced:
#     vector_file_id IS NULL AND old_file_id IS NULL
#     Excludes documents marked no_upload = TRUE
#     """
#     conn = get_connection()
#     cur = conn.cursor()
#     cur.execute("""
#         SELECT COUNT(*)
#         FROM documents
#         WHERE vector_file_id IS NULL
#           AND old_file_id IS NULL
#           AND (no_upload IS FALSE OR no_upload IS NULL)
#     """)
#     (cnt,) = cur.fetchone()
#     cur.close()
#     conn.close()
#     return cnt


def list_new_unsynced_docs(limit: int = 20):
    """
    Optional: list some of the newest unsynced docs for visibility in the UI.
    Excludes documents marked no_upload = TRUE
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, title, updated_at
        FROM documents
        WHERE vector_file_id IS NULL
          AND old_file_id IS NULL
          AND (no_upload IS FALSE OR no_upload IS NULL)
        ORDER BY updated_at DESC NULLS LAST
        LIMIT %s
    """, (limit,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def delete_file_ids(vector_store_id: str, file_ids: set[str], label: str):
    if not file_ids:
        print(f"✅ No {label} to delete.")
        return
    print(f"🗑️ Deleting {len(file_ids)} {label} files...")

    async def _delete_many():
        async_openai = _create_async_openai_client()
        semaphore = asyncio.Semaphore(10)
        total = len(file_ids)
        completed = 0

        async def _delete_single(fid: str):
            nonlocal completed
            if not fid:
                return

            async with semaphore:
                try:
                    await async_openai.vector_stores.files.delete(vector_store_id=vector_store_id, file_id=fid)
                    print(f"🗑️ Detached {label} {fid}")
                except Exception as exc:
                    print(f"⚠️ Detach failed for {fid}: {exc}")

                try:
                    await async_openai.files.delete(fid)
                    print(f"🗑️ Deleted file resource {fid}")
                except Exception as exc:
                    print(f"⚠️ Delete failed for {fid}: {exc}")

            completed += 1
            if completed % 20 == 0:
                print(f"📍 Deleted {completed}/{total} {label} files...")

        try:
            await asyncio.gather(*(_delete_single(fid) for fid in file_ids))
            print(f"✅ Finished deleting {total} {label} files")
        finally:
            await async_openai.close()

    _run_async_task(_delete_many())

    # Clear cache after deletions to ensure fresh data
    _vs_files_cache["data"] = None


def clear_old_file_ids(file_ids: set[str]):
    """
    After successfully deleting pending-replacement files from the vector store,
    null out their old_file_id references in the DB.
    """
    if not file_ids:
        return
    conn = get_connection()
    cur = conn.cursor()
    # convert to list for psycopg2 ANY(%s) usage
    cur.execute(
        "UPDATE documents SET old_file_id = NULL, updated_at = NOW() WHERE old_file_id = ANY(%s)",
        (list(file_ids),)
    )
    conn.commit()
    cur.close()
    conn.close()


def delete_all_files_in_vector_store(vector_store_id: str):
    """
    Delete all files in the vector store, handling pagination and deletion.
    """
    try:
        # List all files in the vector store
        vs_files = list_all_files_in_vector_store(vector_store_id)

        if not vs_files:
            print("✅ No files to delete in the vector store.")
            return

        delete_file_ids(vector_store_id, {vs_file.id for vs_file in vs_files}, label="vector store")
        print("✅ All files deleted from the vector store.")
    except Exception as e:
        print(f"❌ Failed to list or delete files in vector store: {e}")
    
def compute_vector_store_status(vector_store_id: str, *, force_refresh: bool = False) -> dict[str, Any]:
    """Collect lightweight vector store status metrics."""

    vs_files = get_cached_vector_store_files(vector_store_id, force_refresh=force_refresh)
    vs_ids = {f.id for f in vs_files}

    current_ids, pending_replacement_ids, true_orphan_ids = compute_vector_store_sets(
        vector_store_id,
        use_cache=not force_refresh,
    )

    excluded_live_ids: set[str]
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT vector_file_id FROM documents WHERE no_upload IS TRUE AND vector_file_id IS NOT NULL"
        )
        excluded_live_ids = {row[0] for row in cur.fetchall()}
        cur.close()
        conn.close()
    except Exception:
        excluded_live_ids = set()

    live_ids_display = current_ids - excluded_live_ids
    combined_pending_cleanup_ids = (pending_replacement_ids | excluded_live_ids) & vs_ids

    return {
        "vector_store_id": vector_store_id,
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "vs_file_count": len(vs_ids),
        "live_file_count": len(live_ids_display),
        "pending_cleanup_count": len(combined_pending_cleanup_ids),
        "orphan_count": len(true_orphan_ids),
        "excluded_live_count": len(excluded_live_ids & vs_ids),
    }


def _load_vector_details(vector_store_id: str, *, force: bool = False) -> dict[str, Any]:
    """Load detailed vector store information and cache it in session_state."""

    cache_key = "vector_details"
    if not force and cache_key in st.session_state:
        return st.session_state[cache_key]

    vs_files = get_cached_vector_store_files(vector_store_id, force_refresh=force)
    vs_ids = {f.id for f in vs_files}
    current_ids, pending_replacement_ids, true_orphan_ids = compute_vector_store_sets(
        vector_store_id,
        use_cache=not force,
    )

    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT vector_file_id FROM documents WHERE no_upload IS TRUE AND vector_file_id IS NOT NULL"
        )
        excluded_live_ids = {row[0] for row in cur.fetchall()}
        cur.close()
        conn.close()
    except Exception:
        excluded_live_ids = set()

    details = {
        "loaded_at": datetime.datetime.now(datetime.UTC),
        "vs_file_count": len(vs_files),
        "vs_ids": vs_ids,
        "current_ids": current_ids,
        "pending_replacement_ids": pending_replacement_ids,
        "true_orphan_ids": true_orphan_ids,
        "excluded_live_ids": excluded_live_ids,
        "combined_pending_cleanup_ids": (pending_replacement_ids | excluded_live_ids) & vs_ids,
        "live_ids_display": current_ids - excluded_live_ids,
    }

    st.session_state[cache_key] = details
    return details

if HAS_STREAMLIT_CONTEXT:
    authenticated = admin_authentication(return_to="/pages/vectorize")
    render_sidebar(authenticated)

    def read_last_export_timestamp():
        try:
            with open("last_export.txt", "r") as f:
                ts_str = f.read().strip()
                return datetime.datetime.fromisoformat(ts_str)
        except FileNotFoundError:
            # Default to epoch start if no file found
            return datetime.datetime(1970, 1, 1)

    def write_last_export_timestamp(ts):
        with open("last_export.txt", "w") as f:
            f.write(ts.isoformat())
        

    if authenticated:
        if VECTORIZE_CONFIG_ERROR:
            st.error(VECTORIZE_CONFIG_ERROR)
            st.stop()

        # Admin-only content
        conn = get_connection()
        cursor = conn.cursor()

        last_export_timestamp = read_last_export_timestamp()

        cursor.execute("""
            SELECT id, title, markdown_content, crawl_date, lang, summary, tags, updated_at
            FROM documents
            WHERE updated_at > %s
        """, (last_export_timestamp,))

        rows = cursor.fetchall()

        output_dir = "exported_markdown"
        os.makedirs(output_dir, exist_ok=True)
            
        if ICON_PATH.exists():
            encoded_icon = base64.b64encode(ICON_PATH.read_bytes()).decode("utf-8")
            st.markdown(
                f"""
                <div style="display:flex;align-items:center;gap:.75rem;">
                    <img src="data:image/png;base64,{encoded_icon}" width="48" height="48"/>
                    <h1 style="margin:0;">Vector Store Management</h1>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.header("Vector Store Management")

    
        # Quick explanation with performance info
        with st.expander("How this works", icon=":material/info:", expanded=False):
            st.markdown("""
            **What is this?** This tool manages the synchronization between your document database and OpenAI's vector store.
        
            **The Process:**
            1. 📄 Documents are scraped and stored in your database
            2. 🔄 This tool uploads them to OpenAI's vector store for AI search
            3. 🧹 Old versions and orphaned files are cleaned up to save storage space
        
            **File Lifecycle:**
            - **New documents**: Need to be uploaded to vector store
            - **Updated documents**: Create new vector files, old ones marked for cleanup  
            - **Orphaned files**: Vector files no longer linked to any database entry
            - **Old versions**: Previous versions of updated documents that can be safely deleted
        
            **⚡ Performance Optimizations:**
            - **5-minute caching** of vector store data to avoid repeated API calls
            - **Batch processing** for uploads and deletions
            - **Progress indicators** for long-running operations
            - **Efficient filename mapping** with progress tracking
            """)
    
        st.markdown("---")
    
        # --- Enhanced Status Dashboard ---
        try:
            # Get comprehensive counts with single optimized query (now centralized in utils)
            counts = get_document_status_counts()
            total_docs = counts["total_docs"]
            vectorized_docs = counts["vectorized_docs"]
            non_vectorized_docs = counts["non_vectorized_docs"]
            new_unsynced_count = counts["new_unsynced_count"]
            pending_resync_count = counts["pending_resync_count"]
            excluded_docs = counts["excluded_docs"]
        
            # Display basic metrics dashboard (fast loading)
            st.header("Document Status")
        
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1:
                st.metric("Total Documents", total_docs, border=True)
            with col2:
                st.metric("Vectorized", vectorized_docs, border=True)
            with col3:
                st.metric("Excluded", excluded_docs, border=True)
            with col4:
                st.metric("Non-vectorized", non_vectorized_docs, border=True)
            with col5:
                st.metric("New (unsynced)", new_unsynced_count, border=True)
            with col6:
                st.metric("Need re-sync", pending_resync_count, border=True)
        
        
            # Status messages
            if new_unsynced_count > 0:
                st.warning(f"**{new_unsynced_count} newly scraped documents** need to be added to the vector store.", icon=":material/warning:")
        
            if pending_resync_count > 0:
                st.info(f"**{pending_resync_count} documents** need re-sync (content changes or exclusion cleanup).", icon=":material/cached:")
            
            if new_unsynced_count == 0 and pending_resync_count == 0:
                try:
                    last_vs_sync = read_last_vector_sync_timestamp()
                    if last_vs_sync:
                        st.success(f"**All documents are synchronized!** Last sync: {last_vs_sync.isoformat(timespec='seconds')}", icon=":material/check_circle:")
                    else:
                        st.success("**All documents are synchronized!**", icon=":material/check_circle:")
                except:
                    st.success("**All documents are synchronized!**", icon=":material/check_circle:")
        
            st.markdown("---")

            # Vector store status summary (cached + async live refresh)
            if "vector_status" not in st.session_state:
                st.session_state["vector_status"] = read_vector_status()

            snapshot_from_disk = read_vector_status()
            if snapshot_from_disk and snapshot_from_disk != st.session_state.get("vector_status"):
                st.session_state["vector_status"] = snapshot_from_disk

            status_snapshot = st.session_state.get("vector_status")

            st.header("Vector Store Status")

            if status_snapshot:
                generated_raw = status_snapshot.get("generated_at")
                try:
                    generated_at = datetime.datetime.fromisoformat(generated_raw)
                    if generated_at.tzinfo is None:
                        generated_at = generated_at.replace(tzinfo=datetime.UTC)
                    age = datetime.datetime.now(datetime.UTC) - generated_at
                    age_display = f"{int(age.total_seconds() // 60)} min" if age.total_seconds() >= 60 else f"{int(age.total_seconds())} sec"
                except Exception:
                    generated_at = None
                    age_display = "unknown"

                col_status = st.columns(4)
                col_status[0].metric("VS Files", status_snapshot.get("vs_file_count", 0), border=True)
                col_status[1].metric("Live Files", status_snapshot.get("live_file_count", 0), border=True)
                col_status[2].metric("Pending Cleanup", status_snapshot.get("pending_cleanup_count", 0), border=True)
                col_status[3].metric("Orphans", status_snapshot.get("orphan_count", 0), border=True)

                excluded_live = status_snapshot.get("excluded_live_count", 0)
                if status_snapshot.get("orphan_count", 0):
                    st.warning(
                        f"**{status_snapshot['orphan_count']} orphaned files** in vector store are no longer in the database.",
                        icon=":material/warning:",
                    )
                if status_snapshot.get("pending_cleanup_count", 0):
                    st.info(
                        f"**{status_snapshot['pending_cleanup_count']} files** pending cleanup (old versions or excluded docs).",
                        icon=":material/cached:",
                    )
                if excluded_live:
                    st.info(
                        f"**{excluded_live} excluded documents** still have vector files attached.",
                        icon=":material/info:",
                    )

                if generated_at:
                    st.caption(
                        f"Snapshot vom {generated_at.isoformat(timespec='seconds')} (Alter: {age_display})."
                    )
                    if (datetime.datetime.now(datetime.UTC) - generated_at) >= datetime.timedelta(minutes=30):
                        st.info(
                            "Snapshot is over 30 minutes old. Refresh below if required.",
                            icon=":material/info:",
                        )
            else:
                st.info(
                    "No cached vector store status available yet. It will appear after the next CLI vector sync.",
                    icon=":material/info:",
                )

            status_controls = st.columns(2)
            with status_controls[0]:
                if st.button(
                    "Refresh live status",
                    icon=":material/refresh:",
                    help="Updates the metrics directly from the vector store (this may take a few seconds).",
                ):
                    overlay = show_blocking_overlay()
                    try:
                        with st.spinner("Updating vector store status..."):
                            try:
                                status = compute_vector_store_status(VECTOR_STORE_ID, force_refresh=True)
                                write_vector_status(status)
                                st.session_state["vector_status"] = status
                                st.session_state.pop("vector_details", None)
                                st.success("Status erfolgreich aktualisiert.", icon=":material/check_circle:")
                            except Exception as exc:
                                st.error(f"Aktualisierung fehlgeschlagen: {exc}", icon=":material/error:")
                    finally:
                        hide_blocking_overlay(overlay)
            with status_controls[1]:
                if st.button(
                    "Clear cache",
                    icon=":material/delete_forever:",
                    help="Clear the in-memory vector store cache used for detailliertere Operationen.",
                ):
                    _vs_files_cache["data"] = None
                    st.session_state.pop("vector_details", None)
                    st.success("Vector store cache cleared.", icon=":material/check_circle:")

        except Exception as e:
            st.error(f"Failed to compute sync status: {e}", icon=":material/error:")
            # Set defaults on error
            new_unsynced_count = 0
            pending_resync_count = 0
            vs_file_count = 0
            current_ids = set()
            pending_replacement_ids = set()
            true_orphan_ids = set()
            excluded_docs = 0
        
            st.markdown("---")
        
            # Status messages based on counts
            if new_unsynced_count > 0:
                st.warning(f"**{new_unsynced_count} newly scraped documents** need to be added to the vector store.", icon=":material/warning:")
        
            if pending_resync_count > 0:
                st.info(f"**{pending_resync_count} documents** need re-sync (content changes or exclusion cleanup).", icon=":material/cached:")
            
            if len(true_orphan_ids) > 0:
                st.warning(f"**{len(true_orphan_ids)} orphaned files** in vector store are no longer in the database.", icon=":material/cleaning_service:")
            
            if len(pending_replacement_ids) > 0:
                st.info(f"**{len(pending_replacement_ids)} old files** can be cleaned up after successful replacements.", icon=":material/cached:")
            
            if new_unsynced_count == 0 and pending_resync_count == 0:
                last_vs_sync = read_last_vector_sync_timestamp()
                if last_vs_sync:
                    st.success(f"**All documents are synchronized!** Last sync: {last_vs_sync.isoformat(timespec='seconds')}", icon=":material/check_circle:")
                else:
                    st.success("**All documents are synchronized!**", icon=":material/check_circle:")
                
        except Exception as e:
            st.error(f"Failed to compute sync status: {e}", icon=":material/error:")

        # --- Detailed preview of unsynced docs ---
        if new_unsynced_count > 0:
            with st.expander(f"Preview of {min(10, new_unsynced_count)} newest unsynced documents", icon=":material/preview:"):
                try:
                    rows_preview = list_new_unsynced_docs(limit=10)
                    for _id, _title, _updated_at in rows_preview:
                        st.write(f"- **ID {_id}**: {_title or '(untitled)'} · Updated: {str(_updated_at) if _updated_at else 'n/a'}")
                except Exception as e:
                    st.error(f"Failed to load preview: {e}")

        st.markdown("---")
        st.header("Vector Store File Management")

        st.subheader("Manual Sync (override)")
        st.info(
            "Scheduled scrapes already trigger vector syncs. Use the manual controls below for urgent overrides.",
            icon=":material/schedule:",
        )

        with st.expander("Show manual vector sync controls", expanded=False):
            st.caption("Uploads new or changed knowledge base entries to vector store and deletes files marked as excluded from vector store.")
            
            # Sync button with better context
            sync_button_text = "Sync Documents with Vector Store"
            if new_unsynced_count > 0 or pending_resync_count > 0:
                sync_button_text += f" ({new_unsynced_count + pending_resync_count} pending)"
        
            if "vector_cli_job" not in st.session_state:
                st.session_state["vector_cli_job"] = None
            if "vector_cli_message" not in st.session_state:
                st.session_state["vector_cli_message"] = None
            if "vector_cli_last_return" not in st.session_state:
                st.session_state["vector_cli_last_return"] = None

            vector_job: CLIJob | None = st.session_state.get("vector_cli_job")
            job_running = vector_job is not None and vector_job.is_running()

            vector_lock_error = None
            external_job_running = False
            try:
                lock_held = is_job_locked("scrape_job")
                external_job_running = lock_held and not job_running
            except RuntimeError as exc:
                vector_lock_error = str(exc)

            if vector_lock_error:
                st.warning(f"Could not verify job status: {vector_lock_error}", icon=":material/warning:")
            elif external_job_running:
                st.warning(
                    "A scrape/vectorize job is already running (e.g., via the scheduler). "
                    "Wait for it to finish before starting another vector sync.",
                    icon=":material/pending_actions:",
                )

            sync_disabled = job_running or external_job_running

            if st.button(sync_button_text, type="primary", icon=":material/sync:", disabled=sync_disabled):
                if job_running:
                    st.warning("A vector sync job is already running.", icon=":material/warning:")
                elif external_job_running:
                    st.warning("Another scrape/vectorize job is active. Try again once it finishes.", icon=":material/pending_actions:")
                else:
                    overlay = show_blocking_overlay()
                    rerun_needed = False
                    try:
                        job = launch_cli_job(mode="vectorize")
                    except CLIJobError as exc:
                        st.error(f"Failed to start vector sync job: {exc}", icon=":material/error:")
                    else:
                        st.session_state["vector_cli_job"] = job
                        st.session_state["vector_cli_message"] = "Vector sync running via cli_scrape.py."
                        st.session_state["vector_cli_last_return"] = None
                        rerun_needed = True
                    finally:
                        hide_blocking_overlay(overlay)
                    if rerun_needed:
                        rerun_app()

            if st.session_state["vector_cli_message"]:
                st.info(st.session_state["vector_cli_message"], icon=":material/center_focus_weak:")

            if vector_job:
                st.markdown("#### CLI Sync Log")
                log_lines = "\n".join(list(vector_job.logs))

                render_log_output(log_lines, element_id="vector-sync-log")

                if vector_job.is_running():
                    st.info(
                        f"Vector sync in progress (PID {vector_job.process.pid}). Use the controls below to refresh or cancel.",
                        icon=":material/progress_activity:",
                    )
                    refresh_col, cancel_col = st.columns(2)
                    with refresh_col:
                        st.button("Refresh status", key="refresh_vector_job")
                    with cancel_col:
                        if st.button("Cancel job", key="cancel_vector_job"):
                            vector_job.terminate()
                            st.session_state["vector_cli_message"] = "Cancellation requested. Check the log for confirmation."
                            rerun_app()
                else:
                    if st.session_state["vector_cli_last_return"] is None:
                        st.session_state["vector_cli_last_return"] = vector_job.returncode()
                    exit_code = st.session_state["vector_cli_last_return"] or 0
                    if exit_code == 0:
                        st.session_state["vector_cli_message"] = None
                        _vs_files_cache["data"] = None
                        st.session_state.pop("vector_details", None)
                        st.success("Vector sync completed successfully.", icon=":material/check_circle:")
                    else:
                        st.session_state["vector_cli_message"] = None
                        st.error(f"Vector sync finished with exit code {exit_code}.", icon=":material/error:")

                    st.button("Refresh status", key="refresh_vector_job_done")
                    if st.button("Clear log", key="clear_vector_job"):
                        st.session_state["vector_cli_job"] = None
                        st.session_state["vector_cli_message"] = None
                        st.session_state["vector_cli_last_return"] = None

        st.markdown("---")
        st.subheader("Manage Vector Store Files")
        vector_details = st.session_state.get("vector_details")

        load_label = "Reload vector store details" if vector_details else "Load vector store details"
        load_icon = ":material/cached:" if vector_details else ":material/database:"

        if st.button(
            load_label,
            icon=load_icon,
            help="Fetch vector-store file information for the cleanup actions below.",
        ):
            overlay = show_blocking_overlay()
            try:
                with st.spinner("Loading Vector Store data..."):
                    try:
                        vector_details = _load_vector_details(VECTOR_STORE_ID, force=True)
                        st.success("Details updated." if load_label.startswith("Reload") else "Details loaded.", icon=":material/check_circle:")
                    except Exception as exc:
                        st.session_state.pop("vector_details", None)
                        st.error(f"Failed to load: {exc}", icon=":material/error:")
                        vector_details = None
            finally:
                hide_blocking_overlay(overlay)

        vector_details = st.session_state.get("vector_details")

        vector_details = st.session_state.get("vector_details")
        #st.markdown(vector_details)

        if not vector_details:
            st.info("Load vector store data to execute cleanup operations.", icon=":material/info:")
        else:
            st.markdown("#### Vector Store Details")
            loaded_at = vector_details.get("loaded_at")
            if loaded_at:
                if loaded_at.tzinfo is None:
                    loaded_at = loaded_at.replace(tzinfo=datetime.UTC)
                age_minutes = int((datetime.datetime.now(datetime.UTC) - loaded_at).total_seconds() // 60)
                st.caption(f"Details loaded at {loaded_at.isoformat(timespec='seconds')} (Age: {age_minutes} min).")

            details_col1, details_col2, details_col3 = st.columns(3)
            details_col1.metric("Vector store files", vector_details["vs_file_count"], border=True)
            details_col2.metric("Live KB entries", len(vector_details["current_ids"]), border=True)
            details_col3.metric("Pending replacements", len(vector_details["pending_replacement_ids"]), border=True)
            vs_file_count = vector_details["vs_file_count"]
            current_ids = vector_details["current_ids"]
            pending_replacement_ids = vector_details["pending_replacement_ids"]
            true_orphan_ids = vector_details["true_orphan_ids"]
            excluded_live_ids = vector_details["excluded_live_ids"] & vector_details["vs_ids"]
            combined_pending_cleanup_ids = vector_details["combined_pending_cleanup_ids"]
            live_ids_display = vector_details["live_ids_display"]

            st.markdown("#### Clean Excluded Now")
            st.caption("Only deletes vector store files for documents marked as excluded and clears DB references.")

            excluded_live_count = len(excluded_live_ids)
            if excluded_live_count > 0:
                if st.button(
                    f"Clean Excluded Files ({excluded_live_count})",
                    key="clean_excluded_now",
                    icon=":material/mop:",
                ):
                    overlay = show_blocking_overlay()
                    try:
                        with st.spinner("Cleaning excluded files..."):
                            try:
                                conn = get_connection(); cur = conn.cursor()
                                cur.execute(
                                    "SELECT id, vector_file_id FROM documents WHERE no_upload IS TRUE AND vector_file_id IS NOT NULL"
                                )
                                rows = cur.fetchall()
                                excluded_file_ids = {r[1] for r in rows if r[1]}
                                excluded_ids = [r[0] for r in rows]
                                cur.close()

                                if excluded_file_ids:
                                    delete_file_ids(VECTOR_STORE_ID, excluded_file_ids, label="excluded")

                                if excluded_ids:
                                    cur2 = conn.cursor()
                                    cur2.execute(
                                        """
                                        UPDATE documents
                                        SET vector_file_id = NULL,
                                            old_file_id = NULL,
                                            updated_at = NOW()
                                        WHERE id = ANY(%s)
                                        """,
                                        (excluded_ids,)
                                    )
                                    conn.commit(); cur2.close()
                                conn.close()

                                _vs_files_cache["data"] = None
                                st.session_state.pop("vector_details", None)
                                st.success("Cleaned excluded files and cleared references.", icon=":material/check_circle:")
                            except Exception as e:
                                st.error(f"Failed to clean excluded files: {e}", icon=":material/error:")
                    finally:
                        hide_blocking_overlay(overlay)
            else:
                st.caption("No excluded vector files to clean.")

            if vs_file_count > 0:
                st.markdown("#### Vector Store Cleanup")
                st.caption("Clean up unnecessary files to save storage space and keep the vector store organized.")

                col1, col2 = st.columns(2)

                with col1:
                    if len(true_orphan_ids) > 0:
                        st.markdown("**Remove Orphaned Files**")
                        st.warning(f"Found {len(true_orphan_ids)} files in vector store that no longer exist in the database", icon=":material/warning:")
                        st.caption("Safe to delete - these files are no longer referenced by any documents")
                        if st.button("Delete Orphans", key="delete_orphans", icon=":material/delete_sweep:"):
                            overlay = show_blocking_overlay()
                            try:
                                with st.spinner("Deleting orphans..."):
                                    try:
                                        delete_file_ids(VECTOR_STORE_ID, true_orphan_ids, label="orphan")
                                        _vs_files_cache["data"] = None
                                        st.session_state.pop("vector_details", None)
                                        st.success("Deleted all orphaned files from the vector store.", icon=":material/check_circle:")
                                        rerun_app()
                                    except Exception as e:
                                        st.error(f"Failed to delete orphans: {e}", icon=":material/error:")
                            finally:
                                hide_blocking_overlay(overlay)
                    else:
                        st.markdown("**Remove Orphaned Files**")
                        st.success("No orphaned files found", icon=":material/check_circle:")
                        st.caption("All vector store files are properly referenced in the database")

                with col2:
                    if len(pending_replacement_ids) > 0:
                        st.markdown("**Clean Up Old Versions**")
                        st.warning(f"Found {len(pending_replacement_ids)} old file versions after content updates", icon=":material/warning:")
                        st.caption("Safe to delete - these are old versions that have been replaced with updated content")
                        if st.button("Clean Up Old Versions", key="finalize_replacements", icon=":material/restore_page:"):
                            overlay = show_blocking_overlay()
                            try:
                                with st.spinner("Cleaning up old versions..."):
                                    try:
                                        delete_file_ids(VECTOR_STORE_ID, pending_replacement_ids, label="old version")
                                        clear_old_file_ids(pending_replacement_ids)
                                        _vs_files_cache["data"] = None
                                        st.session_state.pop("vector_details", None)
                                        st.success("Cleaned up old file versions and updated database references.", icon=":material/check_circle:")
                                        rerun_app()
                                    except Exception as e:
                                        st.error(f"Failed to clean up old versions: {e}", icon=":material/error:")
                            finally:
                                hide_blocking_overlay(overlay)
                    else:
                        st.markdown("**Clean Up Old Versions**")
                        st.success("No old versions to clean up", icon=":material/check_circle:")
                        st.caption("All file replacements have been properly finalized")

                st.markdown("#### Nuclear Option")
                st.error("**DANGER**: This will permanently delete ALL files in the vector store!", icon=":material/warning:")
                st.caption("Only use this if you want to completely rebuild the vector store from scratch.")

                col_danger1, col_danger2, col_danger3 = st.columns([1, 1, 2])
                with col_danger2:
                    if st.button(
                        "Delete Everything",
                        type="secondary",
                        help="This will permanently delete all files in the vector store",
                        icon=":material/delete_forever:",
                    ):
                        overlay = show_blocking_overlay()
                        try:
                            with st.spinner("Deleting all files..."):
                                try:
                                    delete_all_files_in_vector_store(VECTOR_STORE_ID)
                                    _vs_files_cache["data"] = None
                                    st.session_state.pop("vector_details", None)
                                    st.success("All files deleted from the vector store.", icon=":material/check_circle:")
                                    rerun_app()
                                except Exception as e:
                                    st.error(f"Failed to delete files: {e}", icon=":material/error:")
                        finally:
                            hide_blocking_overlay(overlay)
            else:
                st.info("Vector store is empty - no files to manage.", icon=":material/info:")

    else:
        st.warning("Authentication required to access the Vector Store Management.", icon=":material/lock:")
