import os
import datetime
import asyncio
from pathlib import Path
import time
from io import BytesIO
import base64
import threading
from typing import Any

from openai import OpenAI, AsyncOpenAI
import streamlit as st
from utils import (
    get_connection,
    admin_authentication,
    render_sidebar,
    get_document_status_counts,
    launch_cli_job,
    CLIJob,
    CLIJobError,
    read_vector_status,
    write_vector_status,
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

try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx as _st_add_ctx  # type: ignore
except (ImportError, ModuleNotFoundError):
    _st_add_ctx = None  # type: ignore

_script_ctx = get_script_run_ctx()
if streamlit_runtime_exists:
    HAS_STREAMLIT_CONTEXT = streamlit_runtime_exists() and _script_ctx is not None
else:
    HAS_STREAMLIT_CONTEXT = _script_ctx is not None


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


def _start_thread(target, *, name: str, args: tuple = ()):  # noqa: N802
    """Start a daemon thread and attach Streamlit context when available."""
    t = threading.Thread(target=target, args=args, daemon=True, name=name)
    try:
        if _st_add_ctx:
            _st_add_ctx(t)
    except Exception:
        pass
    t.start()
    return t


def _safe_rerun() -> None:
    """Trigger a Streamlit rerun when supported."""
    try:
        st.rerun()
    except AttributeError:
        try:
            st.experimental_rerun()  # type: ignore[attr-defined]
        except AttributeError:
            pass

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

def write_last_vector_sync_timestamp(ts: datetime.datetime | None = None):
    if ts is None:
        ts = datetime.datetime.now()
    with open("last_vector_sync.txt", "w") as f:
        f.write(ts.isoformat())


def count_new_unsynced_docs() -> int:
    """
    Count docs that were added by scraping and have never been synced:
    vector_file_id IS NULL AND old_file_id IS NULL
    Excludes documents marked no_upload = TRUE
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(*)
        FROM documents
        WHERE vector_file_id IS NULL
          AND old_file_id IS NULL
          AND (no_upload IS FALSE OR no_upload IS NULL)
    """)
    (cnt,) = cur.fetchone()
    cur.close()
    conn.close()
    return cnt


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

        
def find_missing_file_ids(vector_store_id: str):
    """
    Find file IDs in the vector store that are not present in the local database.
    """
    try:
        vs_files = list_all_files_in_vector_store(vector_store_id)
        vector_store_file_ids = {vs_file.id for vs_file in vs_files}
        print(f"📄 Found {len(vector_store_file_ids)} files in the vector store.")

        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT vector_file_id FROM documents WHERE vector_file_id IS NOT NULL")
        db_file_ids = {row[0] for row in cursor.fetchall()}
        print(f"📄 Found {len(db_file_ids)} files in the local database.")

        missing_file_ids = vector_store_file_ids - db_file_ids
        print(f"📂 Missing file IDs: {missing_file_ids}")

        cursor.close()
        conn.close()
        return missing_file_ids

    except Exception as e:
        print(f"❌ Failed to find missing file IDs: {e}")
        return set()

def delete_missing_files_from_vector_store(vector_store_id: str, missing_file_ids: set[str]):
    """
    Delete only the files that exist in the vector store but are missing from the DB.
    """
    if not missing_file_ids:
        print("✅ No missing files to delete.")
        return

    delete_file_ids(vector_store_id, set(missing_file_ids), label="missing")
    print("✅ Finished deleting non-mirrored files.")


def _fetch_missing_file_ids_task(vector_store_id: str):
    """Background task to compute missing file IDs without blocking the UI."""
    try:
        missing_ids = find_missing_file_ids(vector_store_id)
        st.session_state["missing_file_ids"] = sorted(missing_ids)
        st.session_state["missing_file_error"] = None
    except Exception as exc:  # pragma: no cover - network dependent
        st.session_state["missing_file_ids"] = []
        st.session_state["missing_file_error"] = str(exc)
    finally:
        st.session_state["missing_file_fetching"] = False
        st.session_state["missing_file_completed_at"] = datetime.datetime.now()
        try:
            st.rerun()
        except Exception:
            pass


def _fetch_vector_status_task(vector_store_id: str, force_refresh: bool):
    """Background task to compute vector status asynchronously."""
    try:
        status = compute_vector_store_status(vector_store_id, force_refresh=force_refresh)
        write_vector_status(status)
        st.session_state["vector_status"] = status
        st.session_state["vector_status_error"] = None
    except Exception as exc:  # pragma: no cover - network dependent
        st.session_state["vector_status_error"] = str(exc)
    finally:
        st.session_state["vector_status_loading"] = False
        try:
            _safe_rerun()
        except Exception:
            pass


def upload_md_to_vector_store(safe_title: str, content: str, vector_store_id: str) -> str | None:
    file_stream = BytesIO(content.encode("utf-8"))
    file_name = f"{safe_title}.md"

    openai_client = _get_openai_client()

    try:
        batch = openai_client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store_id,
            files=[(file_name, file_stream)]
        )
        print(f"✅ Upload for doc {safe_title} complete. Status: {batch.status}")
        print("📄 File counts:", batch.file_counts)

        # Look through *all* files to find the new one by filename (async metadata fetch)
        vs_files = list_all_files_in_vector_store(vector_store_id)
        filename_map = build_filename_to_id_map_efficiently(vs_files)
        file_id = filename_map.get(file_name)
        if file_id:
            print(f"📂 Upload successful, found vector file ID: {file_id} for {file_name}")
            return file_id

        print(f"⚠️ No matching vector file found for {file_name}")
        return None

    except Exception as e:
        print(f"❌ Upload failed for doc {safe_title}:", e)
        return None
    
def chunked(iterable, size):
    """Yield successive chunks of length `size` from a list."""
    for i in range(0, len(iterable), size):
        yield iterable[i:i+size]
    
    
def build_filename_to_id_map_efficiently(vs_files):
    """
    Build filename to ID mapping more efficiently by batching API calls.
    """
    print(f"🔍 Mapping {len(vs_files)} vector store files to filenames...")

    async def _build_map():
        async_openai = _create_async_openai_client()
        semaphore = asyncio.Semaphore(15)
        filename_to_id: dict[str, str] = {}
        failed_retrievals = 0
        processed = 0
        total = len(vs_files)
        status_lock = asyncio.Lock()

        async def _fetch_metadata(vs_file):
            nonlocal failed_retrievals, processed
            try:
                async with semaphore:
                    file_obj = await async_openai.files.retrieve(vs_file.id)
            except Exception as exc:
                async with status_lock:
                    failed_retrievals += 1
                print(f"⚠️ Failed to retrieve file object for {vs_file.id}: {exc}")
                return

            async with status_lock:
                filename_to_id[file_obj.filename] = vs_file.id
                processed += 1
                if processed % 50 == 0:
                    print(f"📍 Processed {processed}/{total} files...")

        try:
            await asyncio.gather(*(_fetch_metadata(vs_file) for vs_file in vs_files))
        finally:
            await async_openai.close()

        if failed_retrievals > 0:
            print(f"⚠️ {failed_retrievals} file retrievals failed")

        print(f"✅ Successfully mapped {len(filename_to_id)} filenames")
        return filename_to_id

    return _run_async_task(_build_map())


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
        "generated_at": datetime.datetime.utcnow().isoformat(),
        "vs_file_count": len(vs_ids),
        "live_file_count": len(live_ids_display),
        "pending_cleanup_count": len(combined_pending_cleanup_ids),
        "orphan_count": len(true_orphan_ids),
        "excluded_live_count": len(excluded_live_ids & vs_ids),
    }

def sync_vector_store():
    conn = get_connection()
    cur = conn.cursor()
    openai_client = _get_openai_client()

    cur.execute("""
        SELECT id, url, title, safe_title, crawl_date, lang, summary, tags,
               markdown_content, vector_file_id, old_file_id, no_upload
        FROM documents
    """)

    print("🔁 Starting sync for documents...")

    docs_to_upload = []
    id_to_old_file = {}
    excluded_doc_ids = []
    excluded_file_ids = set()

    for row in cur:
        doc_id, url, title, safe_title, crawl_date, lang, summary, tags, markdown, file_id, old_file_id, no_upload = row
        
        # Always ensure safe_title fallback
        safe_title = safe_title or (title or "untitled")
        
        # If document is excluded from vector store
        if no_upload:
            if file_id:
                # Schedule deletion of current vector file
                excluded_doc_ids.append(doc_id)
                excluded_file_ids.add(file_id)
                print(f"🚫 Excluding doc {safe_title} (ID {doc_id}) -> will delete vector file {file_id}")
            else:
                if old_file_id:
                    # No live vector file; clear any lingering old_file_id
                    id_to_old_file[doc_id] = old_file_id
                    print(f"🧹 Excluded doc {safe_title} has old_file_id {old_file_id} scheduled to clear")
            continue  # Skip upload logic entirely
        
        # Include in sync workflow (no_upload is False)
        content = f"""---
title: "{title}"
url: "{url}"
summary: "{summary or ''}"
tags: {tags or []}
crawl_date: "{crawl_date.isoformat() if crawl_date else ''}"
language: "{lang or ''}"
safe_title: "{safe_title or ''}" 
---

{markdown}
"""
        
        if file_id:
            print(f"⏩ Skipping unchanged doc {safe_title} -> {file_id}")
            continue
        else:
            if old_file_id:  # only track real IDs
                id_to_old_file[doc_id] = old_file_id
                print(f"Content change detected for {safe_title}, will delete old file -> {old_file_id}")
            else:
                print(f"ℹ️ No old_file_id for {safe_title} (first-time upload or already cleaned).")

        file_name = f"{doc_id}_{safe_title}.md"
        docs_to_upload.append((doc_id, file_name, content))


    # First, handle deletions for excluded documents (and clear DB refs)
    if excluded_file_ids:
        print(f"🗑️ Deleting {len(excluded_file_ids)} excluded vector files...")
        delete_file_ids(VECTOR_STORE_ID, excluded_file_ids, label="excluded")
    
    if excluded_doc_ids:
        # Clear DB references for excluded docs
        cur2 = conn.cursor()
        cur2.execute(
            """
            UPDATE documents
            SET vector_file_id = NULL,
                old_file_id = NULL,
                updated_at = NOW()
            WHERE id = ANY(%s)
            """,
            (excluded_doc_ids,)
        )
        conn.commit()
        cur2.close()
        print(f"✅ Cleared DB references for {len(excluded_doc_ids)} excluded docs")

    if not docs_to_upload:
        print("✅ No changes to sync.")
        cur.close()
        conn.close()
        return

    print(f"📤 Will upload {len(docs_to_upload)} documents")

    # Batch delete old files for better performance
    old_file_ids = [fid for fid in id_to_old_file.values() if fid]
    if old_file_ids:
        delete_file_ids(VECTOR_STORE_ID, set(old_file_ids), label="old")

    # Prepare files for upload
    upload_files = [
        (file_name, BytesIO(content.encode("utf-8")))
        for (_, file_name, content) in docs_to_upload
    ]

    print(f"📤 Starting batch upload of {len(upload_files)} files...")
    
    try:
        for idx, files_chunk in enumerate(chunked(upload_files, 100), start=1):
            print(f"📦 Uploading batch {idx} ({len(files_chunk)} files)...")
            batch = openai_client.vector_stores.file_batches.upload_and_poll(
                vector_store_id=VECTOR_STORE_ID,
                files=files_chunk
            )
            print(f"✅ Batch {idx} upload complete. Status: {batch.status}")
            print("📄 File counts:", batch.file_counts)
    except Exception as e:
        print("❌ Batch upload failed:", e)
        cur.close()
        conn.close()
        return
    
    print("🔄 Refreshing vector store cache after upload...")
    # Force refresh cache after upload
    vs_files = get_cached_vector_store_files(VECTOR_STORE_ID, force_refresh=True)

    # Build filename mapping more efficiently  
    filename_to_id = build_filename_to_id_map_efficiently(vs_files)

    # Update the database with new file IDs
    print(f"💾 Updating database for {len(docs_to_upload)} documents...")
    for doc_id, file_name, _ in docs_to_upload:
        file_id = filename_to_id.get(file_name)
        if not file_id:
            print(f"❌ Could not find file ID for {file_name}, skipping DB update.")
            continue

        cur.execute("""
            UPDATE documents
            SET vector_file_id = %s,
                old_file_id = NULL,  -- Reset old_file_id since we are updating
                updated_at = NOW()
            WHERE id = %s
        """, (file_id, doc_id))
        print(f"✅ Synced doc {file_name} → file {file_id}")

    conn.commit()
    cur.close()
    conn.close()
    print("🎉 Sync complete.")
    
    # Mark the time of the successful vector sync
    write_last_vector_sync_timestamp(datetime.datetime.now())

    
    # Return summary info for Streamlit
    return {
        "uploaded_count": len(docs_to_upload),
        "synced_files": [file_name for (_, file_name, _) in docs_to_upload]
    }


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
            if "vector_status_loading" not in st.session_state:
                st.session_state["vector_status_loading"] = False
            if "vector_status_error" not in st.session_state:
                st.session_state["vector_status_error"] = None

            snapshot_from_disk = read_vector_status()
            if snapshot_from_disk and snapshot_from_disk != st.session_state.get("vector_status"):
                st.session_state["vector_status"] = snapshot_from_disk

            status_snapshot = st.session_state["vector_status"]
            status_error = st.session_state.get("vector_status_error")
            status_loading = st.session_state.get("vector_status_loading", False)

            if not status_loading:
                should_refresh_status = False
                if status_snapshot is None:
                    should_refresh_status = True
                else:
                    generated_raw = status_snapshot.get("generated_at")
                    try:
                        generated_at = datetime.datetime.fromisoformat(generated_raw)
                    except Exception:
                        should_refresh_status = True
                    else:
                        if datetime.datetime.utcnow() - generated_at >= datetime.timedelta(minutes=30):
                            should_refresh_status = True
                if should_refresh_status:
                    st.session_state["vector_status_loading"] = True
                    _start_thread(
                        _fetch_vector_status_task,
                        name="fetch_vector_status",
                        args=(VECTOR_STORE_ID, True),
                    )

            st.header("Vector Store Status")

            if status_snapshot:
                generated_raw = status_snapshot.get("generated_at")
                try:
                    generated_at = datetime.datetime.fromisoformat(generated_raw)
                    age = datetime.datetime.utcnow() - generated_at
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
                        f"🧹 **{status_snapshot['orphan_count']} orphaned files** in vector store are no longer in the database.",
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
                        f"Snapshot generated at {generated_at.isoformat(timespec='seconds')} (age: {age_display})."
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
                    disabled=status_loading,
                    help="Fetch the latest metrics directly from the vector store without blocking the UI.",
                ):
                    st.session_state["vector_status_loading"] = True
                    st.session_state["vector_status_error"] = None
                    _start_thread(
                        _fetch_vector_status_task,
                        name="refresh_vector_status_manual",
                        args=(VECTOR_STORE_ID, True),
                    )
            with status_controls[1]:
                if st.button(
                    "Clear cache",
                    icon=":material/delete_forever:",
                    help="Clear the in-memory vector store cache used for detailed operations.",
                ):
                    _vs_files_cache["data"] = None
                    st.success("Vector store cache cleared.", icon=":material/check_circle:")

            if status_loading:
                st.info("Refreshing vector store status...", icon=":material/progress_activity:")
            if status_error := st.session_state.get("vector_status_error"):
                st.error(f"Failed to update vector store status: {status_error}", icon=":material/error:")

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
                st.warning(f"🧹 **{len(true_orphan_ids)} orphaned files** in vector store are no longer in the database.", icon=":material/warning:")
            
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

        if "missing_file_fetching" not in st.session_state:
            st.session_state["missing_file_fetching"] = False
            st.session_state["missing_file_ids"] = None
            st.session_state["missing_file_error"] = None
            st.session_state["missing_file_completed_at"] = None
            st.session_state["missing_file_last_start"] = None

        if "missing_file_last_start" not in st.session_state:
            st.session_state["missing_file_last_start"] = None

        # Auto-trigger a background check at most every 30 minutes
        now = datetime.datetime.now()
        auto_interval = datetime.timedelta(minutes=30)
        last_start = st.session_state.get("missing_file_last_start")
        completed_at = st.session_state.get("missing_file_completed_at")
        reference_time = completed_at or last_start

        should_auto_run = False
        if not st.session_state["missing_file_fetching"]:
            if reference_time is None:
                should_auto_run = True
            elif now - reference_time >= auto_interval:
                should_auto_run = True

        if should_auto_run:
            st.session_state["missing_file_fetching"] = True
            st.session_state["missing_file_ids"] = None
            st.session_state["missing_file_error"] = None
            st.session_state["missing_file_last_start"] = now
            _start_thread(
                _fetch_missing_file_ids_task,
                name="fetch_missing_vector_files",
                args=(VECTOR_STORE_ID,),
            )

        if st.session_state["missing_file_fetching"]:
            st.info("Scanning vector store for missing files...", icon=":material/progress_activity:")
        else:
            error_msg = st.session_state.get("missing_file_error")
            result = st.session_state.get("missing_file_ids")
            completed_at = st.session_state.get("missing_file_completed_at")
            if error_msg:
                st.error(f"Failed to check missing files: {error_msg}", icon=":material/error:")
            elif result is not None:
                if result:
                    st.warning(
                        f"Found {len(result)} vector store files that are missing in the database.",
                        icon=":material/warning:",
                    )
                    st.code("\n".join(result))
                else:
                    st.success("No missing vector store files detected.", icon=":material/task_alt:")
                if completed_at:
                    st.caption(
                        f"Last checked: {completed_at.isoformat(sep=' ', timespec='seconds')}"
                    )


        # Action: Sync now
        st.markdown("#### Sync Knowlede Base Now")
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

        if st.button(sync_button_text, type="primary", icon=":material/sync:", disabled=job_running):
            if job_running:
                st.warning("A vector sync job is already running.", icon=":material/warning:")
            else:
                try:
                    job = launch_cli_job(mode="vectorize")
                except CLIJobError as exc:
                    st.error(f"Failed to start vector sync job: {exc}", icon=":material/error:")
                else:
                    st.session_state["vector_cli_job"] = job
                    st.session_state["vector_cli_message"] = "Vector sync running via cli_scrape.py."
                    st.session_state["vector_cli_last_return"] = None
                    _safe_rerun()

        if st.session_state["vector_cli_message"]:
            st.info(st.session_state["vector_cli_message"], icon=":material/center_focus_weak:")

        if vector_job:
            st.markdown("### 📟 CLI Sync Log")
            log_lines = "\n".join(list(vector_job.logs))
            st.text_area(
                "Vector sync log",
                value=log_lines or "Waiting for output...",
                height=320,
                disabled=True,
                label_visibility="collapsed",
            )

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
                        _safe_rerun()
            else:
                if st.session_state["vector_cli_last_return"] is None:
                    st.session_state["vector_cli_last_return"] = vector_job.returncode()
                exit_code = st.session_state["vector_cli_last_return"] or 0
                if exit_code == 0:
                    st.session_state["vector_cli_message"] = None
                    _vs_files_cache["data"] = None
                    st.success("Vector sync completed successfully.", icon=":material/check_circle:")
                else:
                    st.session_state["vector_cli_message"] = None
                    st.error(f"Vector sync finished with exit code {exit_code}.", icon=":material/error:")

                st.button("Refresh status", key="refresh_vector_job_done")
                if st.button("Clear log", key="clear_vector_job"):
                    st.session_state["vector_cli_job"] = None
                    st.session_state["vector_cli_message"] = None
                    st.session_state["vector_cli_last_return"] = None
                    _safe_rerun()


        # Always compute vector store data for file management (this section needs accurate data)
        try:
            # If we didn't load vector store details above, load them now for file management
            if 'vs_file_count' not in locals() or vs_file_count == 0:
                vs_files = get_cached_vector_store_files(VECTOR_STORE_ID)
                vs_file_count = len(vs_files)
                current_ids, pending_replacement_ids, true_orphan_ids = compute_vector_store_sets(VECTOR_STORE_ID, use_cache=True)
        except Exception as e:
            st.error(f"Failed to load vector store data: {e}", icon=":material/error:")
            vs_file_count = 0
            current_ids = set()
            pending_replacement_ids = set()
            true_orphan_ids = set()
            
        # Action: clean excluded now
        st.markdown("#### Clean Excluded Now")
        st.caption("Only deletes vector store files for documents marked as excluded and clears DB references.")
        # Count excluded docs that still have vector files
        try:
            conn = get_connection(); cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM documents WHERE no_upload IS TRUE AND vector_file_id IS NOT NULL")
            (excluded_live_count,) = cur.fetchone()
            cur.close(); conn.close()
        except Exception:
            excluded_live_count = 0

        if excluded_live_count > 0:
            if st.button(f"Clean Excluded Files ({excluded_live_count})", key="clean_excluded_now", icon=":material/mop:"):
                with st.spinner("Cleaning excluded files..."):
                    try:
                        conn = get_connection(); cur = conn.cursor()
                        cur.execute("SELECT id, vector_file_id FROM documents WHERE no_upload IS TRUE AND vector_file_id IS NOT NULL")
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
                    
                        # Clear cache and refresh page
                        _vs_files_cache["data"] = None
                        st.success("Cleaned excluded files and cleared references.", icon=":material/check_circle:")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to clean excluded files: {e}", icon=":material/error:")
        else:
            st.caption("No excluded vector files to clean.")

        # Show cleanup options only if there are files to manage
        if vs_file_count > 0:
            st.markdown("#### Vector Store Cleanup")
            st.caption("Clean up unnecessary files to save storage space and keep the vector store organized.")
        
            # Row 1: Safe cleanup operations
            col1, col2 = st.columns(2)
        
            with col1:
                if len(true_orphan_ids) > 0:
                    st.markdown("**Remove Orphaned Files**")
                    st.write(f"Found {len(true_orphan_ids)} files in vector store that no longer exist in the database")
                    st.caption("Safe to delete - these files are no longer referenced by any documents")
                    if st.button("Delete Orphans", key="delete_orphans", icon=":material/delete_sweep:"):
                        with st.spinner("Deleting orphans..."):
                            try:
                                delete_file_ids(VECTOR_STORE_ID, true_orphan_ids, label="orphan")
                                st.success("Deleted all orphaned files from the vector store.", icon=":material/check_circle:")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to delete orphans: {e}", icon=":material/error:")
                else:
                    st.markdown("**Remove Orphaned Files**")
                    st.write("✅ No orphaned files found")
                    st.caption("All vector store files are properly referenced in the database")
        
            with col2:
                if len(pending_replacement_ids) > 0:
                    st.markdown("**Clean Up Old Versions**")
                    st.write(f"🔄 Found {len(pending_replacement_ids)} old file versions after content updates")
                    st.caption("Safe to delete - these are old versions that have been replaced with updated content")
                    if st.button("Clean Up Old Versions", key="finalize_replacements", icon=":material/restore_page:"):
                        with st.spinner("Cleaning up old versions..."):
                            try:
                                delete_file_ids(VECTOR_STORE_ID, pending_replacement_ids, label="old version")
                                clear_old_file_ids(pending_replacement_ids)
                                st.success("Cleaned up old file versions and updated database references.", icon=":material/check_circle:")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to clean up old versions: {e}", icon=":material/error:")
                else:
                    st.markdown("**Clean Up Old Versions**")
                    st.write("✅ No old versions to clean up")
                    st.caption("All file replacements have been properly finalized")
        
            # Dangerous operation separated and clearly marked
            st.markdown("---")
            st.markdown("#### Nuclear Option")
            st.error("**DANGER**: This will permanently delete ALL files in the vector store!", icon=":material/warning:")
            st.caption("Only use this if you want to completely rebuild the vector store from scratch.")
        
            col_danger1, col_danger2, col_danger3 = st.columns([1, 1, 2])
            with col_danger2:
                if st.button("Delete Everything", type="secondary", help="This will permanently delete all files in the vector store", icon=":material/delete_forever:"):
                    with st.spinner("Deleting all files..."):
                        try:
                            delete_all_files_in_vector_store(VECTOR_STORE_ID)
                            st.success("All files deleted from the vector store.", icon=":material/check_circle:")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to delete files: {e}", icon=":material/error:")
        else:
            st.info("Vector store is empty - no files to manage.", icon=":material/info:")

    
    else:
        st.warning("Authentication required to access the Vector Store Management.", icon=":material/lock:")
