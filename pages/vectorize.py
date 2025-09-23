import os
from utils import get_connection, admin_authentication, render_sidebar, get_document_status_counts
import datetime
from openai import OpenAI
from io import BytesIO
import streamlit as st
import time
from pathlib import Path
import base64

try:
    from streamlit.runtime import exists as streamlit_runtime_exists  # type: ignore
except (ImportError, ModuleNotFoundError):
    streamlit_runtime_exists = None  # type: ignore

try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore
except (ImportError, ModuleNotFoundError):
    def get_script_run_ctx():  # type: ignore
        return None

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
    
    openai_client = _get_openai_client()

    for i, fid in enumerate(file_ids):
        if not fid:
            continue
        try:
            openai_client.vector_stores.files.delete(vector_store_id=vector_store_id, file_id=fid)
            print(f"🗑️ Detached {label} {fid}")
        except Exception as e:
            print(f"⚠️ Detach failed for {fid}: {e}")
            # continue anyway; maybe already detached
        try:
            openai_client.files.delete(fid)
            print(f"🗑️ Deleted file resource {fid}")
        except Exception as e:
            print(f"⚠️ Delete failed for {fid}: {e}")
        
        # Progress indicator for large batches
        if (i + 1) % 20 == 0:
            print(f"📍 Deleted {i + 1}/{len(file_ids)} {label} files...")
    
    print(f"✅ Finished deleting {len(file_ids)} {label} files")
    
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
    openai_client = _get_openai_client()
    try:
        # List all files in the vector store
        vs_files = list_all_files_in_vector_store(vector_store_id)

        if not vs_files:
            print("✅ No files to delete in the vector store.")
            return

        for vs_file in vs_files:
            try:
                openai_client.vector_stores.files.delete(vector_store_id=vector_store_id, file_id=vs_file.id)
                print(f"🗑️ Detached file {vs_file.id} from vector store.")
            except Exception as e:
                print(f"⚠️ Failed to detach file {vs_file.id}: {e}")
                continue

            try:
                openai_client.files.delete(vs_file.id)
                print(f"🗑️ Deleted file resource {vs_file.id}.")
            except Exception as e:
                print(f"⚠️ Failed to delete file resource {vs_file.id}: {e}")

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

    openai_client = _get_openai_client()

    for file_id in missing_file_ids:
        try:
            openai_client.vector_stores.files.delete(vector_store_id=vector_store_id, file_id=file_id)
            print(f"🗑️ Detached file {file_id} from vector store.")
        except Exception as e:
            print(f"⚠️ Failed to detach file {file_id}: {e}")
            continue

        try:
            openai_client.files.delete(file_id)
            print(f"🗑️ Deleted file resource {file_id}.")
        except Exception as e:
            print(f"⚠️ Failed to delete file resource {file_id}: {e}")

    print("✅ Finished deleting non-mirrored files.")
    

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

        # Look through *all* files to find the new one by filename
        vs_files = list_all_files_in_vector_store(vector_store_id)
        for vs_file in vs_files:
            file_meta = openai_client.files.retrieve(vs_file.id)
            if file_meta.filename == file_name:
                print(f"📂 Upload successful, found vector file ID: {vs_file.id} for {file_name}")
                return vs_file.id

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
    filename_to_id = {}
    failed_retrievals = 0
    
    print(f"🔍 Mapping {len(vs_files)} vector store files to filenames...")
    
    openai_client = _get_openai_client()

    for i, vs_file in enumerate(vs_files):
        try:
            file_obj = openai_client.files.retrieve(vs_file.id)
            filename_to_id[file_obj.filename] = vs_file.id
            
            # Progress indicator for large batches
            if (i + 1) % 50 == 0:
                print(f"📍 Processed {i + 1}/{len(vs_files)} files...")
                
        except Exception as e:
            failed_retrievals += 1
            print(f"⚠️ Failed to retrieve file object for {vs_file.id}: {e}")
    
    if failed_retrievals > 0:
        print(f"⚠️ {failed_retrievals} file retrievals failed")
    
    print(f"✅ Successfully mapped {len(filename_to_id)} filenames")
    return filename_to_id

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
        print(f"🗑️ Deleting {len(old_file_ids)} old files...")
        for old_file_id in old_file_ids:
            try:
                openai_client.vector_stores.files.delete(vector_store_id=VECTOR_STORE_ID, file_id=old_file_id)
                print(f"🗑️ Detached file {old_file_id} from vector store.")
            except Exception as e:
                print(f"⚠️ Detach failed for {old_file_id}: {e} (may already be detached/deleted)")

            try:
                openai_client.files.delete(old_file_id)
                print(f"🗑️ Deleted file resource {old_file_id}.")
            except Exception as e:
                print(f"⚠️ Delete failed for {old_file_id}: {e} (may already be gone)")

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

    # -----------------------------
    # Terminal-style logging for vectorize operations
    # -----------------------------
    import io
    import sys
    from contextlib import redirect_stdout

    def capture_sync_output():
        """Context manager to capture print output during sync"""
        output_buffer = io.StringIO()
        return output_buffer

    def display_sync_log(log_container, output_buffer):
        """Display captured output in terminal-style interface"""
        output_text = output_buffer.getvalue()
        if output_text:
            log_container.text_area(
                "📋 Sync Progress Log", 
                value=output_text,
                height=400,
                disabled=True,
                key=f"sync_log_{time.time()}"  # Unique key for each sync
            )

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
        
            # Expensive vector store operations - only load when needed
            vector_mgmt_col1, vector_mgmt_col2 = st.columns([3, 1])
        
            with vector_mgmt_col1:
                show_vector_mgmt = st.checkbox("Show vector store status", help="Load detailed vector store statistics (uses caching for better performance)")
        
            with vector_mgmt_col2:
                if st.button("Refresh Cache", help="Force refresh vector store data cache", icon=":material/refresh:"):
                    _vs_files_cache["data"] = None  # Clear cache
                    st.success("Cache cleared!")
                    st.rerun()
        
            if show_vector_mgmt:
                with st.spinner("Loading vector store details (using cache when possible)..."):
                    try:
                        # Vector store stats (now uses caching for better performance!)
                        vs_files = get_cached_vector_store_files(VECTOR_STORE_ID)
                        vs_file_count = len(vs_files)
                        current_ids, pending_replacement_ids, true_orphan_ids = compute_vector_store_sets(VECTOR_STORE_ID, use_cache=True)
                    
                        # Show cache status
                        cache_age = time.time() - _vs_files_cache["timestamp"]
                        st.caption(f"Using cached data (age: {cache_age:.0f}s)")
                    
                        # Compute excluded live files present in VS and include them in pending cleanup metric
                        vs_ids = {f.id for f in vs_files}
                        try:
                            conn = get_connection(); cur = conn.cursor()
                            cur.execute("SELECT vector_file_id FROM documents WHERE no_upload IS TRUE AND vector_file_id IS NOT NULL")
                            excluded_live_ids = {row[0] for row in cur.fetchall()}
                            cur.close(); conn.close()
                        except Exception:
                            excluded_live_ids = set()
                        combined_pending_cleanup_ids = (pending_replacement_ids | excluded_live_ids) & vs_ids
                    
                        # Live files should exclude excluded docs
                        live_ids_display = current_ids - excluded_live_ids
                    
                        st.header("Vector Store Status")
                        col6, col7, col8, col9 = st.columns(4)
                        with col6:
                            st.metric("VS Files", vs_file_count, border=True)
                        with col7:
                            st.metric("Live Files", len(live_ids_display), border=True)
                        with col8:
                            st.metric("Pending Cleanup", len(combined_pending_cleanup_ids), border=True)
                        with col9:
                            st.metric("Orphans", len(true_orphan_ids), border=True)
                    
                        # Vector store status messages
                        if len(true_orphan_ids) > 0:
                            st.warning(f"🧹 **{len(true_orphan_ids)} orphaned files** in vector store are no longer in the database.")
                        
                        if len(combined_pending_cleanup_ids) > 0:
                            st.info(f"**{len(combined_pending_cleanup_ids)} files** pending cleanup (old versions or excluded docs).", icon=":material/cached:")
                        
                    except Exception as e:
                        st.error(f"Failed to load vector store details: {e}", icon=":material/error:")
                        vs_file_count = 0
                        current_ids = set()
                        pending_replacement_ids = set()
                        true_orphan_ids = set()
            else:
                # Set defaults when not loading vector store details
                vs_file_count = 0
                current_ids = set()
                pending_replacement_ids = set()
                true_orphan_ids = set()

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

        missing_file_ids = find_missing_file_ids(VECTOR_STORE_ID)
        if missing_file_ids:
            print(f"Missing file IDs: {missing_file_ids}")
        else:
            print("✅ No missing file IDs found.")

    
        st.markdown("---")
        st.header("Vector Store File Management")
        
        # Action: Sync now
        st.markdown("#### Sync Knowlede Base Now")
        st.caption("Uploads new or changed knowledge base entries to vector store and deletes files marked as excluded from vector store.")
        
        # Sync button with better context
        sync_button_text = "Sync Documents with Vector Store"
        if new_unsynced_count > 0 or pending_resync_count > 0:
            sync_button_text += f" ({new_unsynced_count + pending_resync_count} pending)"
    
        if st.button(sync_button_text, type="primary", icon=":material/sync:"):
            # Create containers for status and terminal output
            status_container = st.container()
            progress_bar = st.progress(0)
            status_text = st.empty()
            log_container = st.empty()
        
            with status_container:
                status_text.info("🚀 **Starting synchronization process...**")
                progress_bar.progress(10)
            
                # Create output buffer to capture print statements
                output_buffer = io.StringIO()
            
                try:
                    status_text.info("**Uploading documents to vector store...**")
                    progress_bar.progress(30)
                
                    # Capture all print output during sync
                    with redirect_stdout(output_buffer):
                        result = sync_vector_store()
                
                    # Display the captured output in terminal style
                    display_sync_log(log_container, output_buffer)
                
                    status_text.info("**Finalizing synchronization...**", icon=":material/clock_loader_90:")
                    progress_bar.progress(80)
                
                    # Get updated counts after sync
                    counts = get_document_status_counts()
                    updated_new_count = counts["new_unsynced_count"]
                    updated_resync_count = counts["pending_resync_count"]
                    updated_vectorized = counts["vectorized_docs"]
                
                    progress_bar.progress(100)
                
                    # Success message with detailed results
                    st.success("**Synchronization completed successfully!**", icon=":material/check_circle:")
                
                    # Show sync summary in an info box
                    with st.container():
                        st.markdown("### 📊 Sync Results")
                    
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("📤 Uploaded", 
                                     result['uploaded_count'] if result else 0,
                                     help="Documents successfully uploaded to vector store")
                        with col2:
                            st.metric("📝 New Remaining", 
                                     updated_new_count,
                                     help="New documents still waiting for sync")
                        with col3:
                            st.metric("🔄 Re-sync Remaining", 
                                     updated_resync_count,
                                     help="Documents still needing re-synchronization")
                        with col4:
                            st.metric("✅ Total Vectorized", 
                                     updated_vectorized,
                                     help="Total documents now in vector store")
                    
                        if result and result["uploaded_count"] > 0:
                            st.info(f"**{result['uploaded_count']} documents** were successfully uploaded to the vector store and are now available for AI search!", icon=":material/award_star:")
                        
                            if result["synced_files"]:
                                with st.expander("View uploaded files", expanded=False, icon=":material/visibility:"):
                                    for fname in result["synced_files"]:
                                        # Strip the numeric prefix for cleaner display
                                        pretty_name = fname.split("_", 1)[1] if "_" in fname else fname
                                        st.write(f"✅ {pretty_name}")
                    
                        # Show next steps or all-clear message
                        if updated_new_count == 0 and updated_resync_count == 0:
                            st.success("**All documents are now synchronized!** Your knowledge base is fully up to date.", icon=":material/check_circle:")
                        else:
                            remaining_total = updated_new_count + updated_resync_count
                            st.warning(f"**{remaining_total} documents** still need synchronization. Run sync again if needed.", icon=":material/sync_problem:" )
                
                    # Clear the progress indicators
                    progress_bar.empty()
                    status_text.empty()
                
                    # Force cache refresh and page refresh to show updated status
                    _vs_files_cache["data"] = None  # Clear cache to ensure fresh data
                    st.balloons()  # Celebrate successful sync!
                    time.sleep(1)  # Brief pause to show balloons
                    st.rerun()  # Refresh the page to show updated metrics

                except Exception as e:
                    # Still show captured output even if there was an error
                    display_sync_log(log_container, output_buffer)
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"**Synchronization failed:** {e}", icon=":material/error:")
                    st.info("Try running the sync again or check the logs above for more details.", icon=":material/info:")


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

        # Show file management options only if there are files to manage
        if vs_file_count > 0:
            # Recompute combined pending cleanup count for the info summary
            try:
                vs_ids = {f.id for f in vs_files}
                conn = get_connection(); cur = conn.cursor()
                cur.execute("SELECT vector_file_id FROM documents WHERE no_upload IS TRUE AND vector_file_id IS NOT NULL")
                excluded_live_ids = {row[0] for row in cur.fetchall()}
                cur.close(); conn.close()
                combined_pending_cleanup_count = len(((pending_replacement_ids | excluded_live_ids) & vs_ids))
                live_display_count = len(current_ids - excluded_live_ids)
            except Exception:
                combined_pending_cleanup_count = len(pending_replacement_ids)
                live_display_count = len(current_ids)
        
            st.info(f"**Vector Store contains {vs_file_count} files** ({live_display_count} live, {combined_pending_cleanup_count} pending cleanup, {len(true_orphan_ids)} orphans)", icon=":material/storage:")
        
            # Cleanup options - organized by priority and safety
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
