import os
from utils import get_connection, admin_authentication, render_sidebar, compute_sha256
import datetime
from openai import OpenAI
from io import BytesIO
import streamlit as st

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
VECTOR_STORE_ID = st.secrets["VECTOR_STORE_ID"]

client = OpenAI(api_key=OPENAI_API_KEY)

def list_all_files_in_vector_store(vector_store_id: str):
    """
    Retrieve *all* files in the vector store by following pagination.
    """
    try:
        all_files = []
        after = None
        while True:
            resp = client.vector_stores.files.list(
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
    
def compute_vector_store_sets(vector_store_id: str):
    """
    Returns three sets of file IDs:
      - current_ids: files referenced as vector_file_id (the live ones)
      - pending_replacement_ids: files referenced as old_file_id (to delete after successful replacement)
      - true_orphan_ids: files in VS that are not in DB at all
    """
    # 1) Vector store files (paginate!)
    vs_files = list_all_files_in_vector_store(vector_store_id)
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
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(*)
        FROM documents
        WHERE vector_file_id IS NULL
          AND old_file_id IS NULL
    """)
    (cnt,) = cur.fetchone()
    cur.close()
    conn.close()
    return cnt


def list_new_unsynced_docs(limit: int = 20):
    """
    Optional: list some of the newest unsynced docs for visibility in the UI.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, title, updated_at
        FROM documents
        WHERE vector_file_id IS NULL
          AND old_file_id IS NULL
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
    for fid in file_ids:
        if not fid:
            continue
        try:
            client.vector_stores.files.delete(vector_store_id=vector_store_id, file_id=fid)
            print(f"🗑️ Detached {label} {fid}")
        except Exception as e:
            print(f"⚠️ Detach failed for {fid}: {e}")
            # continue anyway; maybe already detached
        try:
            client.files.delete(fid)
            print(f"🗑️ Deleted file resource {fid}")
        except Exception as e:
            print(f"⚠️ Delete failed for {fid}: {e}")


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

        for vs_file in vs_files:
            try:
                client.vector_stores.files.delete(vector_store_id=vector_store_id, file_id=vs_file.id)
                print(f"🗑️ Detached file {vs_file.id} from vector store.")
            except Exception as e:
                print(f"⚠️ Failed to detach file {vs_file.id}: {e}")
                continue

            try:
                client.files.delete(vs_file.id)
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

    for file_id in missing_file_ids:
        try:
            client.vector_stores.files.delete(vector_store_id=vector_store_id, file_id=file_id)
            print(f"🗑️ Detached file {file_id} from vector store.")
        except Exception as e:
            print(f"⚠️ Failed to detach file {file_id}: {e}")
            continue

        try:
            client.files.delete(file_id)
            print(f"🗑️ Deleted file resource {file_id}.")
        except Exception as e:
            print(f"⚠️ Failed to delete file resource {file_id}: {e}")

    print("✅ Finished deleting non-mirrored files.")
    

def upload_md_to_vector_store(safe_title: str, content: str, vector_store_id: str) -> str | None:
    file_stream = BytesIO(content.encode("utf-8"))
    file_name = f"{safe_title}.md"

    try:
        batch = client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store_id,
            files=[(file_name, file_stream)]
        )
        print(f"✅ Upload for doc {safe_title} complete. Status: {batch.status}")
        print("📄 File counts:", batch.file_counts)

        # Look through *all* files to find the new one by filename
        vs_files = list_all_files_in_vector_store(vector_store_id)
        for vs_file in vs_files:
            file_meta = client.files.retrieve(vs_file.id)
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
    
    
def sync_vector_store():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, url, title, safe_title, crawl_date, lang, summary, tags,
               markdown_content, vector_file_id, old_file_id
        FROM documents
    """)

    print("🔁 Starting sync for documents...")

    docs_to_upload = []
    id_to_old_file = {}

    for doc_id, url, title, safe_title, crawl_date, lang, summary, tags, markdown, file_id, old_file_id in cur: 
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


    if not docs_to_upload:
        print("✅ No changes to sync.")
        cur.close()
        conn.close()
        return

    # Detach and delete old files
    for doc_id, old_file_id in id_to_old_file.items():
        if not old_file_id:   # 👈 guard against None
            continue

        try:
            client.vector_stores.files.delete(vector_store_id=VECTOR_STORE_ID, file_id=old_file_id)
            print(f"🗑️ Detached file {old_file_id} from vector store.")
        except Exception as e:
            print(f"⚠️ Detach failed for {old_file_id}: {e} (may already be detached/deleted)")

        try:
            client.files.delete(old_file_id)
            print(f"🗑️ Deleted file resource {old_file_id}.")
        except Exception as e:
            print(f"⚠️ Delete failed for {old_file_id}: {e} (may already be gone)")


    # Prepare files for upload
    upload_files = [
        (file_name, BytesIO(content.encode("utf-8")))
        for (_, file_name, content) in docs_to_upload
    ]

    try:
        for idx, files_chunk in enumerate(chunked(upload_files, 100), start=1):
            batch = client.vector_stores.file_batches.upload_and_poll(
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
    

    # Retrieve all files in the vector store to map filenames to file IDs
    try:
        vs_files = list_all_files_in_vector_store(VECTOR_STORE_ID)
    except Exception as e:
        print("❌ Failed to list vector store files:", e)
        cur.close()
        conn.close()
        return

    # Map filenames to file IDs
    filename_to_id = {}
    for vs_file in vs_files:
        try:
            file_obj = client.files.retrieve(vs_file.id)
            filename_to_id[file_obj.filename] = vs_file.id
        except Exception as e:
            print(f"⚠️ Failed to retrieve file object for {vs_file.id}: {e}")

    # Update the database with new file IDs
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


authenticated = admin_authentication()
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
    
    st.title("Vector Store Sync Tool")
    # --- Sync status banner for newly scraped docs (never synced) ---
    try:
        new_unsynced_count = count_new_unsynced_docs()
        last_vs_sync = read_last_vector_sync_timestamp()

        if new_unsynced_count > 0:
            st.warning(
                f"🟡 {new_unsynced_count} newly scraped document(s) have **not** been added to the vector store yet.",
                icon="⚠️"
            )
            # (Optional) Preview a few
            rows_preview = list_new_unsynced_docs(limit=10)
            if rows_preview:
                st.write("Newest unsynced docs:")
                for _id, _title, _updated_at in rows_preview:
                    st.write(f"- `{_id}` · **{_title or '(untitled)'}** · updated: {str(_updated_at) if _updated_at else 'n/a'}")
        else:
            if last_vs_sync:
                st.success(f"✅ Vector store is up to date for newly scraped docs. Last sync: {last_vs_sync.isoformat(timespec='seconds')}")
            else:
                st.info("ℹ️ No previous vector store sync timestamp found.")
    except Exception as e:
        st.error(f"❌ Failed to compute sync status: {e}")

    missing_file_ids = find_missing_file_ids(VECTOR_STORE_ID)
    if missing_file_ids:
        print(f"Missing file IDs: {missing_file_ids}")
    else:
        print("✅ No missing file IDs found.")

    if st.button("Sync Documents with OpenAI Vector Store"):
        with st.spinner("Synchronizing documents..."):
            try:
                result = sync_vector_store()
                st.success("✅ Upload completed successfully.")
                if result:
                    st.subheader("📤 Sync Summary")
                    st.markdown(f"**Uploaded {result['uploaded_count']} new/updated document(s) to the vector store.**")

                    if result["synced_files"]:
                        with st.expander("See synced file list"):
                            for fname in result["synced_files"]:
                                # Strip the numeric prefix if you want, keep just the human part
                                pretty_name = fname.split("_", 1)[1] if "_" in fname else fname
                                st.write(f"- {pretty_name}")

            except Exception as e:
                st.error(f"❌ Upload failed: {e}")

    st.markdown("#### Vector Store Management")

    # Use set classification to show accurate buckets
    try:
        vs_files = list_all_files_in_vector_store(VECTOR_STORE_ID)
        file_count = len(vs_files)

        current_ids, pending_replacement_ids, true_orphan_ids = compute_vector_store_sets(VECTOR_STORE_ID)

        st.markdown(
            f"**📄 Total Files in Vector Store:**\n"
            f"* {file_count} vector-embedded files\n"
            f"* {len(true_orphan_ids)} are **not mirrored** in the local Knowledge Base (true orphans)\n"
            f"* {len(pending_replacement_ids)} are **pending replacements** (listed as `old_file_id`)\n"
            f"* {len(current_ids)} are live (mirrored) files"
        )
    except Exception as e:
        st.error(f"❌ Failed to retrieve file count: {e}")



    # Button to delete all files in the vector store
    if st.button("🗑️ Delete All Files in Vector Store"):
        with st.spinner("Deleting all files..."):
            try:
                delete_all_files_in_vector_store(VECTOR_STORE_ID)
                st.success("✅ All files deleted from the vector store.")
            except Exception as e:
                st.error(f"❌ Failed to delete files: {e}")
                
    # Delete TRUE ORPHANS only (safe cleanup)
    if st.button("🧹 Delete TRUE ORPHANS only"):
        with st.spinner("Deleting true orphans..."):
            try:
                # Recompute to be fresh at click time
                _, _, true_orphan_ids = compute_vector_store_sets(VECTOR_STORE_ID)
                delete_file_ids(VECTOR_STORE_ID, true_orphan_ids, label="orphan")
                st.success("✅ Deleted all true orphans from the vector store.")
            except Exception as e:
                st.error(f"❌ Failed to delete true orphans: {e}")

    # Finalize replacements: delete old_file_id files, then clear DB pointers
    if st.button("♻️ Finalize replacements (delete old_file_id)"):
        with st.spinner("Deleting pending replacement files and clearing DB pointers..."):
            try:
                _, pending_replacement_ids, _ = compute_vector_store_sets(VECTOR_STORE_ID)
                delete_file_ids(VECTOR_STORE_ID, pending_replacement_ids, label="pending replacement")
                clear_old_file_ids(pending_replacement_ids)
                st.success("✅ Finalized replacements: deleted old files and cleared old_file_id in DB.")
            except Exception as e:
                st.error(f"❌ Failed to finalize replacements: {e}")

