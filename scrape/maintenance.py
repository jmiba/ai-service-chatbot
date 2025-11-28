from __future__ import annotations

import asyncio
import datetime
from collections.abc import Iterable, Mapping
from io import BytesIO
from typing import Any, Callable

from utils import get_connection


_VECTORIZE_MODULE: Any | None = None


def _get_vectorize_module():
    """Lazy import to avoid pulling Streamlit context when unused."""
    global _VECTORIZE_MODULE
    if _VECTORIZE_MODULE is None:
        import pages.vectorize as vectorize_module

        _VECTORIZE_MODULE = vectorize_module
    return _VECTORIZE_MODULE


def update_stale_documents(
    conn,
    dry_run: bool = False,
    log_callback=None,
    *,
    config_latest_urls: Mapping[int, Iterable[str]] | None = None,
    verify_url_deleted: Callable[[str, Callable[[str], None] | None], tuple[bool, str]] | None = None,
):
    """Update `documents.is_stale` by confirming which pages disappeared during this crawl."""

    if verify_url_deleted is None:
        raise ValueError("verify_url_deleted callback is required.")

    connection_owned = False
    if conn is None:
        if dry_run:
            try:
                conn = get_connection()
                connection_owned = True
                if log_callback:
                    log_callback("🔍 [DRY RUN] Opened temporary DB connection for stale check")
            except Exception as exc:
                if log_callback:
                    log_callback(f"⚠️ Could not open DB connection for stale detection: {exc}")
                return []
        else:
            return []

    def _log(message: str):
        if log_callback:
            log_callback(message)

    visited_source = config_latest_urls or {}
    visited_by_config_id: dict[int, set[str]] = {
        config_id: set(urls) for config_id, urls in visited_source.items()
    }

    stale_checks: list[dict] = []
    seen_ids: set[int] = set()

    try:
        with conn.cursor() as cur:
            for config_id, urls in visited_by_config_id.items():
                cur.execute(
                    """SELECT d.id, uc.recordset, d.url, d.title, d.crawl_date 
                       FROM documents d
                       LEFT JOIN url_configs uc ON d.source_config_id = uc.id
                       WHERE d.source_config_id = %s""",
                    (config_id,),
                )
                rows = cur.fetchall()
                for doc_id, doc_recordset, doc_url, doc_title, doc_crawl in rows:
                    if doc_url in urls:
                        seen_ids.add(doc_id)
                    else:
                        stale_checks.append(
                            {
                                "id": doc_id,
                                "recordset": doc_recordset,
                                "url": doc_url,
                                "title": doc_title,
                                "crawl_date": doc_crawl.isoformat() if doc_crawl else None,
                            }
                        )
    except Exception as exc:
        _log(f"⚠️ Could not compute stale documents: {exc}")
        try:
            conn.rollback()
        except Exception:
            pass
        return []

    # Ensure pages we actually saw are marked fresh
    if seen_ids and not dry_run:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE documents SET is_stale = FALSE WHERE id = ANY(%s)",
                    (list(seen_ids),),
                )
        except Exception as exc:
            _log(f"⚠️ Could not reset stale flag for seen documents: {exc}")

    confirmed_stale: list[dict] = []
    retained_ids: list[int] = []

    for entry in stale_checks:
        is_missing, reason = verify_url_deleted(entry["url"], log_callback=_log)
        if is_missing:
            entry["reason"] = reason
            confirmed_stale.append(entry)
            _log(f"🗑️ Confirmed missing: {entry['url']} ({reason})")
        else:
            retained_ids.append(entry["id"])
            _log(f"✅ Still reachable: {entry['url']}")

    if not dry_run:
        try:
            with conn.cursor() as cur:
                if confirmed_stale:
                    cur.execute(
                        "UPDATE documents SET is_stale = TRUE, updated_at = NOW() WHERE id = ANY(%s)",
                        ([entry["id"] for entry in confirmed_stale],),
                    )
                if retained_ids:
                    cur.execute(
                        "UPDATE documents SET is_stale = FALSE WHERE id = ANY(%s)",
                        (retained_ids,),
                    )
            conn.commit()
        except Exception as exc:
            _log(f"⚠️ Failed to persist stale status updates: {exc}")
            try:
                conn.rollback()
            except Exception:
                pass

    if connection_owned and conn:
        try:
            conn.close()
        except Exception:
            pass

    _log(f"📦 Stale detection complete: {len(confirmed_stale)} confirmed missing / {len(stale_checks)} checked")
    return confirmed_stale


def write_last_vector_sync_timestamp(ts: datetime.datetime | None = None):
    if ts is None:
        ts = datetime.datetime.now()
    with open("last_vector_sync.txt", "w") as handle:
        handle.write(ts.isoformat())


def chunked(iterable, size: int):
    """Yield successive chunks of length `size` from a list."""
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


def build_filename_to_id_map_efficiently(vs_files):
    """Build filename to ID mapping more efficiently by batching API calls."""

    vectorize = _get_vectorize_module()
    _create_async_openai_client = vectorize._create_async_openai_client
    _run_async_task = vectorize._run_async_task

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


def purge_orphaned_vector_files(force_refresh: bool = True) -> dict[str, int] | None:
    """Delete vector store files that are no longer referenced in the database."""

    vectorize = _get_vectorize_module()
    VECTOR_STORE_ID = vectorize.VECTOR_STORE_ID
    collect_vector_store_details = vectorize.collect_vector_store_details
    delete_file_ids = vectorize.delete_file_ids

    print("🔎 Inspecting vector store for orphaned files...")
    details = collect_vector_store_details(VECTOR_STORE_ID, force_refresh=force_refresh)
    true_orphan_ids = set(details.get("true_orphan_ids") or [])

    if not true_orphan_ids:
        print("✅ No orphaned vector files detected.")
        return {"deleted_count": 0}

    print(f"🧹 Deleting {len(true_orphan_ids)} orphaned vector files...")
    delete_file_ids(VECTOR_STORE_ID, true_orphan_ids, label="orphan")

    try:
        status = vectorize.compute_vector_store_status(VECTOR_STORE_ID, force_refresh=True)
        vectorize.write_vector_status(status)
        refreshed_details = collect_vector_store_details(VECTOR_STORE_ID, force_refresh=True)
        vectorize.write_vector_store_details(refreshed_details)
        vectorize.clear_vector_store_dirty()
        try:
            vectorize._vs_files_cache["data"] = None  # ensure next read hits OpenAI
        except Exception:
            pass
    except Exception as exc:  # pragma: no cover - just logging
        print(f"[WARN] Could not refresh vector snapshots after cleanup: {exc}")

    return {"deleted_count": len(true_orphan_ids)}


def sync_vector_store():
    vectorize = _get_vectorize_module()
    VECTOR_STORE_ID = vectorize.VECTOR_STORE_ID
    delete_file_ids = vectorize.delete_file_ids
    get_cached_vector_store_files = vectorize.get_cached_vector_store_files

    conn = get_connection()
    cur = conn.cursor()
    openai_client = vectorize._get_openai_client()

    cur.execute(
        """
        SELECT id, url, title, safe_title, crawl_date, lang, summary, tags,
               markdown_content, vector_file_id, old_file_id, no_upload
        FROM documents
    """
    )

    print("🔁 Starting sync for documents...")

    docs_to_upload = []
    id_to_old_file = {}
    excluded_doc_ids = []
    excluded_file_ids = set()

    for row in cur:
        (
            doc_id,
            url,
            title,
            safe_title,
            crawl_date,
            lang,
            summary,
            tags,
            markdown,
            file_id,
            old_file_id,
            no_upload,
        ) = row

        safe_title = safe_title or (title or "untitled")

        if no_upload:
            if file_id:
                excluded_doc_ids.append(doc_id)
                excluded_file_ids.add(file_id)
                print(f"🚫 Excluding doc {safe_title} (ID {doc_id}) -> will delete vector file {file_id}")
            else:
                if old_file_id:
                    id_to_old_file[doc_id] = old_file_id
                    print(f"🧹 Excluded doc {safe_title} has old_file_id {old_file_id} scheduled to clear")
            continue

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
            if old_file_id:
                id_to_old_file[doc_id] = old_file_id
                print(f"Content change detected for {safe_title}, will delete old file -> {old_file_id}")
            else:
                print(f"ℹ️ No old_file_id for {safe_title} (first-time upload or already cleaned).")

        file_name = f"{doc_id}_{safe_title}.md"
        docs_to_upload.append((doc_id, file_name, content))

    if excluded_file_ids:
        print(f"🗑️ Deleting {len(excluded_file_ids)} excluded vector files...")
        delete_file_ids(VECTOR_STORE_ID, excluded_file_ids, label="excluded")

    if excluded_doc_ids:
        cur2 = conn.cursor()
        cur2.execute(
            """
            UPDATE documents
            SET vector_file_id = NULL,
                old_file_id = NULL,
                updated_at = NOW()
            WHERE id = ANY(%s)
            """,
            (excluded_doc_ids,),
        )
        conn.commit()
        cur2.close()
        print(f"✅ Cleared DB references for {len(excluded_doc_ids)} excluded docs")

    if not docs_to_upload:
        print("✅ No changes to sync.")
        # Record the health check so the UI shows the latest run even if nothing changed.
        write_last_vector_sync_timestamp(datetime.datetime.now())
        cur.close()
        conn.close()
        return

    print(f"📤 Will upload {len(docs_to_upload)} documents")

    old_file_ids = [fid for fid in id_to_old_file.values() if fid]
    if old_file_ids:
        delete_file_ids(VECTOR_STORE_ID, set(old_file_ids), label="old")

    upload_files = [
        (file_name, BytesIO(content.encode("utf-8"))) for (_, file_name, content) in docs_to_upload
    ]

    print(f"📤 Starting batch upload of {len(upload_files)} files...")

    try:
        for idx, files_chunk in enumerate(chunked(upload_files, 100), start=1):
            print(f"📦 Uploading batch {idx} ({len(files_chunk)} files)...")
            batch = openai_client.vector_stores.file_batches.upload_and_poll(
                vector_store_id=VECTOR_STORE_ID, files=files_chunk
            )
            print(f"✅ Batch {idx} upload complete. Status: {batch.status}")
            print("📄 File counts:", batch.file_counts)
    except Exception as exc:
        print("❌ Batch upload failed:", exc)
        cur.close()
        conn.close()
        return

    print("🔄 Refreshing vector store cache after upload...")
    vs_files = get_cached_vector_store_files(VECTOR_STORE_ID, force_refresh=True)

    filename_to_id = build_filename_to_id_map_efficiently(vs_files)

    print(f"💾 Updating database for {len(docs_to_upload)} documents...")
    for doc_id, file_name, _ in docs_to_upload:
        file_id = filename_to_id.get(file_name)
        if not file_id:
            print(f"❌ Could not find file ID for {file_name}, skipping DB update.")
            continue

        cur.execute(
            """
            UPDATE documents
            SET vector_file_id = %s,
                old_file_id = NULL,
                updated_at = NOW()
            WHERE id = %s
        """,
            (file_id, doc_id),
        )
        print(f"✅ Synced doc {file_name} → file {file_id}")

    conn.commit()
    cur.close()
    conn.close()
    print("🎉 Sync complete.")

    write_last_vector_sync_timestamp(datetime.datetime.now())

    return {"uploaded_count": len(docs_to_upload), "synced_files": [file_name for (_, file_name, _) in docs_to_upload]}
