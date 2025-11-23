#!/usr/bin/env python3
"""
Standalone CLI scraper runner for cron/systemd.
- Loads URL configs from DB (utils.load_url_configs)
- Reuses scrape.core.scrape() to crawl and write to DB
- Logs to stdout
- After scraping, optionally uploads new/updated docs to the vector store

Usage examples:
  python scripts/cli_scrape.py --budget 2000 --dry-run
  python scripts/cli_scrape.py --only "university_main"
  python scripts/cli_scrape.py --keep-query page,lang
  python scripts/cli_scrape.py --no-vectorize  # skip vector store sync

Environment / Secrets:
  - Provide Streamlit secrets in one of these ways:
    a) Env var STREAMLIT_SECRETS_JSON containing JSON, e.g.
       '{"postgres": {"host": "...", "port": 5432, "user": "...", "password": "...", "database": "..."}, "OPENAI_API_KEY": "sk-...", "MODEL": "gpt-4o-mini", "VECTOR_STORE_ID": "vs_..."}'
    b) A secrets.toml at ~/.streamlit/secrets.toml or <project>/.streamlit/secrets.toml

Exit codes: 0 success, 1 error
"""
import os
import sys
import argparse
import time
import json
import tomllib
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root on sys.path when run from cron
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Provide Streamlit secrets early, before importing utils/scraper
import streamlit as st


def _ensure_streamlit_secrets():
    # Option 1: JSON via env
    env_json = os.environ.get("STREAMLIT_SECRETS_JSON")
    if env_json:
        try:
            st.secrets._secrets = json.loads(env_json)
            return
        except Exception as e:
            print(f"[WARN] Failed to parse STREAMLIT_SECRETS_JSON: {e}")

    # Option 2: secrets.toml — Streamlit loads it lazily when st.secrets is first accessed.
    candidates = [
        Path.home() / ".streamlit" / "secrets.toml",
        Path(ROOT) / ".streamlit" / "secrets.toml",
    ]
    for p in candidates:
        if p.exists():
            # Try Streamlit's loader; if empty, fall back to manual parse
            try:
                existing = st.secrets  # triggers load
                loaded = False
                try:
                    loaded = bool(existing) and len(existing) > 0
                except Exception:
                    loaded = False
                if loaded:
                    return
            except Exception:
                pass

            try:
                data = tomllib.loads(p.read_text(encoding="utf-8"))
                secrets_obj = getattr(st, "secrets", None)
                if secrets_obj is not None and hasattr(secrets_obj, "_secrets"):
                    secrets_obj._secrets = data  # populate underlying store
                else:
                    st.secrets = data  # fallback mapping
                return
            except Exception as e:
                print(f"[WARN] Failed to load secrets from {p}: {e}")
                continue


_ensure_streamlit_secrets()


def _patch_streamlit_caching():
    """Disable Streamlit's runtime-dependent caches when running headless."""
    try:
        from streamlit.runtime.runtime import Runtime
        runtime_exists = Runtime.exists()
    except Exception:
        runtime_exists = False

    if runtime_exists:
        return

    import functools

    class _NoOpCacheDecorator:
        def __call__(self, *dec_args, **dec_kwargs):
            if dec_args and callable(dec_args[0]) and len(dec_args) == 1 and not dec_kwargs:
                func = dec_args[0]

                @functools.wraps(func)
                def wrapped(*args, **kwargs):
                    return func(*args, **kwargs)

                return wrapped

            def decorator(func):
                @functools.wraps(func)
                def wrapped(*args, **kwargs):
                    return func(*args, **kwargs)

                return wrapped

            return decorator

        def clear(self):
            # Mirrors the cache clear API without doing anything.
            return None

    noop_cache = _NoOpCacheDecorator()

    if hasattr(st, "cache_data"):
        st.cache_data = noop_cache
    if hasattr(st, "cache_resource"):
        st.cache_resource = noop_cache


_patch_streamlit_caching()

# Patch Streamlit UI functions to be safe in headless runs
for _fn_name in ("error", "warning", "info", "success"):
    try:
        setattr(st, _fn_name, lambda msg, _n=_fn_name: print(f"[ST.{_n.upper()}] {msg}"))
    except Exception:
        pass

# Now import the rest (after secrets are in place)
from scrape.maintenance import sync_vector_store, update_stale_documents, purge_orphaned_vector_files

try:
    from utils import write_vector_store_details, clear_vector_store_dirty, update_last_scrape_run
except ImportError:
    def write_vector_store_details(_data):
        return None

    def clear_vector_store_dirty():
        return None

    def update_last_scrape_run(*_args, **_kwargs):
        return None

from utils import (
    get_connection,
    create_knowledge_base_table,
    load_url_configs,
    write_vector_status,
)

from scrape import core as scraper_mod
from scrape.core import reset_scraper_state, scrape, verify_url_deleted, set_run_logger
from scrape.state import CrawlerState

def main():
    parser = argparse.ArgumentParser(description="Run scheduled scraping of configured URLs")
    parser.add_argument(
        "--budget",
        type=int,
        default=5000,
        help="Max URLs per run (crawl budget). Default: 5,000",
    )
    parser.add_argument("--keep-query", type=str, default="", help="Comma-separated query keys to whitelist")
    parser.add_argument("--dry-run", action="store_true", help="Traverse without DB writes or LLM calls")
    parser.add_argument("--only", type=str, default="", help="Only run configs whose recordset contains this substring")
    parser.add_argument("--exclude", type=str, default="", help="Skip configs whose recordset contains this substring")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between configs (rate limiting)")
    # Vector store sync control (enabled by default)
    parser.add_argument("--no-vectorize", dest="vectorize", action="store_false", help="Skip uploading to vector store after scraping")
    parser.add_argument(
        "--mode",
        choices=["scrape", "vectorize", "sync", "all", "cleanup"],
        default="all",
        help=(
            "Which parts of the pipeline to run: 'scrape' for crawling only, "
            "'vectorize' for vector-store sync only, 'sync' for scrape+vectorize without cleanup, "
            "'cleanup' to purge orphaned vector files, or 'all' for scrape+vectorize+cleanup."
        ),
    )
    parser.set_defaults(vectorize=True)
    args = parser.parse_args()

    start_ts = datetime.now(timezone.utc).isoformat()
    print(
        f"[INFO] cli_scrape start {start_ts}Z budget={args.budget} "
        f"dry_run={args.dry_run} mode={args.mode} vectorize={args.vectorize}"
    )

    # Ensure tables exist
    try:
        create_knowledge_base_table()
    except Exception as e:
        print(f"[ERROR] DB init failed: {e}")
        return 1

    run_scrape = args.mode in {"scrape", "sync", "all"}
    run_vectorize = args.mode in {"vectorize", "sync", "all"} and args.vectorize
    cleanup_only_mode = args.mode == "cleanup"
    sync_mode = args.mode == "sync"

    # Load configs only when scraping
    run_configs = []
    total_configs = 0
    if run_scrape:
        try:
            configs = load_url_configs()
        except Exception as e:
            print(f"[ERROR] Failed to load URL configs: {e}")
            return 1

        if not configs:
            print("[WARN] No URL configurations found. Exiting.")
            return 0

        # Filter configs
        def match(c):
            rs = (c.get("recordset") or "")
            if args.only and args.only not in rs:
                return False
            if args.exclude and args.exclude in rs:
                return False
            return True

        run_configs = [c for c in configs if c.get("url") and match(c)]
        total_configs = len(configs)
        print(f"[INFO] {len(run_configs)}/{total_configs} configs selected")

    keep_query_keys = set(s.strip() for s in args.keep_query.split(',') if s.strip()) if args.keep_query else None

    # Acquire simple DB lock to prevent concurrent runs
    lock_conn = None
    try:
        lock_conn = get_connection()
        with lock_conn.cursor() as cur:
            lock_name = "scrape_job"
            # Cleanup stale lock older than 6h
            cur.execute("DELETE FROM job_locks WHERE name=%s AND acquired_at < NOW() - INTERVAL '6 hours'", (lock_name,))
            try:
                cur.execute("INSERT INTO job_locks(name) VALUES (%s)", (lock_name,))
                lock_conn.commit()
            except Exception:
                print("[WARN] Another scrape job is running; exiting.")
                return 0

        # Prepare connection if not dry-run
        conn = None if args.dry_run else get_connection()

        # Configure per-run JSONL log (overwrite each run)
        log_path = Path(ROOT) / "logs" / "scrape-run.jsonl"
        try:
            set_run_logger(log_path, overwrite=True)
            print(f"[INFO] Writing scrape log to {log_path}")
        except Exception as e:
            print(f"[WARN] Could not initialize scrape log {log_path}: {e}")

        # Logging callbacks
        def log_cb(msg, level="INFO"):
            ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
            print(f"[{ts}] [{level}] {msg}")

        def prog_cb():
            # no-op; could add rate-based prints
            pass

        # Global dedupe per run: reset state once before processing all configs
        reset_scraper_state()
        run_state = CrawlerState()

        if run_scrape:
            total = len(run_configs)
            for idx, cfg in enumerate(run_configs, 1):
                url = cfg["url"].strip()
                depth = int(cfg.get("depth", 3))
                recordset = (cfg.get("recordset") or f"recordset_{idx}").strip()
                config_id = cfg.get("id")
                exclude_paths = cfg.get("exclude_paths") or []
                include_lang_prefixes = cfg.get("include_lang_prefixes") or []

                if config_id is None:
                    print(f"[WARN] Config {idx} ({recordset}) has no database ID; skipping to avoid orphaned documents.")
                    continue

                print(f"[INFO] Config {idx}/{total}: url={url} recordset={recordset} depth={depth}")

                try:
                    scrape(
                        url,
                        depth=0,
                        max_depth=depth,
                        recordset=recordset,
                        source_config_id=config_id,
                        conn=conn,
                        exclude_paths=exclude_paths,
                        include_lang_prefixes=include_lang_prefixes,
                        keep_query_keys=keep_query_keys,
                        max_urls_per_run=args.budget,
                        dry_run=args.dry_run,
                        progress_callback=prog_cb,
                        log_callback=log_cb,
                        state=run_state,
                        headers=scraper_mod.HEADERS,
                    )
                except Exception as e:
                    print(f"[ERROR] Error scraping {url}: {e}")

                if args.sleep:
                    time.sleep(args.sleep)

            if conn:
                stale_candidates = []
                try:
                    stale_candidates = update_stale_documents(
                        conn,
                        dry_run=args.dry_run,
                        log_callback=log_cb,
                        recordset_latest_urls=run_state.recordset_latest_urls,
                        verify_url_deleted=verify_url_deleted,
                    )
                except Exception as e:
                    print(f"[WARN] Failed to compute stale documents: {e}")
                else:
                    count = len(stale_candidates)
                    if count:
                        print(f"[INFO] Marked {count} document(s) as stale.")
                        preview = stale_candidates[:5]
                        for entry in preview:
                            rec = entry.get("recordset") or "(none)"
                            print(f"[INFO] Stale candidate: recordset={rec} url={entry.get('url')}")
                        if count > len(preview):
                            print(f"[INFO] ...and {count - len(preview)} more.")
                    else:
                        print("[INFO] No stale documents detected.")
        else:
            if cleanup_only_mode:
                print("[INFO] Skipping scraping stage (--mode=cleanup).")
            elif args.mode == "vectorize":
                print("[INFO] Skipping scraping stage (--mode=vectorize).")
            elif sync_mode:
                print("[WARN] Requested sync mode but no configs available; skipping scrape.")
            else:
                print("[INFO] Skipping scraping stage (no configs requested).")

        if conn:
            try:
                conn.close()
            except Exception:
                pass
            conn = None

        vectorize_success = False

        # After scraping, optionally sync to vector store (skip in dry-run)
        if run_vectorize:
            if args.dry_run:
                print("[INFO] Dry-run mode: skipping vector store sync.")
            else:
                print("[INFO] Starting vector store sync...")
                try:
                    from pages.vectorize import (
                        collect_vector_store_details,
                        compute_vector_store_status,
                        VECTOR_STORE_ID,
                    )
                except Exception as e:
                    print(f"[ERROR] Could not import vectorization module: {e}")
                else:
                    try:
                        result = sync_vector_store()
                        if result and isinstance(result, dict):
                            uploaded = result.get('uploaded_count', 0)
                            print(f"[INFO] Vector store sync finished. Uploaded: {uploaded}")
                        else:
                            print("[INFO] Vector store sync finished.")
                        try:
                            status = compute_vector_store_status(VECTOR_STORE_ID, force_refresh=True)
                            write_vector_status(status)
                            details = collect_vector_store_details(VECTOR_STORE_ID, force_refresh=True)
                            write_vector_store_details(details)
                            clear_vector_store_dirty()
                            print("[INFO] Vector store status and details snapshots updated.")
                        except Exception as status_exc:
                            print(f"[WARN] Could not update vector store snapshots: {status_exc}")
                        vectorize_success = True
                    except Exception as e:
                        print(f"[ERROR] Vector store sync failed: {e}")
        elif args.mode == "scrape":
            print("[INFO] Vector store sync disabled via --mode=scrape.")
        elif sync_mode:
            print("[INFO] Vector store sync requested but skipped (sync mode without successful scrape).")
        elif cleanup_only_mode:
            print("[INFO] Skipping vector store sync (--mode=cleanup).")
        else:
            print("[INFO] Vector store sync disabled via --no-vectorize.")

        should_cleanup = False
        cleanup_reason = ""
        if args.dry_run:
            if cleanup_only_mode:
                print("[INFO] Dry-run mode: skipping orphan cleanup (--mode=cleanup).")
        else:
            if cleanup_only_mode:
                should_cleanup = True
                cleanup_reason = "cleanup-only mode"
            elif args.mode == "all" and vectorize_success:
                should_cleanup = True
                cleanup_reason = "post-sync cleanup"

        if should_cleanup:
            print(f"[INFO] Starting vector store orphan cleanup ({cleanup_reason}).")
            try:
                cleanup_result = purge_orphaned_vector_files()
                deleted = cleanup_result.get("deleted_count", 0) if cleanup_result else 0
                print(f"[INFO] Orphan cleanup finished. Deleted {deleted} file(s).")
            except Exception as exc:
                print(f"[ERROR] Vector store cleanup failed: {exc}")

    finally:
        try:
            set_run_logger(None)
        except Exception:
            pass
        # Release lock
        if lock_conn:
            try:
                with lock_conn.cursor() as cur:
                    cur.execute("DELETE FROM job_locks WHERE name=%s", ("scrape_job",))
                    lock_conn.commit()
                lock_conn.close()
            except Exception:
                pass

    print("[INFO] cli_scrape completed")
    try:
        update_last_scrape_run()
    except Exception as exc:  # pragma: no cover - best effort logging only
        print(f"[WARN] Could not update schedule metadata: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
