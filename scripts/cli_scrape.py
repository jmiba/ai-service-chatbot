#!/usr/bin/env python3
"""
Standalone CLI scraper runner for cron/systemd.
- Loads URL configs from DB (utils.load_url_configs)
- Reuses pages.scrape.scrape() to crawl and write to DB
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
from datetime import datetime
from pathlib import Path

# Ensure project root on sys.path when run from cron
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Provide Streamlit secrets in headless mode
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
            try:
                _ = st.secrets  # triggers load
                return
            except Exception:
                pass

_ensure_streamlit_secrets()

# Patch Streamlit UI functions to be safe in headless runs
for _fn_name in ("error", "warning", "info", "success"):
    try:
        setattr(st, _fn_name, lambda msg, _n=_fn_name: print(f"[ST.{_n.upper()}] {msg}"))
    except Exception:
        pass

# Monkey-patch UI-only utilities before importing pages.scrape
# This avoids Streamlit UI calls during module import
try:
    import utils as utils_pkg
    utils_pkg.admin_authentication = lambda *a, **k: False  # prevent main() from running
    utils_pkg.render_sidebar = lambda *a, **k: None
except Exception as e:
    print(f"[WARN] Could not monkey-patch utils UI helpers: {e}")

# Now import DB helpers and scraper
from utils import get_connection, create_knowledge_base_table, load_url_configs

scraper_mod = None
try:
    import pages.scrape as scraper_mod  # module (to access globals)
    from pages.scrape import scrape  # function
except Exception as e:
    print(f"[ERROR] Failed to import scraper: {e}")
    sys.exit(1)


def reset_scraper_state():
    # Mimic the UI reset before a run
    try:
        scraper_mod.visited_raw.clear()
        scraper_mod.visited_norm.clear()
        scraper_mod.frontier_seen.clear()
        scraper_mod.base_path = None
        scraper_mod.processed_pages_count = 0
        scraper_mod.dry_run_llm_eligible_count = 0
        scraper_mod.llm_analysis_results = []
        if hasattr(scraper_mod, "recordset_latest_urls"):
            scraper_mod.recordset_latest_urls.clear()
    except Exception as e:
        print(f"[WARN] Could not reset scraper state: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run scheduled scraping of configured URLs")
    parser.add_argument("--budget", type=int, default=5000, help="Max URLs per run (crawl budget)")
    parser.add_argument("--keep-query", type=str, default="", help="Comma-separated query keys to whitelist")
    parser.add_argument("--dry-run", action="store_true", help="Traverse without DB writes or LLM calls")
    parser.add_argument("--only", type=str, default="", help="Only run configs whose recordset contains this substring")
    parser.add_argument("--exclude", type=str, default="", help="Skip configs whose recordset contains this substring")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between configs (rate limiting)")
    # Vector store sync control (enabled by default)
    parser.add_argument("--no-vectorize", dest="vectorize", action="store_false", help="Skip uploading to vector store after scraping")
    parser.set_defaults(vectorize=True)
    args = parser.parse_args()

    start_ts = datetime.utcnow().isoformat()
    print(f"[INFO] cli_scrape start {start_ts}Z budget={args.budget} dry_run={args.dry_run} vectorize={args.vectorize}")

    # Ensure tables exist
    try:
        create_knowledge_base_table()
    except Exception as e:
        print(f"[ERROR] DB init failed: {e}")
        return 1

    # Load configs
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
    print(f"[INFO] {len(run_configs)}/{len(configs)} configs selected")

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

        # Reset scraper globals for a clean run
        reset_scraper_state()

        # Logging callbacks
        def log_cb(msg, level="INFO"):
            ts = datetime.utcnow().strftime("%H:%M:%S")
            print(f"[{ts}] [{level}] {msg}")

        def prog_cb():
            # no-op; could add rate-based prints
            pass

        total = len(run_configs)
        for idx, cfg in enumerate(run_configs, 1):
            url = cfg["url"].strip()
            depth = int(cfg.get("depth", 2))
            recordset = (cfg.get("recordset") or f"recordset_{idx}").strip()
            exclude_paths = cfg.get("exclude_paths") or []
            include_lang_prefixes = cfg.get("include_lang_prefixes") or []

            print(f"[INFO] Config {idx}/{total}: url={url} recordset={recordset} depth={depth}")

            try:
                scrape(
                    url,
                    depth=0,
                    max_depth=depth,
                    recordset=recordset,
                    conn=conn,
                    exclude_paths=exclude_paths,
                    include_lang_prefixes=include_lang_prefixes,
                    keep_query_keys=keep_query_keys,
                    max_urls_per_run=args.budget,
                    dry_run=args.dry_run,
                    progress_callback=prog_cb,
                    log_callback=log_cb,
                )
            except Exception as e:
                print(f"[ERROR] Error scraping {url}: {e}")

            if args.sleep:
                time.sleep(args.sleep)

        if conn:
            stale_candidates = []
            try:
                stale_candidates = scraper_mod.update_stale_documents(
                    conn,
                    dry_run=args.dry_run,
                    log_callback=log_cb,
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
            try:
                conn.close()
            except Exception:
                pass

        # After scraping, optionally sync to vector store (skip in dry-run)
        if not args.dry_run and args.vectorize:
            print("[INFO] Starting vector store sync...")
            try:
                # Import lazily to avoid heavy Streamlit UI code; utils UI was monkey-patched above
                from pages.vectorize import sync_vector_store
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
                except Exception as e:
                    print(f"[ERROR] Vector store sync failed: {e}")
        elif args.dry_run:
            print("[INFO] Dry-run mode: skipping vector store sync.")
        else:
            print("[INFO] Vector store sync disabled via --no-vectorize.")

    finally:
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
