# Scraping Pipeline (KISS Overview)

This summarizes the scraping stack after the recent refactor, focusing on how to run it, what state it keeps, and where to change things.

- `scrape/config.py`: shared prompt loader `load_summarize_prompts()` (reads `.streamlit/prompts.json`).
- `scrape/settings.py`: `ScrapeSettings` (max depth, budget, filters, headers) and `build_headers(admin_email)`.
- `scrape/state.py`: `CrawlerState` dataclass (visited sets, counters, per-recordset URLs, collected LLM results) with `reset()`.
- `scrape/core.py`: BFS crawler + LLM summarization + DB write. Accepts `state` and `headers` so callers control dedupe scope and UA.
- `scripts/cli_scrape.py`: CLI runner that loads secrets, builds headers, instantiates a per-run `CrawlerState`, runs configs with global dedupe, writes JSONL log to `logs/scrape-run.jsonl`.
- `pages/scrape.py`: Streamlit UI; now uses the shared prompt loader. Should call the CLI or the core with an explicit `CrawlerState` and settings.

## Dedupe scope
- Per run, global across all URL configs. The same `CrawlerState` is reused for the entire run, so a normalized URL is crawled at most once per run.
- Normalization: lowercase scheme/host, strip default ports/fragments, collapse slashes, trim trailing slash (except root), optional query whitelist.

## Crawl flow (BFS)
1. Normalize seed URL, derive base path (stay under same path unless root).
2. Enqueue normalized URL if budget/depth allow.
3. Loop queue:
   - HEAD advisory check (skip non-HTML files by extension/content-type).
   - GET; skip 3xx/4xx/5xx as noted.
   - Enforce HTML content-type.
   - Normalize effective URL, apply canonical; dedupe against `state.visited_norm`.
   - Extract main content; skip obvious 404 pages.
   - Hash content; skip duplicates/unchanged; in dry-run just count LLM-eligible pages.
   - If not dry-run: LLM summarize/tag, save to DB, increment processed counter.
   - Discover links on same host; apply include/exclude path filters, base-path fence, query loop guard; enqueue unseen URLs.

## Logging
- CLI writes JSONL to `logs/scrape-run.jsonl` (overwritten each run). Each line has `timestamp`, `event`, and contextual fields (`normalized_url`, `depth`, `recordset`, etc.).
- UI can pass a log callback or wire its own logger; core logging is lightweight and non-fatal.

## Secrets/config
- Core shouldn’t read `st.secrets` directly. Callers (CLI/UI) should load secrets, build headers via `build_headers(admin_email)`, load prompts via `load_summarize_prompts()`, and pass settings/state into `scrape()`.
- Prompts live in `.streamlit/prompts.json` under the `summarize_and_tag` object with `system`/`user` strings. Shared loader is in `scrape/config.py`.

## How to run
- CLI: `python3 scripts/cli_scrape.py --dry-run --budget 5000 --mode all`
- UI: use the Streamlit page; ensure it passes a fresh `CrawlerState` per run and settings built from UI inputs.

## Next simplifications (optional)
- Split `core.py` into smaller modules (`normalize`, `fetch`, `extract`, `persist`, `logging`) and import them in a thin runner.
- Standardize logging on `logging.Logger` with a JSONL handler instead of bespoke writer.
