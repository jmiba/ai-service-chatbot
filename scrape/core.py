from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
import fnmatch
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests
import streamlit as st
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from openai import OpenAI

from utils import compute_sha256, get_connection, normalize_tags_for_storage
from scrape.state import CrawlerState
from scrape.settings import build_headers, ScrapeSettings
from scrape.config import load_summarize_prompts

BASE_DIR = Path(__file__).resolve().parent.parent

PROMPT_LOAD_ERROR = None
try:
    SUMMARIZE_SYSTEM_PROMPT, SUMMARIZE_USER_TEMPLATE = load_summarize_prompts()
except RuntimeError as exc:  # pragma: no cover - requires misconfiguration
    PROMPT_LOAD_ERROR = str(exc)
    SUMMARIZE_SYSTEM_PROMPT = ""
    SUMMARIZE_USER_TEMPLATE = ""

try:
    _admin_email = st.secrets.get("ADMIN_EMAIL")
except Exception:
    _admin_email = None

# Default headers (can be overridden per-call)
HEADERS = build_headers(_admin_email)

# Default crawl state (legacy module-level access preserved for collections)
_default_state = CrawlerState()
visited_raw = _default_state.visited_raw
visited_norm = _default_state.visited_norm
frontier_seen = _default_state.frontier_seen
recordset_latest_urls = _default_state.recordset_latest_urls


def reset_scraper_state(state: CrawlerState | None = None):
    """Reset crawl state; defaults to the module-level state for backward compatibility."""
    target = state or _default_state
    target.reset()

_run_log_fp = None
_run_log_path: Path | None = None


def set_run_logger(path: str | Path | None, *, overwrite: bool = True):
    """Configure JSONL logger for a run. Pass None to disable."""
    global _run_log_fp, _run_log_path
    if _run_log_fp and not _run_log_fp.closed:
        try:
            _run_log_fp.close()
        except Exception:
            pass
    _run_log_fp = None
    _run_log_path = None

    if path is None:
        return

    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "a"
    _run_log_fp = log_path.open(mode, encoding="utf-8")
    _run_log_path = log_path


def _log_event(event: str, **fields):
    """Write a single JSONL log entry if a run logger is configured."""
    if _run_log_fp is None:
        return
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": event}
    record.update({k: v for k, v in fields.items() if v is not None})
    try:
        _run_log_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        _run_log_fp.flush()
    except Exception:
        # Logging should never break the crawl.
        pass


# URL normalization utilities
DEFAULT_PORTS = {"http": "80", "https": "443"}


def strip_default_port(netloc: str, scheme: str) -> str:
    if ":" in netloc:
        host, port = netloc.rsplit(":", 1)
        if port == DEFAULT_PORTS.get(scheme, ""):
            return host
    return netloc


def normalize_query(query: str, keep: set[str] | None = None) -> str:
    """
    If keep is None: drop all query params.
    If keep is a set: only keep those keys (preserving values).
    """
    if keep is None:
        return ""
    pairs = [(k, v) for k, v in parse_qsl(query, keep_blank_values=True) if k in keep]
    return urlencode(pairs, doseq=True)


def normalize_url(base_url: str, href: str, keep_query: set[str] | None = None) -> str:
    """
    Resolve href against base_url, then:
    - lowercase scheme/host
    - strip fragment
    - strip default ports
    - collapse duplicate slashes
    - strip trailing slash (except root)
    - drop or whitelist query parameters
    """
    full = urljoin(base_url, href)
    p = urlparse(full)

    scheme = p.scheme.lower()
    netloc = strip_default_port(p.netloc.lower(), scheme)

    path = re.sub(r"/{2,}", "/", p.path or "/")
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]

    query = normalize_query(p.query, keep=keep_query)
    # never keep fragment — it’s purely client-side
    return urlunparse((scheme, netloc, path, "", query, ""))


def compute_base_path(norm_url: str) -> str:
    """Derive a crawl base path from a normalized URL."""
    parsed = urlparse(norm_url)
    path_parts = parsed.path.rstrip('/').split('/')
    if path_parts and path_parts[-1] and '.' in path_parts[-1]:
        base = '/'.join(path_parts[:-1])
    else:
        base = '/'.join(path_parts)
    if base in ("", "/"):
        return "/"
    if not base.endswith("/"):
        base += "/"
    return base


def get_canonical_url(soup: BeautifulSoup, fetched_url: str) -> str | None:
    """
    Respect <link rel="canonical"> when present.
    """
    link = soup.find("link", rel=lambda x: x and "canonical" in x.lower())
    if link and link.get("href"):
        href = link["href"].strip()
        try:
            cand = urljoin(fetched_url, href)
            return cand
        except Exception:
            return None
    return None


def get_html_lang(soup):
    html_tag = soup.find("html")
    if html_tag and html_tag.has_attr("lang"):
        return html_tag["lang"].split('-')[0]
    meta_lang = soup.find("meta", attrs={"http-equiv": "content-language"})
    if meta_lang and meta_lang.has_attr("content"):
        return meta_lang["content"].split('-')[0]
    meta_name_lang = soup.find("meta", attrs={"name": "language"})
    if meta_name_lang and meta_name_lang.has_attr("content"):
        return meta_name_lang["content"].split('-')[0]
    return None


def get_existing_markdown(conn, url):
    with conn.cursor() as cur:
        cur.execute("SELECT markdown_content FROM documents WHERE url = %s", (url,))
        result = cur.fetchone()
        if result:
            return result[0]
        return None


def save_document_to_db(
    conn,
    url,
    title,
    safe_title,
    crawl_date,
    lang,
    summary,
    tags,
    markdown_content,
    markdown_hash,
    recordset,
    page_type,
    no_upload,
    vector_file_id=None,
    source_config_id=None,
):
    # NOTE: 'url' is assumed to be the normalized URL.
    markdown_hash = compute_sha256(markdown_content)
    tags = normalize_tags_for_storage(tags)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents (
                url, title, safe_title, crawl_date, lang, summary, tags,
                markdown_content, markdown_hash, recordset, vector_file_id, old_file_id,
                updated_at, page_type, no_upload, is_stale, source_config_id
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                NOW(), %s, %s, FALSE, %s
            )
            ON CONFLICT (url) DO UPDATE SET
                title = EXCLUDED.title,
                safe_title = EXCLUDED.safe_title,
                crawl_date = EXCLUDED.crawl_date,
                lang = EXCLUDED.lang,
                summary = EXCLUDED.summary,
                tags = EXCLUDED.tags,
                markdown_content = EXCLUDED.markdown_content,
                markdown_hash = EXCLUDED.markdown_hash,
                recordset = EXCLUDED.recordset,
                vector_file_id = EXCLUDED.vector_file_id,
                old_file_id = documents.vector_file_id,
                updated_at = NOW(),
                page_type = EXCLUDED.page_type,
                no_upload = documents.no_upload,
                is_stale = FALSE,
                source_config_id = EXCLUDED.source_config_id
        """,
            (
                url,
                title,
                safe_title,
                crawl_date,
                lang,
                summary,
                tags,
                markdown_content,
                markdown_hash,
                recordset,
                vector_file_id,
                None,
                page_type,
                no_upload,
                source_config_id,
            ),
        )
        conn.commit()


def get_existing_markdown_hash(conn, url):
    with conn.cursor() as cur:
        cur.execute("SELECT markdown_hash FROM documents WHERE url = %s", (url,))
        result = cur.fetchone()
        return result[0] if result else None


def check_content_hash_exists(conn, content_hash):
    """Check if any document with this content hash already exists in the database"""
    with conn.cursor() as cur:
        cur.execute("SELECT url FROM documents WHERE markdown_hash = %s LIMIT 1", (content_hash,))
        result = cur.fetchone()
        return result[0] if result else None


def summarize_and_tag_tooluse(markdown_content, log_callback=None, depth=0):
    """
    Summarize and tag markdown content using OpenAI API from secrets.
    Falls back to local model if secrets are not available.
    Includes retry logic for token limit errors.
    """
    MAX_INPUT_CHARS = 10000  # Keep summaries lean for latency/token safety

    # Try with progressively smaller inputs if we hit token limits
    input_sizes = [MAX_INPUT_CHARS, 2000, 1500, 1000]

    for attempt, input_size in enumerate(input_sizes):
        truncated_content = markdown_content[:input_size]
        if attempt > 0:
            if log_callback:
                log_callback(f"{'  ' * depth}🔄 LLM retry with reduced input: {input_size} chars")

        result = _attempt_summarize_and_tag(truncated_content, log_callback, depth)
        if result is not None:
            if attempt > 0 and log_callback:
                log_callback(f"{'  ' * depth}✅ LLM processing succeeded with {input_size} characters (attempt {attempt + 1})")
            return result
        elif attempt < len(input_sizes) - 1:
            if log_callback:
                log_callback(f"{'  ' * depth}⚠️ LLM failed with {input_size} chars, trying smaller input...")
            continue
        else:
            if log_callback:
                log_callback(f"{'  ' * depth}❌ LLM: All retry attempts failed")
            return None


def _attempt_summarize_and_tag(markdown_content, log_callback=None, depth=0):
    """
    Single attempt at summarizing and tagging content.
    Returns None if token limit is hit, allowing for retry with smaller input.
    """

    if PROMPT_LOAD_ERROR:
        if log_callback:
            log_callback(f"{'  ' * depth}❌ {PROMPT_LOAD_ERROR}")
        st.error(PROMPT_LOAD_ERROR)
        raise RuntimeError(PROMPT_LOAD_ERROR)

    # Try to get OpenAI credentials from secrets
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        model = st.secrets.get("EVAL_MODEL", "gpt-4o-mini")
        use_openai = True
    except KeyError:
        # Fall back to local model
        api_key = "1234"
        model = "openai/gpt-oss-20b"
        api_url = "http://localhost:1234/v1/chat/completions"
        use_openai = False

    tools = [
        {
            "type": "function",
            "function": {
                "name": "summarize_and_tag",
                "description": (
                    "Summarizes a document, provides tags, and detects the language. "
                    "The 'detected_language' must be the two-letter ISO 639-1 code for the primary language of the markdown content."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Relevant tags for the document, must be a JSON array (e.g., [\"tag1\", \"tag2\"]). Never return as a string or with single quotes.",
                        },
                        "detected_language": {
                            "type": "string",
                            "description": "Two-letter ISO 639-1 code for the main language of the markdown content(e.g. 'de', 'en', 'fr').",
                        },
                    },
                    "required": ["summary", "tags", "detected_language"],
                },
            },
        }
    ]

    try:
        user_message = SUMMARIZE_USER_TEMPLATE.format(markdown_content=markdown_content)
    except KeyError:
        # Fallback in case the template itself contains stray braces
        user_message = f"{SUMMARIZE_USER_TEMPLATE}\n\n{markdown_content}"

    messages = [
        {"role": "system", "content": SUMMARIZE_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    print(f"[LLM] Starting analysis with {len(markdown_content)} characters...")

    try:
        if use_openai:
            # Use OpenAI client
            client = OpenAI(api_key=api_key)

            if log_callback:
                log_callback(f"{'  ' * depth}🤖 Calling OpenAI API ({model})...")

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="required",
                # temperature removed - model only supports default value (1)
                max_completion_tokens=4096,  # Increased from 512 to handle longer content
            )

            if log_callback:
                log_callback(f"{'  ' * depth}📡 Response received from OpenAI")

            # Add proper null checks for response structure
            if not response or not response.choices or len(response.choices) == 0:
                if log_callback:
                    log_callback(f"{'  ' * depth}❌ Invalid response from OpenAI API")
                return None

            choice = response.choices[0]

            if not choice or not choice.message:
                if log_callback:
                    log_callback(f"{'  ' * depth}❌ Invalid message in OpenAI response")
                return None

            message = choice.message
            tool_calls = message.tool_calls
            if tool_calls and len(tool_calls) > 0 and tool_calls[0].function:
                arguments_json = tool_calls[0].function.arguments
                try:
                    result = json.loads(arguments_json)
                    if log_callback:
                        summary = result.get('summary', '')
                        tags = result.get('tags', [])
                        lang = result.get('detected_language', 'unknown')
                        log_callback(f"{'  ' * depth}✅ LLM analysis complete - Summary: {len(summary)} chars, Tags: {len(tags)}, Language: {lang}")
                    return result
                except json.JSONDecodeError:
                    if log_callback:
                        log_callback(f"{'  ' * depth}❌ Failed to parse LLM response JSON")
                    return None
            else:
                if log_callback:
                    log_callback(f"{'  ' * depth}❌ No valid tool_calls in OpenAI response")
                return None
        else:
            # Fall back to local/other model using requests
            if log_callback:
                log_callback(f"{'  ' * depth}🤖 Using local/other model: {model}")
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
            data = {
                "model": model,
                "messages": messages,
                "tools": tools,
                "tool_choice": "required",
                "temperature": 0.7,
                "max_tokens": 512,  # Local models typically still use max_tokens
            }
            response = requests.post(api_url, json=data, headers=headers)
            response.raise_for_status()

            resp_json = response.json()
            tool_calls = resp_json['choices'][0]['message'].get('tool_calls', [])
            if tool_calls and 'function' in tool_calls[0] and 'arguments' in tool_calls[0]['function']:
                arguments_json = tool_calls[0]['function']['arguments']
                try:
                    result = json.loads(arguments_json)
                    if log_callback:
                        log_callback(f"{'  ' * depth}✅ Local model analysis complete")
                    return result
                except json.JSONDecodeError:
                    if log_callback:
                        log_callback(f"{'  ' * depth}❌ Failed to decode local model response")
                    return None
            else:
                if log_callback:
                    log_callback(f"{'  ' * depth}❌ No valid tool_calls in local model response")
                return None

    except Exception as e:
        error_message = str(e)
        if log_callback:
            log_callback(f"{'  ' * depth}❌ LLM API error: {str(e)[:100]}")

        # Check if this is a token limit error - return None to trigger retry with smaller input
        if use_openai and ("max_tokens" in error_message or "output limit" in error_message or "token" in error_message.lower()):
            if log_callback:
                log_callback(f"{'  ' * depth}⚠️ Token limit detected, will retry with smaller input")
            return None

        # For other errors, raise to stop processing
        raise


def is_link_page(markdown_content, min_links=4, max_text_blocks=2):
    links = re.findall(r'\[([^\]]+)\]\([^)]+\)', markdown_content)
    text_blocks = [line for line in markdown_content.splitlines() if len(line.strip()) > 100 and not re.match(r'^\s*[\-\*\+]\s*\[.+\]\(.+\)', line)]
    return len(links) >= min_links and len(text_blocks) <= max_text_blocks


def extract_main_content(soup, depth=0):
    containers = []
    for selector in ['article', 'main', 'div#content']:
        element = soup.select_one(selector)
        if element:
            containers.append(element)

    if not containers:
        containers = [soup.body]

    unique_containers = []
    for container in containers:
        if not any(container in other.descendants for other in containers if container != other):
            unique_containers.append(container)

    tags_to_exclude = ['nav', 'header', 'footer', 'img', 'script', 'style', 'noscript', 'aside']
    classes_to_exclude = ['sidebar', 'advertisement', 'cookie-banner', 'copyright', 'slider', 'course-finder']
    ids_to_exclude = ['rechte_spalte', 'social-share-bar', 'cookie-consent', 'cookie-banner']

    unique_containers = [c for c in unique_containers if c is not None]

    for container in unique_containers:
        if not hasattr(container, "find_all"):
            continue
        for tag in container.find_all(tags_to_exclude):
            tag.decompose()
        for cls in classes_to_exclude:
            for div in container.find_all("div", class_=lambda c: c and cls in c):
                div.decompose()
        for id_ in ids_to_exclude:
            for tag in container.find_all(attrs={"id": id_}):
                tag.decompose()

    combined_html = ''.join(str(c) for c in unique_containers)
    return BeautifulSoup(combined_html, 'html.parser')


def normalize_text(text):
    text = text.strip()
    text = re.sub(r'[ \t]+', ' ', text)
    return text


def is_page_not_found(text):
    text = text.strip().lower()
    # Extended 404 detection patterns
    error_patterns = [
        "page not found", "seite nicht gefunden", "404-fehler", "# page not found",
        "404 error", "not found", "page does not exist", "seite existiert nicht",
        "fehler 404", "error 404", "diese seite wurde nicht gefunden",
        "the requested page could not be found", "die angeforderte seite wurde nicht gefunden",
        "sorry, this page doesn't exist", "entschuldigung, diese seite existiert nicht",
        "oops! page not found", "ups! seite nicht gefunden",
        "the page you are looking for", "die seite, die sie suchen"
    ]

    # Check for multiple patterns to increase confidence
    matches = sum(1 for pattern in error_patterns if pattern in text)

    # Also check for very short content that just says "not found" or similar
    if len(text.strip()) < 100 and any(pattern in text for pattern in ["not found", "404", "nicht gefunden"]):
        return True

    # Require at least one clear error pattern
    return matches > 0


def verify_url_deleted(url: str, log_callback=None) -> tuple[bool, str]:
    """Return (is_deleted, reason) for a URL by checking HTTP status/content."""
    def _log(msg: str):
        if log_callback:
            log_callback(msg)

    head_resp = None
    try:
        head_resp = requests.head(url, headers=HEADERS, allow_redirects=True, timeout=10)
    except requests.RequestException as exc:
        _log(f"⚠️ HEAD check failed for {url}: {exc}")

    if head_resp is not None:
        status = head_resp.status_code
        if status in (404, 410):
            return True, f"HTTP {status}"
        if status >= 400 and status not in (405, 501):
            # Other client/server errors – treat as reachable but note status
            return False, f"HTTP {status}"

    try:
        response = requests.get(url, headers=HEADERS, timeout=15, allow_redirects=True)
    except requests.RequestException as exc:
        _log(f"⚠️ GET check failed for {url}: {exc}")
        return False, f"verification failed: {exc}"

    status = response.status_code
    if status in (404, 410):
        return True, f"HTTP {status}"
    if status >= 400:
        return False, f"HTTP {status}"

    try:
        text = BeautifulSoup(response.text, 'html.parser').get_text(separator=' ', strip=True)
    except Exception as exc:
        _log(f"⚠️ Could not parse HTML from {url}: {exc}")
        return False, "parse error"

    if is_page_not_found(text):
        return True, "page content indicates removal"

    return False, "reachable"


def is_empty_markdown(markdown, min_length=50):
    if not markdown or not markdown.strip():
        return True
    if len(markdown.strip()) < min_length:
        return True
    if all(phrase in markdown.lower() for phrase in ["impressum", "kontakt", "sitemap"]):
        return True
    return False


def normalize_path_prefix(value: str) -> str:
    """Normalize include/exclude patterns to '/path' form, preserving wildcard markers."""
    if value is None:
        return ""
    cleaned = value.strip()
    if not cleaned:
        return ""
    prefixed = "/" + cleaned.lstrip("/")
    if prefixed != "/" and prefixed.endswith("/"):
        prefixed = prefixed[:-1]
    return prefixed


def path_matches_prefix(path: str, pattern: str) -> bool:
    """Return True when `path` matches `pattern` (supports '*' wildcards)."""
    if not pattern:
        return False
    if pattern == "/":
        return True
    if "*" in pattern or "?" in pattern:
        # Also test with a trailing slash for directory-style globs like '/foo/*'
        return fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(f"{path}/", pattern)
    return path == pattern or path.startswith(pattern)


def scrape(
    url,
    depth,
    max_depth,
    recordset,
    source_config_id: int | None = None,
    conn=None,
    exclude_paths=None,
    include_lang_prefixes=None,
    keep_query_keys: set[str] | None = None,
    max_urls_per_run: int = 5000,
    dry_run: bool = False,
    progress_callback: Callable | None = None,
    log_callback: Callable | None = None,
    *,
    state: CrawlerState | None = None,
    headers: dict | None = None,
):
    """
    Crawl starting at `url` using a queue-based BFS, with global per-run dedupe.
    Normalizes URLs (drops fragments, strips default ports, lowercases host, removes unwanted query params),
    respects canonical URLs, enforces path filters, and guards against non-HTML + runaway crawls.

    If dry_run=True: no LLM calls and no DB writes; collects 'frontier_seen' for UI display.
    """
    state = state or _default_state
    headers = headers or HEADERS

    keep_query_keys = keep_query_keys or None
    recordset_key = (recordset or "").strip()
    normalized_exclude_paths = [prefix for prefix in (normalize_path_prefix(p) for p in (exclude_paths or [])) if prefix]
    normalized_include_prefixes = [prefix for prefix in (normalize_path_prefix(p) for p in (include_lang_prefixes or [])) if prefix]

    start_norm = normalize_url(url, "", keep_query=keep_query_keys)
    start_base_path = compute_base_path(start_norm)

    def _log(msg: str, level: str = "INFO"):
        if log_callback:
            log_callback(msg)
        _log_event(level.lower(), normalized_url=start_norm, recordset=recordset_key, msg=msg)

    def _progress():
        if progress_callback:
            try:
                progress_callback()
            except Exception:
                pass

    queue = deque()

    def enqueue(target_url: str, d: int, base_path_for_seed: str):
        if d > max_depth:
            return
        norm_candidate = normalize_url(target_url, "", keep_query=keep_query_keys)
        if norm_candidate in state.visited_norm:
            _log_event("skip_visited", normalized_url=norm_candidate, depth=d, recordset=recordset_key)
            return
        if len(state.visited_norm) >= max_urls_per_run:
            _log_event("skip_budget", normalized_url=norm_candidate, depth=d, recordset=recordset_key, max_urls=max_urls_per_run)
            return
        state.visited_norm.add(norm_candidate)
        state.frontier_seen.append(norm_candidate)
        queue.append((norm_candidate, d, base_path_for_seed))
        _log_event("enqueue", normalized_url=norm_candidate, depth=d, recordset=recordset_key, base_path=base_path_for_seed)

    enqueue(start_norm, 0, start_base_path)

    while queue:
        norm_url, current_depth, current_base_path = queue.popleft()
        state.base_path = current_base_path  # maintain legacy global for UI display
        visited_raw.add(norm_url)
        _log_event("dequeue", normalized_url=norm_url, depth=current_depth, recordset=recordset_key)

        if current_depth > max_depth:
            _log_event("skip_depth", normalized_url=norm_url, depth=current_depth, recordset=recordset_key)
            continue
        if len(state.visited_norm) > max_urls_per_run:
            _log_event("skip_budget", normalized_url=norm_url, depth=current_depth, recordset=recordset_key, max_urls=max_urls_per_run)
            continue

        # HEAD pre-check (advisory)
        try:
            head = requests.head(norm_url, headers=headers, allow_redirects=True, timeout=10)
            ctype = head.headers.get("Content-Type", "")
            looks_like_file = re.search(
                r"\.(pdf|docx?|xlsx?|pptx?|odt|ods|odp|rtf|txt|csv|zip|rar|7z|tar(?:\.gz)?|tgz|gz|bz2|xz|"
                r"jpg|jpeg|png|gif|svg|webp|tiff?|bmp|ico|heic|"
                r"mp4|webm|mov|mkv|avi|mp3|m4a|wav|ogg|flac|"
                r"exe|dmg|pkg|iso|apk|woff2?|ttf)$",
                urlparse(norm_url).path,
                re.I,
            )
            if looks_like_file and ctype and ("text/html" not in ctype and "application/xhtml+xml" not in ctype):
                _log_event("skip_non_html_head", normalized_url=norm_url, depth=current_depth, content_type=ctype)
                if log_callback:
                    log_callback(f"{'  ' * current_depth}🚫 Skipping non-HTML (HEAD suggests file): {norm_url} ({ctype})")
                continue
        except requests.RequestException:
            pass

        # GET the page
        try:
            response = requests.get(norm_url, headers=headers, timeout=15)
        except requests.RequestException as e:
            if log_callback:
                log_callback(f"{'  ' * current_depth}❌ Error fetching {norm_url}: {e}")
            _log_event("fetch_error", normalized_url=norm_url, depth=current_depth, error=str(e)[:200])
            continue

        status = response.status_code
        if status == 404:
            if log_callback:
                log_callback(f"{'  ' * current_depth}🚫 Skipping 404 Not Found: {norm_url}")
            _log_event("skip_http", normalized_url=norm_url, depth=current_depth, status=status)
            continue
        if status >= 400:
            if log_callback:
                log_callback(f"{'  ' * current_depth}❌ HTTP {status} error for {norm_url}")
            _log_event("skip_http", normalized_url=norm_url, depth=current_depth, status=status)
            continue
        if status >= 300 and log_callback:
            log_callback(f"{'  ' * current_depth}🔄 HTTP {status} redirect for {norm_url} (following redirect)")

        # Final content-type check
        ctype = response.headers.get("Content-Type", "")
        if "text/html" not in ctype and "application/xhtml+xml" not in ctype:
            if log_callback:
                log_callback(f"{'  ' * current_depth}🚫 Skipping non-HTML: {norm_url} ({ctype})")
            _log_event("skip_non_html", normalized_url=norm_url, depth=current_depth, content_type=ctype)
            continue

        # Check for cross-domain redirects - still process the page but don't follow links
        original_host = urlparse(norm_url).netloc
        effective_url = response.url
        effective_norm = normalize_url(effective_url, "", keep_query=keep_query_keys)
        effective_host = urlparse(effective_norm).netloc
        is_cross_domain_redirect = (effective_host != original_host)
        
        if is_cross_domain_redirect:
            if log_callback:
                log_callback(f"{'  ' * current_depth}🔀 Cross-domain redirect: {norm_url} → {effective_norm} (will save but not follow links)")
            _log_event("cross_domain_redirect", normalized_url=norm_url, redirect_to=effective_norm, 
                      original_host=original_host, redirect_host=effective_host, depth=current_depth,
                      recordset=recordset_key)
            # Update norm_url to the redirected URL for saving, but remember we crossed domains
            if effective_norm in state.visited_norm:
                if log_callback:
                    log_callback(f"{'  ' * current_depth}↩️ Cross-domain redirect target already visited: {effective_norm}")
                _log_event("skip_cross_domain_redirect_seen", normalized_url=effective_norm, from_url=norm_url, depth=current_depth)
                continue
            state.visited_norm.discard(norm_url)
            state.visited_norm.add(effective_norm)
            state.frontier_seen.append(effective_norm)
            norm_url = effective_norm
        elif effective_norm != norm_url:
            if effective_norm in state.visited_norm:
                if log_callback:
                    log_callback(f"{'  ' * current_depth}↩️ Redirect target already visited: {effective_norm}")
                _log_event("skip_redirect_seen", normalized_url=effective_norm, from_url=norm_url, depth=current_depth)
                continue
            state.visited_norm.discard(norm_url)
            state.visited_norm.add(effective_norm)
            state.frontier_seen.append(effective_norm)
            norm_url = effective_norm

        # Encoding handling
        declared_encoding = None
        try:
            soup_temp = BeautifulSoup(response.content, 'html.parser')
            meta = soup_temp.find('meta', attrs={'charset': True}) or soup_temp.find('meta', attrs={'http-equiv': 'Content-Type'})
            if meta:
                content = meta.get('charset') or meta.get('content', '')
                if 'charset=' in content:
                    declared_encoding = content.split('charset=')[-1]
        except Exception as e:
            if log_callback:
                log_callback(f"{'  ' * current_depth}⚠️ Encoding detection failed: {e}")

        response.encoding = declared_encoding or response.apparent_encoding
        soup = BeautifulSoup(response.text, 'html.parser')
        if log_callback:
            log_callback(f"{'  ' * current_depth}🕷️ Scraping: {norm_url}")

        # Respect canonical if present (but not cross-domain)
        current_host = urlparse(norm_url).netloc
        canon = get_canonical_url(soup, norm_url)
        if canon:
            canon_norm = normalize_url(canon, "", keep_query=keep_query_keys)
            canon_host = urlparse(canon_norm).netloc
            if canon_host != current_host:
                if log_callback:
                    log_callback(f"{'  ' * current_depth}🚫 Ignoring cross-domain canonical: {canon_norm}")
                _log_event("skip_cross_domain_canonical", normalized_url=norm_url, canonical=canon_norm,
                          current_host=current_host, canonical_host=canon_host, depth=current_depth)
                # Continue processing with original URL, don't skip
            elif canon_norm != norm_url:
                if canon_norm in state.visited_norm:
                    if log_callback:
                        log_callback(f"{'  ' * current_depth}🔗 Canonical already visited: {canon_norm}")
                    _log_event("skip_canonical_seen", normalized_url=canon_norm, from_url=norm_url, depth=current_depth)
                    continue
                state.visited_norm.discard(norm_url)
                state.visited_norm.add(canon_norm)
                state.frontier_seen.append(canon_norm)
                norm_url = canon_norm

        state.recordset_latest_urls[recordset_key].add(norm_url)

        # Content extraction
        main = extract_main_content(soup, current_depth)
        if main:
            title = soup.title.string.strip().lower() if soup.title and soup.title.string else ""
            if title and any(pattern in title for pattern in ["404", "not found", "nicht gefunden", "error", "fehler"]):
                if log_callback:
                    log_callback(f"{'  ' * current_depth}🚫 Skipping error page (title indicates 404): {norm_url}")
                _log_event("skip_error_title", normalized_url=norm_url, depth=current_depth)
                continue

            main_text = main.get_text(separator=' ', strip=True).lower()
            if is_page_not_found(main_text):
                if log_callback:
                    log_callback(f"{'  ' * current_depth}🚫 Skipping 'Page not found' at {norm_url}")
                _log_event("skip_404_body", normalized_url=norm_url, depth=current_depth)
                continue

            markdown = md(str(main), heading_style="ATX")
            markdown = normalize_text(markdown)
            if is_empty_markdown(markdown):
                if log_callback:
                    log_callback(f"{'  ' * current_depth}📄 Skipping {norm_url}: no meaningful content after extraction.")
                _log_event("skip_empty", normalized_url=norm_url, depth=current_depth)
                continue

            current_hash = compute_sha256(markdown)

            if dry_run:
                try:
                    temp_conn = get_connection()
                    existing_hash = get_existing_markdown_hash(temp_conn, norm_url)
                    duplicate_content_url = check_content_hash_exists(temp_conn, current_hash)
                    temp_conn.close()
                except Exception as e:
                    if log_callback:
                        log_callback(f"{'  ' * current_depth}⚠️ Could not check existing hash in dry run: {e}")
                    existing_hash = None
                    duplicate_content_url = None
            else:
                existing_hash = get_existing_markdown_hash(conn, norm_url) if conn else None
                duplicate_content_url = check_content_hash_exists(conn, current_hash) if conn else None

            if duplicate_content_url and duplicate_content_url != norm_url:
                if log_callback:
                    if dry_run:
                        log_callback(f"{'  ' * current_depth}🔄 [DRY RUN] DUPLICATE content - would skip (same as {duplicate_content_url}): {norm_url}")
                    else:
                        log_callback(f"{'  ' * current_depth}🔄 DUPLICATE content - skipping (same as {duplicate_content_url}): {norm_url}")
                _log_event("skip_duplicate", normalized_url=norm_url, duplicate_of=duplicate_content_url, depth=current_depth)
                continue

            skip_summary_and_db = (existing_hash == current_hash) if existing_hash is not None else False

            if dry_run and not skip_summary_and_db:
                state.dry_run_llm_eligible_count += 1
                if log_callback:
                    if existing_hash is None:
                        log_callback(f"{'  ' * current_depth}🤖 [DRY RUN] NEW page - would process with LLM: {norm_url}")
                    else:
                        log_callback(f"{'  ' * current_depth}🤖 [DRY RUN] CHANGED page - would process with LLM: {norm_url}")
                    log_callback(f"{'  ' * current_depth}📊 Pages that would be LLM processed so far: {state.dry_run_llm_eligible_count}")
                _progress()
            elif dry_run and skip_summary_and_db:
                if log_callback:
                    log_callback(f"{'  ' * current_depth}⏭️ [DRY RUN] UNCHANGED page - would skip LLM: {norm_url}")

            if not (skip_summary_and_db or dry_run):
                try:
                    if log_callback:
                        log_callback(f"{'  ' * current_depth}🤖 Processing with LLM: {norm_url}")
                    llm_output = summarize_and_tag_tooluse(markdown, log_callback, current_depth)
                    if llm_output is None:
                        if log_callback:
                            log_callback(f"{'  ' * current_depth}❌ Failed to get LLM output for {norm_url}, skipping save")
                        _log_event("llm_failed", normalized_url=norm_url, depth=current_depth)
                        continue

                    summary = llm_output.get("summary", "No summary available")
                    lang_from_html = get_html_lang(soup)
                    lang = (lang_from_html.lower() if lang_from_html else llm_output.get("detected_language", "unknown"))
                    tags = llm_output.get("tags", [])
                    page_type = "links" if is_link_page(markdown) else "text"

                    if isinstance(tags, str):
                        try:
                            tags = json.loads(tags)
                        except Exception:
                            tags = [t.strip().strip("'\" ") for t in tags.strip("[]").split(",")]

                    state.llm_analysis_results.append(
                        {
                            'url': norm_url,
                            'title': title,
                            'summary': summary,
                            'language': lang,
                            'tags': tags,
                            'page_type': page_type,
                        }
                    )

                    title_full = soup.title.string.strip() if soup.title else 'Untitled'
                    safe_title = re.sub(r'_+', '_', "".join(c if c.isalnum() else "_" for c in (title_full or "untitled")))[:64]
                    crawl_date = datetime.now().strftime("%Y-%m-%d")

                    if conn:
                        save_document_to_db(
                            conn,
                            norm_url,
                            title_full,
                            safe_title,
                            crawl_date,
                            lang,
                            summary,
                            tags,
                            markdown,
                            current_hash,
                            recordset,
                            page_type,
                            no_upload=False,
                            source_config_id=source_config_id,
                        )
                        if log_callback:
                            log_callback(f"{'  ' * current_depth}💾 Saved {norm_url} to database with recordset '{recordset}'")
                        state.processed_pages_count += 1
                        _progress()
                except Exception as e:
                    if log_callback:
                        log_callback(f"❌ LLM ERROR for {norm_url}: {e}")
                    st.error(f"Failed to summarize or save {norm_url}: {e}")
                    _log_event("llm_exception", normalized_url=norm_url, depth=current_depth, error=str(e)[:200])
                    continue

        # Link discovery (dedup + filters) - skip if cross-domain redirect
        if is_cross_domain_redirect:
            if log_callback:
                log_callback(f"{'  ' * current_depth}🚫 Skipping link discovery (cross-domain redirect page)")
            _log_event("skip_link_discovery_cross_domain", normalized_url=norm_url, depth=current_depth, recordset=recordset_key)
            continue
            
        base_host = urlparse(norm_url).netloc
        links_found = 0
        link_base_url = effective_url
        for a in soup.find_all('a', href=True):
            href = a['href'].strip()
            if href.startswith("#"):
                continue

            next_url = normalize_url(link_base_url, href, keep_query=keep_query_keys)
            if not next_url:
                continue

            links_found += 1

            if urlparse(next_url).netloc != base_host:
                if log_callback and current_depth <= 1:
                    log_callback(f"{'  ' * (current_depth+1)}🌐 Skipping external link: {next_url}")
                continue

            parsed_link = urlparse(next_url)
            normalized_path = '/' + parsed_link.path.lstrip('/')
            if normalized_path != '/' and normalized_path.endswith('/'):
                normalized_path = normalized_path.rstrip('/')

            if normalized_exclude_paths and any(path_matches_prefix(normalized_path, excl) for excl in normalized_exclude_paths):
                continue
            if normalized_include_prefixes and not any(path_matches_prefix(normalized_path, prefix) for prefix in normalized_include_prefixes):
                continue

            if current_base_path and current_base_path != '/' and not normalized_path.startswith(current_base_path):
                if log_callback:
                    log_callback(f"{'  ' * (current_depth+1)}🚫 Skipping {next_url} - outside base path {current_base_path}")
                continue

            if re.search(
                r"\.(pdf|docx?|xlsx?|pptx?|odt|ods|odp|rtf|txt|csv|zip|rar|7z|tar(?:\.gz)?|tgz|gz|bz2|xz|"
                r"jpg|jpeg|png|gif|svg|webp|tiff?|bmp|ico|heic|"
                r"mp4|webm|mov|mkv|avi|mp3|m4a|wav|ogg|flac|"
                r"exe|dmg|pkg|iso|apk|woff2?|ttf)$",
                parsed_link.path,
                re.IGNORECASE,
            ):
                continue

            if parsed_link.query and keep_query_keys is None:
                if re.search(r'(page|p|sort|month|year|q|search)=', parsed_link.query, re.I):
                    continue

            enqueue(next_url, current_depth + 1, current_base_path)

        if log_callback:
            log_callback(f"{'  ' * current_depth}🔗 Link discovery: processed {links_found} candidate links")
