import re
from datetime import datetime
from urllib.parse import urljoin, urlparse, urlunparse, parse_qsl, urlencode
from bs4 import BeautifulSoup
import requests
from markdownify import markdownify as md
import streamlit as st
import json
from openai import OpenAI
from utils import get_connection, get_kb_entries, create_knowledge_base_table, admin_authentication, render_sidebar, compute_sha256, create_url_configs_table, save_url_configs, load_url_configs, initialize_default_url_configs
from pathlib import Path
  
BASE_DIR = Path(__file__).parent.parent

SCRAPE_SVG = (BASE_DIR / "assets" / "home_storage.svg").read_text()
HEADERS = {"User-Agent": "Viadrina-Indexer/1.0 (+https://www.europa-uni.de/)"}
admin_email = st.secrets["ADMIN_EMAIL"]
HEADERS = {
    "User-Agent": (
        f"Mozilla/5.0 (compatible; Viadrina-Indexer/1.0; +mailto:{admin_email})"
    )
}

# -----------------------------
# Auth / sidebar
# -----------------------------
authenticated = admin_authentication()
render_sidebar(authenticated)

# -----------------------------
# Global crawl state (session-safe)
# -----------------------------
visited_raw = set()   # legacy tracking of the exact strings encountered (optional)
visited_norm = set()  # authoritative dedupe on normalized URLs
frontier_seen = []    # for Dry Run reporting in the UI
base_path = None      # base path to restrict scraping to
processed_pages_count = 0  # count of pages processed by LLM and saved to DB
dry_run_llm_eligible_count = 0  # count of pages that would be LLM processed in dry run mode
llm_analysis_results = []  # collect LLM analysis summaries for display in info box

# -----------------------------
# URL Normalization Utilities
# -----------------------------
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

# -----------------------------
# DB helpers (unchanged, except we upsert by normalized URL)
# -----------------------------
def delete_docs():
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM documents")
        conn.commit()
    except Exception as e:
        print(f"[DB ERROR] Failed to delete documents: {e}")
        raise e
    finally:
        cursor.close()
        conn.close()

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

def save_document_to_db(conn, url, title, safe_title, crawl_date, lang, summary, tags, markdown_content, markdown_hash, recordset, page_type, no_upload, vector_file_id=None):
    # NOTE: 'url' is assumed to be the normalized URL.
    markdown_hash = compute_sha256(markdown_content)
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO documents (
                url, title, safe_title, crawl_date, lang, summary, tags,
                markdown_content, markdown_hash, recordset, vector_file_id, old_file_id,
                updated_at, page_type, no_upload
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                NOW(), %s, %s
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
                no_upload = documents.no_upload
        """, (
            url, title, safe_title, crawl_date, lang, summary, tags,
            markdown_content, markdown_hash, recordset, vector_file_id, None,
            page_type, no_upload
        ))
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
    MAX_INPUT_CHARS = 3000  # Reduced from 4000 to give more room for output tokens
    
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
    
    # Try to get OpenAI credentials from secrets
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        model = st.secrets.get("MODEL", "gpt-4o-mini")
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
                            "description": "Relevant tags for the document, must be a JSON array (e.g., [\"tag1\", \"tag2\"]). Never return as a string or with single quotes."
                            },
                        "detected_language": {
                            "type": "string",
                            "description": "Two-letter ISO 639-1 code for the main language of the markdown content(e.g. 'de', 'en', 'fr')."
                        }
                    },
                    "required": ["summary", "tags", "detected_language"],
                }
            }
        }
    ]
    
    messages = [
        {"role": "system", "content":
            "You summarize academic/institutional markdown, add tags, and detect language. "
            "When detecting language, ignore this prompt and ONLY consider the markdown content. "
            "Return 'detected_language' as a two-letter ISO 639-1 code (e.g. 'de' for German, 'en' for English, etc.). "
            "If uncertain, return 'unknown'. Respond only using the provided tool."
        },
        {"role": "user", "content":
            "Summarize the following markdown in two to three English sentences, suggest 3–7 relevant English tags, and detect the main language of the content. "
            "IMPORTANT: Return 'detected_language' as a two-letter ISO 639-1 code based ONLY on the markdown content.\n\n"
            f"{markdown_content}"  # Content is already truncated when passed to this function
        }
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
                max_completion_tokens=2048  # Increased from 512 to handle longer content
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
                except json.JSONDecodeError as e:
                    if log_callback:
                        log_callback(f"{'  ' * depth}❌ Failed to parse LLM response JSON")
                    return None
            else:
                if log_callback:
                    log_callback(f"{'  ' * depth}❌ No valid tool_calls in OpenAI response")
                return None
        else:
            # Fall back to local model using requests
            if log_callback:
                log_callback(f"{'  ' * depth}🤖 Using local model: {model}")
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
            data = {
                "model": model,
                "messages": messages,
                "tools": tools,
                "tool_choice": "required",
                "temperature": 0.7,
                "max_tokens": 512  # Local models typically still use max_tokens
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
                except json.JSONDecodeError as e:
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

def is_empty_markdown(markdown, min_length=50):
    if not markdown or not markdown.strip():
        return True
    if len(markdown.strip()) < min_length:
        return True
    if all(phrase in markdown.lower() for phrase in ["impressum", "kontakt", "sitemap"]):
        return True
    return False

# -----------------------------
# Core scraper (with normalization + dry run)
# -----------------------------
def scrape(url,
           depth,
           max_depth,
           recordset,
           conn=None,
           exclude_paths=None,
           include_lang_prefixes=None,
           keep_query_keys: set[str] | None = None,
           max_urls_per_run: int = 5000,
           dry_run: bool = False,
           progress_callback=None,
           log_callback=None):
    """
    Recursively scrape a URL, process its content, and save it to the database.
    Normalizes URLs (drops fragments, strips default ports, lowercases host, removes unwanted query params),
    dedupes by normalized URL, respects canonical, and guards against non-HTML + runaway crawls.

    If dry_run=True: no LLM calls and no DB writes; collects 'frontier_seen' for UI display.
    """
    # Normalize the start URL immediately (drop fragments, normalize query, etc.)
    norm_url = normalize_url(url, "", keep_query=keep_query_keys)
    if log_callback:
        log_callback(f"{'  ' * depth}🔍 Original URL: {url}")
        log_callback(f"{'  ' * depth}🔗 Normalized URL: {norm_url}")
    
    # Set base path for path-restricted scraping (only on first call, depth 0)
    global base_path
    if depth == 0:
        parsed_start = urlparse(norm_url)
        # Extract directory path (remove filename if present)
        path_parts = parsed_start.path.rstrip('/').split('/')
        if path_parts[-1] and '.' in path_parts[-1]:  # Remove filename
            base_path = '/'.join(path_parts[:-1])
        else:
            base_path = '/'.join(path_parts)
        
        # For root domains, allow the entire site
        if base_path == '' or base_path == '/':
            base_path = '/'  # Allow entire site for root domains
        elif not base_path.endswith('/'):
            base_path += '/'
            
        if log_callback:
            log_callback(f"{'  ' * depth}📁 Set base path for session: {base_path}")
            if base_path == '/':
                log_callback(f"{'  ' * depth}🌐 Root domain detected - will crawl entire site")

    # Stop recursion if max depth, already visited, or crawl budget exceeded
    if depth > max_depth:
        if log_callback:
            log_callback(f"{'  ' * depth}⏹️ Skipping - depth {depth} > max_depth {max_depth}")
        return
    if len(visited_norm) >= max_urls_per_run:
        if log_callback:
            log_callback(f"{'  ' * depth}📊 Skipping - crawl budget exceeded: {len(visited_norm)} >= {max_urls_per_run}")
        return
    if norm_url in visited_norm:
        if log_callback:
            log_callback(f"{'  ' * depth}♻️ Skipping - URL already visited: {norm_url}")
        return

    if log_callback:
        log_callback(f"{'  ' * depth}✅ URL passed all checks, proceeding with scraping")

    # Track visited
    visited_raw.add(url)
    visited_norm.add(norm_url)
    frontier_seen.append(norm_url)

    # HEAD pre-check (advisory)
    try:
        head = requests.head(norm_url, headers=HEADERS, allow_redirects=True, timeout=10)
        ctype = head.headers.get("Content-Type", "")
        # Only trust HEAD to skip when URL clearly looks like a file (by extension)
        looks_like_file = re.search(
            r'\.(pdf|docx?|xlsx?|pptx?|zip|rar|tar\.gz|7z|jpg|jpeg|png|gif|svg|mp4|webm)$',
            urlparse(norm_url).path,
            re.I
        )
        if looks_like_file and ctype and ("text/html" not in ctype and "application/xhtml+xml" not in ctype):
            if log_callback:
                log_callback(f"{'  ' * depth}🚫 Skipping non-HTML (HEAD suggests file): {norm_url} ({ctype})")
            return
    except requests.RequestException:
        # No HEAD? Just proceed.
        pass

    # GET the page
    try:
        response = requests.get(norm_url, headers=HEADERS, timeout=15)
        
        # Check HTTP status code - skip 404s and other error pages
        if response.status_code == 404:
            if log_callback:
                log_callback(f"{'  ' * depth}🚫 Skipping 404 Not Found: {norm_url}")
            return
        elif response.status_code >= 400:
            if log_callback:
                log_callback(f"{'  ' * depth}❌ HTTP {response.status_code} error for {norm_url}")
            return
        elif response.status_code >= 300:
            if log_callback:
                log_callback(f"{'  ' * depth}🔄 HTTP {response.status_code} redirect for {norm_url} (following redirect)")
            # Note: requests follows redirects by default, so this is just for logging
            
    except requests.RequestException as e:
        if log_callback:
            log_callback(f"{'  ' * depth}❌ Error fetching {norm_url}: {e}")
        return

    # Final content-type check
    ctype = response.headers.get("Content-Type", "")
    if "text/html" not in ctype and "application/xhtml+xml" not in ctype:
        if log_callback:
            log_callback(f"{'  ' * depth}🚫 Skipping non-HTML: {norm_url} ({ctype})")
        return

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
            log_callback(f"{'  ' * depth}⚠️ Encoding detection failed: {e}")

    response.encoding = declared_encoding or response.apparent_encoding
    soup = BeautifulSoup(response.text, 'html.parser')
    if log_callback:
        log_callback(f"{'  ' * depth}🕷️ Scraping: {norm_url}")

    # Respect canonical if present
    canon = get_canonical_url(soup, norm_url)
    if canon:
        canon_norm = normalize_url(canon, "", keep_query=keep_query_keys)
        if canon_norm != norm_url:
            if canon_norm in visited_norm:
                if log_callback:
                    log_callback(f"{'  ' * depth}🔗 Canonical already visited: {canon_norm}")
                return
            norm_url = canon_norm
            visited_norm.add(norm_url)
            frontier_seen.append(norm_url)

    # Content extraction
    main = extract_main_content(soup, depth)
    if main:
        # Check page title for 404 indicators
        title = soup.title.string.strip().lower() if soup.title and soup.title.string else ""
        if title and any(pattern in title for pattern in ["404", "not found", "nicht gefunden", "error", "fehler"]):
            if log_callback:
                log_callback(f"{'  ' * depth}🚫 Skipping error page (title indicates 404): {norm_url}")
            return
            
        main_text = main.get_text(separator=' ', strip=True).lower()
        if is_page_not_found(main_text):
            if log_callback:
                log_callback(f"{'  ' * depth}🚫 Skipping 'Page not found' at {norm_url}")
            return

        markdown = md(str(main), heading_style="ATX")
        markdown = normalize_text(markdown)
        if is_empty_markdown(markdown):
            if log_callback:
                log_callback(f"{'  ' * depth}📄 Skipping {norm_url}: no meaningful content after extraction.")
            return

        # DB churn prevention - check even in dry run to get accurate "new/changed" count
        current_hash = compute_sha256(markdown)
        
        if dry_run:
            # For dry run, create a temporary connection to check existing hashes
            try:
                temp_conn = get_connection()
                existing_hash = get_existing_markdown_hash(temp_conn, norm_url)
                duplicate_content_url = check_content_hash_exists(temp_conn, current_hash)
                temp_conn.close()
            except Exception as e:
                if log_callback:
                    log_callback(f"{'  ' * depth}⚠️ Could not check existing hash in dry run: {e}")
                existing_hash = None
                duplicate_content_url = None
        else:
            existing_hash = get_existing_markdown_hash(conn, norm_url) if conn else None
            duplicate_content_url = check_content_hash_exists(conn, current_hash) if conn else None
            
        # Skip if this exact content already exists anywhere in the database
        if duplicate_content_url and duplicate_content_url != norm_url:
            if log_callback:
                if dry_run:
                    log_callback(f"{'  ' * depth}🔄 [DRY RUN] DUPLICATE content - would skip (same as {duplicate_content_url}): {norm_url}")
                else:
                    log_callback(f"{'  ' * depth}🔄 DUPLICATE content - skipping (same as {duplicate_content_url}): {norm_url}")
            return
            
        skip_summary_and_db = (existing_hash == current_hash) if existing_hash is not None else False

        # In dry run mode, count pages that would be LLM processed (new or changed content)
        if dry_run and not skip_summary_and_db:
            global dry_run_llm_eligible_count
            dry_run_llm_eligible_count += 1
            if existing_hash is None:
                if log_callback:
                    log_callback(f"{'  ' * depth}🤖 [DRY RUN] NEW page - would process with LLM: {norm_url}")
            else:
                if log_callback:
                    log_callback(f"{'  ' * depth}🤖 [DRY RUN] CHANGED page - would process with LLM: {norm_url}")
            if log_callback:
                log_callback(f"{'  ' * depth}📊 Pages that would be LLM processed so far: {dry_run_llm_eligible_count}")
            
            # Update UI progress to show real-time dry run metrics
            if progress_callback:
                try:
                    progress_callback()
                except Exception:
                    pass  # Don't let UI errors break scraping
        elif dry_run and skip_summary_and_db:
            if log_callback:
                log_callback(f"{'  ' * depth}⏭️ [DRY RUN] UNCHANGED page - would skip LLM: {norm_url}")

        if not (skip_summary_and_db or dry_run):
            try:
                if log_callback:
                    log_callback(f"{'  ' * depth}🤖 Processing with LLM: {norm_url}")
                llm_output = summarize_and_tag_tooluse(markdown, log_callback, depth)
                if llm_output is None:
                    if log_callback:
                        log_callback(f"{'  ' * depth}❌ Failed to get LLM output for {norm_url}, skipping save")
                    return
                    
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

                # Display LLM output in the log
                if log_callback:
                    log_callback(f"{'  ' * depth}✨ LLM Analysis Complete:")
                    log_callback(f"{'  ' * (depth+1)}📝 Summary: {summary[:100]}{'...' if len(summary) > 100 else ''}")
                    log_callback(f"{'  ' * (depth+1)}🌍 Language: {lang}")
                    log_callback(f"{'  ' * (depth+1)}🏷️ Tags: {', '.join(tags[:5])}{'...' if len(tags) > 5 else ''}")
                    log_callback(f"{'  ' * (depth+1)}📄 Type: {page_type}")

                # Collect LLM results for info box display
                global llm_analysis_results
                llm_analysis_results.append({
                    'url': norm_url,
                    'title': title,
                    'summary': summary,
                    'language': lang,
                    'tags': tags,
                    'page_type': page_type
                })

                title = soup.title.string.strip() if soup.title else 'Untitled'
                safe_title = re.sub(r'_+', '_', "".join(c if c.isalnum() else "_" for c in (title or "untitled")))[:64]
                crawl_date = datetime.now().strftime("%Y-%m-%d")

                if conn:
                    # IMPORTANT: we save by the normalized URL
                    save_document_to_db(conn, norm_url, title, safe_title, crawl_date, lang, summary, tags, markdown, current_hash, recordset, page_type, no_upload=False)
                    if log_callback:
                        log_callback(f"{'  ' * depth}💾 Saved {norm_url} to database with recordset '{recordset}'")
                    
                    # Increment counter for successfully processed pages
                    global processed_pages_count
                    processed_pages_count += 1
                    if log_callback:
                        log_callback(f"{'  ' * depth}📊 Pages processed by LLM so far: {processed_pages_count}")
                    
                    # Update UI progress if callback provided
                    if progress_callback:
                        try:
                            progress_callback()
                        except Exception:
                            pass  # Don't let UI errors break scraping
            except Exception as e:
                if log_callback:
                    log_callback(f"❌ LLM ERROR for {norm_url}: {e}")
                st.error(f"Failed to summarize or save {norm_url}: {e}")
                return

    # Link discovery (dedup + filters)
    if soup is not None:
        base_host = urlparse(norm_url).netloc
        links_found = []
        links_processed = 0
        
        for a in soup.find_all('a', href=True):
            href = a['href'].strip()
            links_found.append(href)

            # Skip same-page anchors like '#section'
            if href.startswith("#"):
                continue

            # Normalize (drops fragments, unwanted queries)
            next_url = normalize_url(norm_url, href, keep_query=keep_query_keys)
            if not next_url:
                continue
                
            links_processed += 1

            # Stay on-site
            if urlparse(next_url).netloc != base_host:
                if log_callback and depth <= 1:  # Only log for shallow depths to avoid spam
                    log_callback(f"{'  ' * (depth+1)}🌐 Skipping external link: {next_url}")
                continue

            # Include/exclude path filters
            parsed_link = urlparse(next_url)
            normalized_path = '/' + parsed_link.path.lstrip('/')
            if exclude_paths and any(excl in parsed_link.path for excl in exclude_paths):
                if log_callback and depth <= 1:
                    log_callback(f"{'  ' * (depth+1)}❌ Excluded by path filter: {next_url}")
                continue
            if include_lang_prefixes and not any(normalized_path.startswith(prefix) for prefix in include_lang_prefixes):
                if log_callback and depth <= 1:
                    log_callback(f"{'  ' * (depth+1)}🚫 Not in allowed language prefix: {next_url}")
                continue
            
            # Restrict to base path only (stay within the starting URL's path)
            # For root domains (base_path = '/'), allow all paths on the same domain
            if base_path and base_path != '/' and not normalized_path.startswith(base_path):
                if log_callback:
                    log_callback(f"{'  ' * (depth+1)}🚫 Skipping {next_url} - outside base path {base_path}")
                continue

            # Skip non-HTML resources by extension
            if re.search(r'\.(pdf|docx?|xlsx?|pptx?|zip|rar|tar\.gz|7z|jpg|jpeg|png|gif|svg|mp4|webm)$', parsed_link.path, re.IGNORECASE):
                continue

            # Avoid common query-driven loopers unless whitelisted
            if parsed_link.query and keep_query_keys is None:
                if re.search(r'(page|p|sort|month|year|q|search)=', parsed_link.query, re.I):
                    continue

            # Dedup frontier
            if next_url in visited_norm:
                continue

            scrape(next_url, depth + 1, max_depth, recordset, conn,
                   exclude_paths, include_lang_prefixes,
                   keep_query_keys=keep_query_keys,
                   max_urls_per_run=max_urls_per_run,
                   dry_run=dry_run,
                   progress_callback=progress_callback,
                   log_callback=log_callback)
        
        # Log link discovery summary
        if log_callback:
            log_callback(f"{'  ' * depth}🔗 Link discovery: Found {len(links_found)} total links, processed {links_processed} valid links")

# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    # Ensure table exists before any DB operations
    create_knowledge_base_table()

    st.set_page_config(page_title="Content Indexing", layout="wide")
    #st.title("🔧 Content Indexing & Management")
    st.markdown(
        f"""
        <h1 style="display:flex; align-items:center; gap:.5rem; margin:0;">
            {SCRAPE_SVG}
            Content Indexing
        </h1>
        """,
        unsafe_allow_html=True
    )
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Knowledge Base","Indexing Tool"])
    
    with tab2:
        st.header("Scrape webpages")
    
        st.info("**Path-Based Scraping**: The scraper will only follow links that are within the same path as the starting URL. "
                "For example, if you start with `/suche/index.html`, it will only scrape pages under `/suche/` "
                "and its subdirectories.")

        # Initialize database table for URL configs
        try:
            create_url_configs_table()
            initialize_default_url_configs()
        except Exception as e:
            st.error(f"Failed to initialize URL configurations table: {e}")

        # Initialize URL configs in session state from database
        if "url_configs" not in st.session_state:
            try:
                st.session_state.url_configs = load_url_configs()
                if not st.session_state.url_configs:
                    # If no configs in DB, start with empty list
                    st.session_state.url_configs = []
            except Exception as e:
                st.error(f"Failed to load URL configurations: {e}")
                st.session_state.url_configs = []

        # Status Dashboard - Give users immediate overview of system state
        st.markdown("---")
        st.markdown("#### 📊 System Status")
        show_sys = st.checkbox("Show system status (vector sync, configs, counts)", value=False, key="show_sys_status")
        if show_sys:
            try:
                # Get current statistics
                all_entries = get_kb_entries()
                total_pages = len(all_entries)
                
                # Count pending vector sync
                conn = get_connection()
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM documents WHERE vector_file_id IS NULL")
                    pending_sync = cur.fetchone()[0]
                conn.close()
                
                # Count configurations
                total_configs = len(st.session_state.url_configs)
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📄 Total Pages", total_pages)
                with col2:
                    sync_color = "🟢" if pending_sync == 0 else "🟡"
                    st.metric(f"{sync_color} Pending Sync", pending_sync)
                with col3:
                    config_color = "🟢" if total_configs > 0 else "🔴"
                    st.metric(f"{config_color} Configurations", total_configs)
                with col4:
                    # Vector synchronization is handled in the separate 'Vectorize' page
                    if pending_sync > 0:
                        st.info(f"⏳ There are {pending_sync} documents waiting for vector store synchronization.\n"
                                "Run the 'Vectorize' page to perform batch vector synchronization (keeps scraping responsive).")
                    else:
                        st.success("✅ All synced")
                        
                # Show status summary
                if total_pages == 0:
                    st.info("🚀 **Welcome!** Add your first URL configuration below to start indexing content.")
                elif pending_sync > 0:
                    st.warning(f"⏳ **{pending_sync} pages** are waiting for vector store synchronization.")
                else:
                    st.success(f"✅ **System healthy** - All {total_pages} pages are indexed and synchronized.")
                    
            except Exception as e:
                st.error(f"Could not load system status: {e}")

        # Crawl settings (global for a run)
        st.markdown("---")
        st.subheader("⚙️  URL Configurations")

        # Create a placeholder for status messages
        message_placeholder = st.empty()


        # Render each URL config with improved layout
        for i, config in enumerate(st.session_state.url_configs):
            # Create a container for each configuration with better visual separation
            with st.expander(f"🔗 URL Configuration {i+1}" + (f" - {config.get('url', 'No URL set')[:50]}..." if config.get('url') else ""), expanded=False):
                # Use columns for better layout
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.session_state.url_configs[i]["url"] = st.text_input(
                        f"Start URL", 
                        value=config["url"], 
                        key=f"start_url_{i}",
                        help="The scraper will only follow links within the same path as this starting URL",
                        placeholder="https://example.com/path/to/start"
                    )
                
                with col2:
                    st.session_state.url_configs[i]["depth"] = st.number_input(
                        f"Max Scraping Depth", 
                        min_value=0, max_value=20, 
                        value=config["depth"], 
                        step=1, 
                        key=f"depth_{i}",
                        help="How many levels deep to follow links"
                    )

                # Recordset selection
                entries = sorted(get_kb_entries(), key=lambda entry: len(entry[1]))
                recordsets = sorted(set(entry[9] for entry in entries))
                
                recordset_options = [r for r in recordsets if r]
                recordset_options.append("Create a new one...")
                
                selected_recordset = st.selectbox(
                    f"Available Recordsets",
                    options=recordset_options,
                    index=recordset_options.index(config["recordset"]) if config["recordset"] in recordset_options else (len(recordset_options)-1 if config["recordset"] and config["recordset"] not in recordset_options else 0),
                    key=f"recordset_select_{i}",
                    help="Choose an existing recordset or create a new one to group your scraped content"
                )

                if selected_recordset == "Create a new one...":
                    custom_recordset = st.text_input(
                        f"New Recordset Name",
                        value=config["recordset"] if config["recordset"] not in recordset_options else "",
                        key=f"recordset_custom_{i}",
                        placeholder="Enter a descriptive name for this content group"
                    )
                    st.session_state.url_configs[i]["recordset"] = custom_recordset
                else:
                    st.session_state.url_configs[i]["recordset"] = selected_recordset

                # Path filters with better layout
                col1, col2 = st.columns(2)
                
                with col1:
                    exclude_paths_str = st.text_area(
                        f"🚫 Exclude Paths (comma-separated)", 
                        value=", ".join(config.get("exclude_paths", [])), 
                        key=f"exclude_paths_{i}",
                        height=100,
                        help="Paths to exclude from scraping (e.g., /en/, /admin/, /old/)",
                        placeholder="/en/, /pl/, /_ablage-alte-www/"
                    )
                    st.session_state.url_configs[i]["exclude_paths"] = [path.strip() for path in exclude_paths_str.split(",") if path.strip()]

                with col2:
                    include_prefixes_str = st.text_area(
                        f"✅ Include Language Prefixes (comma-separated)", 
                        value=", ".join(config.get("include_lang_prefixes", [])), 
                        key=f"include_lang_prefixes_{i}",
                        height=100,
                        help="Only include paths starting with these prefixes (e.g., /de/, /fr/)",
                        placeholder="/de/, /fr/"
                    )
                    st.session_state.url_configs[i]["include_lang_prefixes"] = [prefix.strip() for prefix in include_prefixes_str.split(",") if prefix.strip()]

                # Save and Delete buttons for this configuration
                st.markdown("---")
                col_save, col_delete, col_spacer = st.columns([1, 1, 2])
                
                with col_save:
                    if st.button(f"Save Config {i+1}", icon=":material/save:", key=f"save_config_{i}", type="primary"):
                        try:
                            save_url_configs(st.session_state.url_configs)
                            st.success(f"Configuration {i+1} saved!", icon=":material/check_circle:")
                        except Exception as e:
                            st.error(f"Failed to save configuration {i+1}: {e}")
                            
                with col_delete:
                    if st.button(f"Delete Config {i+1}", icon=":material/delete:", key=f"delete_config_{i}", type="secondary"):
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
        with st.expander("**⚙️ Add URL Configuration**", expanded=True):
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
                if st.button("⚡ Add URL", disabled=not quick_url, type="primary"):
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
                        "url": quick_url,
                        "recordset": f"Quick_{quick_template}_{len(st.session_state.url_configs)+1}",
                        "depth": config["depth"],
                        "exclude_paths": config["exclude_paths"],
                        "include_lang_prefixes": config["include_lang_prefixes"]
                    })
                    
                    try:
                        save_url_configs(st.session_state.url_configs)
                        st.success(f"✅ Quick configuration added! Using {quick_template} template.")
                        st.info("💡 **Next step**: Scroll down to the 'Start Indexing' section to begin scraping!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to save quick configuration: {e}")
        
        # Global configuration management
        st.markdown("---")
        st.markdown("### 💾 Configuration Management")
        st.info("💡 **Tip**: Individual configurations auto-save when you use their save buttons. Use the buttons below for bulk operations.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**💾 Save All Changes**")
            st.caption("Save all configuration changes to the database")
            if st.button("💾 Save All Configurations", type="primary"):
                try:
                    save_url_configs(st.session_state.url_configs)
                    message_placeholder.success("✅ All configurations saved to database!")
                except Exception as e:
                    message_placeholder.error(f"Failed to save configurations: {e}")
        
        with col2:
            st.markdown("**🔄 Reset to Database**")
            st.caption("Discard unsaved changes and reload from database")
            if st.button("🔄 Reset to Saved", type="secondary"):
                try:
                    st.session_state.url_configs = load_url_configs()
                    message_placeholder.success(f"✅ Reset to saved state: {len(st.session_state.url_configs)} configurations loaded!")
                    st.rerun()
                except Exception as e:
                    message_placeholder.error(f"Failed to reload configurations: {e}")
                    
        if st.session_state.url_configs:
            st.success(f"📊 **{len(st.session_state.url_configs)} configuration(s)** ready for indexing")
        else:
            st.info("➕ **No configurations yet** - Add your first URL configuration above")


        # Index button section
        st.markdown("---")
        st.subheader("🔧 Crawler Settings")
        colA, colB, colC = st.columns(3)
        with colA:
            max_urls_per_run = st.number_input("Max URLs per run (crawl budget)",
                                            min_value=100, max_value=100000, value=5000, step=100)
        with colB:
            keep_query_str = st.text_input("Whitelist query keys (comma-separated)",
                                        value="", help="Leave empty to drop ALL query params. Example: page,lang")
        with colC:
            dry_run = st.checkbox("Dry run (no DB writes, no LLM calls)", value=True,
                                help="When enabled, the crawler won't write to the database or call the LLM. It will only traverse and show which URLs would be processed.")
        keep_query_keys = set([x.strip() for x in keep_query_str.split(",") if x.strip()]) if keep_query_str else None

        st.markdown("---")
        st.markdown("## 🚀 Start Indexing")
        
        # Pre-flight check and guidance
        if not st.session_state.url_configs:
            st.warning("⚠️ **No URLs configured yet!** Add at least one URL configuration above to start indexing.")
            st.stop()
        
        # Show what will be processed
        total_configs = len(st.session_state.url_configs)
        configured_urls = [config.get('url', 'No URL') for config in st.session_state.url_configs if config.get('url')] 

        st.markdown("**Indexing will:** Discover  and crawl pages, process content with LLM, save to knowledge base, show real-time progress")
        
        # Main action button - more prominent
        if st.button("🚀 **START INDEXING ALL URLS**", type="primary", use_container_width=True):
            if not any(config.get('url', '').strip() for config in st.session_state.url_configs):
                st.error("❌ No valid URLs found in configurations. Please add at least one URL.")
                st.stop()
            
            # Initialize log system
            if 'scrape_log' not in st.session_state:
                st.session_state.scrape_log = []
            
            def add_log(message, level="INFO"):
                """Add a message to the scrape log with timestamp"""
                from datetime import datetime
                timestamp = datetime.now().strftime("%H:%M:%S")
                log_entry = f"[{timestamp}] [{level}] {message}"
                st.session_state.scrape_log.append(log_entry)
                # Keep only last 100 log entries to prevent memory issues
                if len(st.session_state.scrape_log) > 100:
                    st.session_state.scrape_log = st.session_state.scrape_log[-100:]
            
            # Create terminal-like log display
            st.markdown("##### 📟 Processing Log")
            log_container = st.container()
            with log_container:
                # Create a fixed-height terminal-style log display
                log_display = st.empty()
                
                def update_log_display():
                    """Update the terminal display with current log"""
                    if st.session_state.scrape_log:
                        log_text = "\n".join(st.session_state.scrape_log[-20:])  # Show last 20 entries
                    else:
                        log_text = "Waiting for processing to start..."
                    
                    # Use text_area with fixed height for consistent layout
                    # Clear the container and recreate to avoid key conflicts
                    log_display.empty()
                    with log_display.container():
                        st.text_area(
                            label="",
                            value=log_text,
                            height=300,  # Fixed height prevents layout jumping
                            disabled=True,  # Read-only
                            label_visibility="collapsed"  # Hide the label
                        )
            
            # Clear log and start fresh
            st.session_state.scrape_log = []
            add_log("🚀 Starting indexing process...")
            update_log_display()
            
            # reset per-run state
            add_log("🧹 Clearing visited sets before scraping")
            add_log(f"📊 Before clear - visited_norm has {len(visited_norm)} URLs")
            visited_raw.clear()
            visited_norm.clear()
            frontier_seen.clear()
            global base_path, processed_pages_count, dry_run_llm_eligible_count, llm_analysis_results
            base_path = None  # Reset base path for new session
            processed_pages_count = 0  # Reset processed pages counter
            dry_run_llm_eligible_count = 0  # Reset dry run LLM counter
            llm_analysis_results = []  # Reset LLM analysis results
            add_log("✅ Visited sets, base path, and all counters cleared")
            update_log_display()

            try:
                conn = None if dry_run else get_connection()
                add_log(f"🔌 Database connection: {'DRY RUN (no writes)' if dry_run else 'CONNECTED'}")
                update_log_display()
                
                # Create progress indicators for real-time updates
                progress_container = st.container()
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Create fixed metric placeholders to prevent duplication
                    metrics_cols = st.columns(3)
                    with metrics_cols[0]:
                        metric_urls = st.empty()
                    with metrics_cols[1]:
                        metric_llm = st.empty()
                    with metrics_cols[2]:
                        metric_config = st.empty()
                
                def update_metrics():
                    """Update metrics without creating new containers"""
                    metric_urls.metric("URLs Visited", len(visited_norm))
                    if dry_run:
                        metric_llm.metric("New/Changed Pages", dry_run_llm_eligible_count)
                    else:
                        metric_llm.metric("LLM Processed", processed_pages_count)
                    metric_config.metric("Current Config", f"{current_config}/{total_configs}")
                
                # Count total configurations to estimate progress
                total_configs = len([c for c in st.session_state.url_configs if c["url"].strip()])
                current_config = 0
                add_log(f"📋 Found {total_configs} URL configuration(s) to process")
                update_log_display()
                
                for config in st.session_state.url_configs:
                    url = config["url"].strip()
                    recordset = config["recordset"].strip()
                    depth = config["depth"]
                    exclude_paths = config["exclude_paths"]
                    include_lang_prefixes = config["include_lang_prefixes"]
                    if url:
                        current_config += 1
                        
                        add_log(f"🔍 Starting config {current_config}/{total_configs}: {url}")
                        add_log(f"   📂 Recordset: {recordset}")
                        add_log(f"   🔢 Max depth: {depth}")
                        add_log(f"   🚫 Exclude paths: {exclude_paths}")
                        add_log(f"   ✅ Include prefixes: {include_lang_prefixes}")
                        update_log_display()
                        
                        # Update status
                        status_text.write(f"🔄 **Processing URL {current_config}/{total_configs}:** {url}")
                        
                        # Update progress bar
                        progress_bar.progress((current_config - 1) / total_configs)
                        
                        # Initialize metrics display
                        update_metrics()
                        
                        # Create a progress callback closure with logging
                        def update_progress():
                            try:
                                # Update status text immediately  
                                if dry_run:
                                    status_text.write(f"🔄 **Processing Config {current_config}/{total_configs}:** {url} | Visited: {len(visited_norm)} | New/Changed: {dry_run_llm_eligible_count}")
                                else:
                                    status_text.write(f"🔄 **Processing Config {current_config}/{total_configs}:** {url} | Visited: {len(visited_norm)} | LLM Processed: {processed_pages_count}")
                                
                                # Add log entry for significant progress
                                if len(visited_norm) % 10 == 0:  # Log every 10 URLs
                                    if dry_run:
                                        add_log(f"📈 Progress: {len(visited_norm)} URLs visited, {dry_run_llm_eligible_count} new/changed pages")
                                    else:
                                        add_log(f"📈 Progress: {len(visited_norm)} URLs visited, {processed_pages_count} LLM processed")
                                    update_log_display()
                                
                                # Update metrics using the dedicated function
                                update_metrics()
                            except Exception as e:
                                print(f"[UI UPDATE ERROR] {e}")  # Debug UI errors
                        
                        scrape(
                            url, depth=0, max_depth=depth, recordset=recordset,
                            conn=conn, exclude_paths=exclude_paths, include_lang_prefixes=include_lang_prefixes,
                            keep_query_keys=keep_query_keys, max_urls_per_run=max_urls_per_run, dry_run=dry_run,
                            progress_callback=update_progress,
                            log_callback=add_log
                        )
                        
                        add_log(f"✅ Completed config {current_config}/{total_configs}: {url}")
                        if dry_run:
                            add_log(f"   📊 URLs visited: {len(visited_norm)}, new/changed pages: {dry_run_llm_eligible_count}")
                        else:
                            add_log(f"   📊 URLs visited: {len(visited_norm)}, LLM processed: {processed_pages_count}")
                        update_log_display()
                
                # Final progress update
                progress_bar.progress(1.0)
                status_text.write("✅ **Completed all URL configurations**")
                add_log("🎉 All URL configurations completed!")
                if dry_run:
                    add_log(f"📈 Final stats: {len(visited_norm)} URLs visited, {dry_run_llm_eligible_count} new/changed pages would be LLM processed")
                else:
                    add_log(f"📈 Final stats: {len(visited_norm)} URLs visited, {processed_pages_count} pages processed by LLM")
                update_log_display()
                
                if not dry_run:
                    st.success("Indexing completed.")
                    add_log("💾 Indexing completed - data saved to database")
                else:
                    st.success("Dry run completed. No DB writes or LLM calls were performed.")
                    add_log("🧪 Dry run completed - no database writes performed")
                update_log_display()
            except Exception as e:
                st.error(f"Error during crawl: {e}")
                add_log(f"❌ ERROR during crawl: {e}", "ERROR")
                update_log_display()
                print(f"[CRAWL ERROR] {e}")
            finally:
                if not dry_run and conn:
                    conn.close()
                    add_log("🔌 Database connection closed")
                    update_log_display()

            # Show frontier results in the UI (Dry Run visibility; also useful for normal runs)
            st.markdown("### 📊 Crawl Summary")
            
            # Create metrics columns for better display
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("URLs Visited", f"{len(visited_norm)}", help="Total number of URLs that were visited and processed")
            with col2:
                st.metric("Crawl Budget", f"{max_urls_per_run}", help="Maximum number of URLs allowed per crawl session")
            with col3:
                if not dry_run:
                    st.metric("Pages Processed by LLM", f"{processed_pages_count}", 
                            help="Number of pages that were processed by LLM and saved to database (waiting for vector sync)")
                else:
                    st.metric("Would LLM Process", f"{dry_run_llm_eligible_count}", 
                            help="Number of pages that would be processed by LLM in a real run (excluding duplicates)")
            
            if not dry_run:
                if processed_pages_count > 0:
                    st.success(f"✅ {processed_pages_count} pages are ready for vector store synchronization")
                else:
                    st.info("ℹ️ No new pages were processed (all pages may have been skipped or already exist)")
            else:
                if dry_run_llm_eligible_count > 0:
                    st.info(f"🧪 **Dry Run Results**: {dry_run_llm_eligible_count} pages would be processed by LLM in a real run")
                else:
                    st.info("🧪 **Dry Run Results**: No pages would be processed by LLM (all may be duplicates or filtered out)")
            
            # Display LLM Analysis Results
            if llm_analysis_results and not dry_run:
                with st.expander(f"🤖 LLM Analysis Results ({len(llm_analysis_results)} pages)", expanded=True):
                    for i, result in enumerate(llm_analysis_results, 1):
                        st.markdown(f"**{i}. {result['title']}**")
                        st.markdown(f"🔗 *{result['url']}*")
                        st.markdown(f"📝 **Summary:** {result['summary']}")
                        st.markdown(f"🌍 **Language:** {result['language']}")
                        st.markdown(f"🏷️ **Tags:** {', '.join(result['tags'])}")
                        st.markdown(f"📄 **Type:** {result['page_type']}")
                        if i < len(llm_analysis_results):
                            st.divider()        # Improved frontier display
            if frontier_seen:
                with st.expander(f"View Processed URLs ({len(frontier_seen)} total)", expanded=False, icon=":material/visibility:"):
                    # Group URLs by domain for better readability
                    from collections import defaultdict
                    url_groups = defaultdict(list)
                    for url in frontier_seen[:1000]:  # Limit to prevent performance issues
                        domain = urlparse(url).netloc
                        url_groups[domain].append(url)
                    
                    for domain, urls in url_groups.items():
                        st.markdown(f"**{domain}** ({len(urls)} URLs)")
                        # Show first 10 URLs for each domain
                        display_urls = urls[:10]
                    for url in display_urls:
                        st.text(f"  • {url}")
                    if len(urls) > 10:
                        st.text(f"  ... and {len(urls) - 10} more URLs")
                    st.markdown("---")
                
                if len(frontier_seen) > 1000:
                    st.info(f"Showing first 1000 URLs. Total processed: {len(frontier_seen)}")

        # --- Show current knowledge base entries ---
        # Fetch entries for filters and display
        entries = sorted(get_kb_entries(), key=lambda entry: len(entry[1]))
        recordsets = sorted(set(entry[9] for entry in entries))
        page_types = sorted(set(entry[11] for entry in entries))

    with tab1:
        st.header("Browse Knowledge Base")
        st.markdown("*View, search, and manage your indexed content*")

        # Add summary of pages waiting for vector sync
        try:
            conn = get_connection()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 
                        COUNT(*) FILTER (WHERE vector_file_id IS NULL AND (no_upload IS FALSE OR no_upload IS NULL)) AS pending_sync,
                        COUNT(*) FILTER (WHERE no_upload IS TRUE AND vector_file_id IS NOT NULL) AS excluded_needing_cleanup
                    FROM documents
                    """
                )
                row = cur.fetchone()
                pending_vector_sync = row[0]
                excluded_needing_cleanup = row[1]
            conn.close()
                
            if pending_vector_sync > 0 and excluded_needing_cleanup > 0:
                st.warning(
                    f"**{pending_vector_sync} page(s)** are waiting for vector store synchronization, "
                    f"and **{excluded_needing_cleanup} excluded page(s)** still have vector files that need cleanup.", icon=":material/warning:"
                )
            elif pending_vector_sync > 0:
                st.warning(
                    f"**{pending_vector_sync} page(s)** are waiting for vector store synchronization. "
                    f"These pages have been processed by LLM but haven't been vectorized yet.", icon=":material/warning:"
                )
            elif excluded_needing_cleanup > 0:
                st.info(
                    f"**{excluded_needing_cleanup} excluded page(s)** still have vector files and need cleanup (run Vectorize).", icon=":material/info:"
                )
            else:
                st.info("All pages are synchronized and no excluded files need cleanup.", icon=":material/check_circle:")
        except Exception as e:
            st.error(f"Could not check vector sync status: {e}")

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
            options=["All", "Non-vectorized (waiting for sync)", "Vectorized (synced)"],
            index=0,
            help="Filter entries based on whether they have been vectorized and synced to the vector store"
        )
        # New: filter by exclusion from vector store (no_upload)
        selected_exclusion_status = st.selectbox(
            "Filter by vector store exclusion",
            options=["All", "Excluded", "Included"],
            index=0,
            help="Filter entries that are excluded from vectorization (no_upload flag)"
        )

        filtered = entries
        if selected_recordset != "All":
            filtered = [entry for entry in filtered if entry[9] == selected_recordset]
        if selected_page_type != "All":
            filtered = [entry for entry in filtered if entry[11] == selected_page_type]
        if selected_vector_status != "All":
            if selected_vector_status == "Non-vectorized (waiting for sync)":
                filtered = [entry for entry in filtered if entry[10] is None]  # vector_file_id is None
            elif selected_vector_status == "Vectorized (synced)":
                filtered = [entry for entry in filtered if entry[10] is not None]  # vector_file_id is not None
        # Apply exclusion filter using no_upload (index 12)
        if selected_exclusion_status != "All":
            if selected_exclusion_status == "Excluded":
                filtered = [entry for entry in filtered if len(entry) > 12 and bool(entry[12])]
            else:  # Included
                filtered = [entry for entry in filtered if len(entry) > 12 and not bool(entry[12])]

        try:
            if not filtered:
                st.info("No entries found in the knowledge base.")
            else:
                # Show count summary
                total_entries = len(entries)
                filtered_entries = len(filtered)
                non_vectorized_total = len([entry for entry in entries if entry[10] is None])
                vectorized_total = len([entry for entry in entries if entry[10] is not None])
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Entries", total_entries, border=True)
                with col2:
                    st.metric("Filtered Results", filtered_entries, border=True)
                with col3:
                    st.metric("Non-vectorized", non_vectorized_total, border=True)
                with col4:
                    st.metric("Vectorized", vectorized_total, border=True)
                
                st.markdown("---")
            
            for id, url, title, safe_title, crawl_date, lang, summary, tags, markdown, recordset, vector_file_id, page_type, no_upload in filtered:
                tags_str = " ".join(f"#{tag}" for tag in tags)
                
                # Create status indicators
                vector_status = "✅ Vectorized" if vector_file_id else "⏳ Waiting for sync"
                vector_id_display = f"`{vector_file_id}`" if vector_file_id else "`None`"
                excluded_badge = " · 🚫 Excluded" if no_upload else ""
                
                st.markdown(f"**{title or '(no title)'}** (ID {id}) - [{url}]({url}) - {vector_status} {vector_id_display} - **Page Type:** {page_type}{excluded_badge}")
                with st.expander(f"**{safe_title}.md** - {recordset} ({crawl_date}) (`{tags_str}`)\n\n**Summary:** {summary or '(no summary)'} (**Language:** {lang})"): 
                    st.info(markdown or "(no content)")
                
                col1, col2 = st.columns(2)
                with col1:
                    # Toggle include/exclude from vector store depending on current state
                    toggle_label = (
                        f"Include Record {id} in Vector Store" if no_upload else f"Exclude Record {id} from Vector Store"
                    )
                    toggle_icon = ":material/check_circle:" if no_upload else ":material/block:"
                    if st.button(toggle_label, key=f"toggle_upload_{id}", icon=toggle_icon, type="secondary"):
                        try:
                            conn = get_connection()
                            with conn.cursor() as cur:
                                cur.execute("UPDATE documents SET no_upload = %s WHERE id = %s", (not no_upload, id))
                                conn.commit()
                            conn.close()
                            if no_upload:
                                st.success(f"Record {id} included in vector store.")
                            else:
                                st.success(f"Record {id} excluded from vector store.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to update vector store inclusion for record {id}: {e}")
                with col2:    
                    if st.button(f"Delete Record {id}", key=f"delete_button_{id}", icon=":material/delete:", type="secondary"):
                        try:
                            conn = get_connection()
                            with conn.cursor() as cur:
                                cur.execute("DELETE FROM documents WHERE id = %s", (id,))
                                conn.commit()
                            conn.close()
                            st.success(f"Record {id} deleted successfully.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to delete record {id}: {e}")

        except Exception as e:
            st.error(f"Failed to load entries: {e}")
        
        # --- Delete all entries button ---
        st.markdown("### Delete All Records")
        if selected_recordset != "All":
            recordset_docs_count = len([entry for entry in filtered if entry[9] == selected_recordset])
            if st.button(f"Delete All Records in '{selected_recordset}' ({recordset_docs_count} docs)", icon =":material/delete_forever:", type="secondary"):
                try:
                    conn = get_connection()
                    with conn.cursor() as cur:
                        cur.execute("DELETE FROM documents WHERE recordset = %s", (selected_recordset,))
                        conn.commit()
                    st.success(f"All documents in recordset '{selected_recordset}' have been deleted.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to delete documents in recordset '{selected_recordset}': {e}")
        else:
            total_docs_count = len(filtered)
            if st.button(f"Delete All Records ({total_docs_count} docs)", type="secondary", icon=":material/delete_forever:"):
                try:
                    delete_docs()
                    st.success("All documents have been deleted.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to delete documents: {e}")

if authenticated:
    main()
