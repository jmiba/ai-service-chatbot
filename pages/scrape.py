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

def save_document_to_db(conn, url, title, safe_title, crawl_date, lang, summary, tags, markdown_content, markdown_hash, recordset, page_type, deleted, vector_file_id=None):
    # NOTE: 'url' is assumed to be the normalized URL.
    markdown_hash = compute_sha256(markdown_content)
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO documents (
                url, title, safe_title, crawl_date, lang, summary, tags,
                markdown_content, markdown_hash, recordset, vector_file_id, old_file_id,
                updated_at, page_type, deleted
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
                deleted = EXCLUDED.deleted
        """, (
            url, title, safe_title, crawl_date, lang, summary, tags,
            markdown_content, markdown_hash, recordset, vector_file_id, None,
            page_type, deleted
        ))
        conn.commit()

def get_existing_markdown_hash(conn, url):
    with conn.cursor() as cur:
        cur.execute("SELECT markdown_hash FROM documents WHERE url = %s", (url,))
        result = cur.fetchone()
        return result[0] if result else None

def summarize_and_tag_tooluse(markdown_content):
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
            print(f"[RETRY] Attempting with reduced input size: {input_size} chars")
            
        result = _attempt_summarize_and_tag(truncated_content)
        if result is not None:
            if attempt > 0:
                print(f"[SUCCESS] LLM processing succeeded with {input_size} characters (attempt {attempt + 1})")
            return result
        elif attempt < len(input_sizes) - 1:
            print(f"[RETRY] Failed with {input_size} chars, trying smaller input...")
            continue
        else:
            print("[ERROR] All retry attempts failed")
            return None

def _attempt_summarize_and_tag(markdown_content):
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
    
    print(f"[DEBUG] Tool schema: {json.dumps(tools, indent=2)}")
    print(f"[DEBUG] Messages being sent: {json.dumps(messages, indent=2)[:500]}...")
    
    try:
        if use_openai:
            # Use OpenAI client
            client = OpenAI(api_key=api_key)
            
            print(f"[DEBUG] Calling OpenAI API with model: {model}")
            print(f"[DEBUG] Message length: {len(messages[1]['content'])}")
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="required",
                # temperature removed - model only supports default value (1)
                max_completion_tokens=2048  # Increased from 512 to handle longer content
            )
            
            print(f"[DEBUG] OpenAI Response received")
            print(f"[DEBUG] Response object type: {type(response)}")
            
            # Add proper null checks for response structure
            if not response or not response.choices or len(response.choices) == 0:
                print("[ERROR] Invalid or empty response from OpenAI API")
                print(f"[DEBUG] Response: {response}")
                return None
                
            choice = response.choices[0]
            print(f"[DEBUG] Choice object: {type(choice)}")
            print(f"[DEBUG] Finish reason: {getattr(choice, 'finish_reason', 'Unknown')}")
            
            if not choice or not choice.message:
                print("[ERROR] Invalid choice or message in OpenAI response")
                return None
                
            message = choice.message
            print(f"[DEBUG] Message content: {getattr(message, 'content', 'None')}")
            print(f"[DEBUG] Message tool_calls: {getattr(message, 'tool_calls', 'None')}")
            
            tool_calls = message.tool_calls
            if tool_calls and len(tool_calls) > 0 and tool_calls[0].function:
                print(f"[DEBUG] Found {len(tool_calls)} tool calls")
                arguments_json = tool_calls[0].function.arguments
                print(f"[DEBUG] Function arguments: {arguments_json}")
                try:
                    result = json.loads(arguments_json)
                    print(f"[DEBUG] Successfully parsed JSON: {result}")
                    return result
                except json.JSONDecodeError as e:
                    print(f"[ERROR] Failed to decode tool function arguments as JSON: {e}")
                    print(f"[DEBUG] Arguments content: {arguments_json}")
                    return None
            else:
                print("[ERROR] No valid tool_calls in OpenAI response")
                print(f"[DEBUG] Response structure: choices={len(response.choices) if response.choices else 0}")
                if choice.message:
                    print(f"[DEBUG] Message content: {getattr(choice.message, 'content', 'No content')}")
                    print(f"[DEBUG] Tool calls: {getattr(choice.message, 'tool_calls', 'No tool calls')}")
                return None
        else:
            # Fall back to local model using requests
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
                    return json.loads(arguments_json)
                except json.JSONDecodeError as e:
                    print(f"Failed to decode tool function arguments as JSON: {e}")
                    print("Arguments content:", arguments_json)
                    return None
            else:
                print("No valid tool_calls in local model response.")
                return None
                
    except Exception as e:
        error_message = str(e)
        print(f"Error calling {'OpenAI' if use_openai else 'local'} API: {e}")
        
        # Check if this is a token limit error - return None to trigger retry with smaller input
        if use_openai and ("max_tokens" in error_message or "output limit" in error_message or "token" in error_message.lower()):
            print("[TOKEN_LIMIT] Detected token limit error, returning None for retry with smaller input")
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
    return any(phrase in text for phrase in [
        "page not found", "seite nicht gefunden", "404-fehler", "# page not found"
    ])

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
           progress_callback=None):
    """
    Recursively scrape a URL, process its content, and save it to the database.
    Normalizes URLs (drops fragments, strips default ports, lowercases host, removes unwanted query params),
    dedupes by normalized URL, respects canonical, and guards against non-HTML + runaway crawls.

    If dry_run=True: no LLM calls and no DB writes; collects 'frontier_seen' for UI display.
    """
    # Normalize the start URL immediately (drop fragments, normalize query, etc.)
    norm_url = normalize_url(url, "", keep_query=keep_query_keys)
    print(f"{'  ' * depth}[DEBUG] Original URL: {url}")
    print(f"{'  ' * depth}[DEBUG] Normalized URL: {norm_url}")
    
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
        if not base_path.endswith('/'):
            base_path += '/'
        print(f"{'  ' * depth}[DEBUG] Set base path for session: {base_path}")

    # Stop recursion if max depth, already visited, or crawl budget exceeded
    if depth > max_depth:
        print(f"{'  ' * depth}[DEBUG] Skipping - depth {depth} > max_depth {max_depth}")
        return
    if len(visited_norm) >= max_urls_per_run:
        print(f"{'  ' * depth}[DEBUG] Skipping - crawl budget exceeded: {len(visited_norm)} >= {max_urls_per_run}")
        return
    if norm_url in visited_norm:
        print(f"{'  ' * depth}[DEBUG] Skipping - URL already visited: {norm_url}")
        return

    print(f"{'  ' * depth}[DEBUG] URL passed all checks, proceeding with scraping")

    # Track visited
    visited_raw.add(url)
    visited_norm.add(norm_url)
    frontier_seen.append(norm_url)

    # HEAD pre-check (best-effort)
    try:
        head = requests.head(norm_url, allow_redirects=True, timeout=10)
        ctype = head.headers.get("Content-Type", "")
        if "text/html" not in ctype and "application/xhtml+xml" not in ctype:
            print(f"{'  ' * depth}[!] Skipping non-HTML: {norm_url} ({ctype})")
            return
    except requests.RequestException:
        pass

    # GET the page
    try:
        response = requests.get(norm_url, timeout=15)
    except requests.RequestException as e:
        print(f"{'  ' * depth}[!] Error fetching {norm_url}: {e}")
        return

    # Final content-type check
    ctype = response.headers.get("Content-Type", "")
    if "text/html" not in ctype and "application/xhtml+xml" not in ctype:
        print(f"{'  ' * depth}[!] Skipping non-HTML: {norm_url} ({ctype})")
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
        print(f"{'  ' * depth}[!] Encoding detection failed: {e}")

    response.encoding = declared_encoding or response.apparent_encoding
    soup = BeautifulSoup(response.text, 'html.parser')
    print(f"{'  ' * depth}Scraping: {norm_url}")

    # Respect canonical if present
    canon = get_canonical_url(soup, norm_url)
    if canon:
        canon_norm = normalize_url(canon, "", keep_query=keep_query_keys)
        if canon_norm != norm_url:
            if canon_norm in visited_norm:
                print(f"{'  ' * depth}↳ Canonical already visited: {canon_norm}")
                return
            norm_url = canon_norm
            visited_norm.add(norm_url)
            frontier_seen.append(norm_url)

    # Content extraction
    main = extract_main_content(soup, depth)
    if main:
        main_text = main.get_text(separator=' ', strip=True).lower()
        if is_page_not_found(main_text):
            print(f"{'  ' * depth}[!] Skipping 'Page not found' at {norm_url}")
            return

        markdown = md(str(main), heading_style="ATX")
        markdown = normalize_text(markdown)
        if is_empty_markdown(markdown):
            print(f"{'  ' * depth}[!] Skipping {norm_url}: no meaningful content after extraction.")
            return

        # DB churn prevention
        existing_hash = get_existing_markdown_hash(conn, norm_url) if (conn and not dry_run) else None
        current_hash = compute_sha256(markdown)
        skip_summary_and_db = (existing_hash == current_hash) if existing_hash is not None else False

        if not (skip_summary_and_db or dry_run):
            try:
                llm_output = summarize_and_tag_tooluse(markdown)
                if llm_output is None:
                    print(f"{'  ' * depth}[!] Failed to get LLM output for {norm_url}, skipping save")
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

                title = soup.title.string.strip() if soup.title else 'Untitled'
                safe_title = re.sub(r'_+', '_', "".join(c if c.isalnum() else "_" for c in (title or "untitled")))[:64]
                crawl_date = datetime.now().strftime("%Y-%m-%d")

                if conn:
                    # IMPORTANT: we save by the normalized URL
                    save_document_to_db(conn, norm_url, title, safe_title, crawl_date, lang, summary, tags, markdown, current_hash, recordset, page_type, deleted=False)
                    print(f"{'  ' * depth}Saved {norm_url} to database with recordset '{recordset}'")
                    
                    # Increment counter for successfully processed pages
                    global processed_pages_count
                    processed_pages_count += 1
                    print(f"{'  ' * depth}[INFO] Pages processed by LLM so far: {processed_pages_count}")
                    
                    # Update UI progress if callback provided
                    if progress_callback:
                        try:
                            progress_callback()
                        except Exception:
                            pass  # Don't let UI errors break scraping
            except Exception as e:
                print("[LLM ERROR]", norm_url)
                st.error(f"Failed to summarize or save {norm_url}: {e}")
                return

    # Link discovery (dedup + filters)
    if soup is not None:
        base_host = urlparse(norm_url).netloc
        for a in soup.find_all('a', href=True):
            href = a['href'].strip()

            # Skip same-page anchors like '#section'
            if href.startswith("#"):
                continue

            # Normalize (drops fragments, unwanted queries)
            next_url = normalize_url(norm_url, href, keep_query=keep_query_keys)
            if not next_url:
                continue

            # Stay on-site
            if urlparse(next_url).netloc != base_host:
                continue

            # Include/exclude path filters
            parsed_link = urlparse(next_url)
            normalized_path = '/' + parsed_link.path.lstrip('/')
            if exclude_paths and any(excl in parsed_link.path for excl in exclude_paths):
                continue
            if include_lang_prefixes and not any(normalized_path.startswith(prefix) for prefix in include_lang_prefixes):
                continue
            
            # Restrict to base path only (stay within the starting URL's path)
            if base_path and not normalized_path.startswith(base_path):
                print(f"{'  ' * (depth+1)}[DEBUG] Skipping {next_url} - outside base path {base_path}")
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
                   progress_callback=progress_callback)

# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    # Ensure table exists before any DB operations
    create_knowledge_base_table()

    st.set_page_config(page_title="Knowledge Base", layout="wide")
    st.title("🌐 Knowledge Base")
    st.markdown("## Indexing Tool")
    
    st.info("**Path-Based Scraping**: The scraper will only follow links that are within the same path as the starting URL. "
            "For example, if you start with `/de/suche/fachinformationen/index.html`, it will only scrape pages under `/de/suche/fachinformationen/` "
            "and its subdirectories, preventing it from wandering to unrelated sections of the website.")

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

    # Crawl settings (global for a run)
    st.subheader("Crawler Settings")
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
    st.subheader("📋 URL Configurations")
    
    # Show number of configurations
    if st.session_state.url_configs:
        st.info(f"📊 **{len(st.session_state.url_configs)} configuration(s)** ready for indexing")
    else:
        st.info("➕ **No configurations yet** - Add your first URL configuration below")

    # Buttons to add/remove URL config blocks
    st.markdown("##### Configuration Management")
    st.info("💡 **Auto-save**: Configurations are automatically saved when you add/remove entries. Use the Save button after making changes to existing configurations.")
    
    # Create a placeholder for status messages
    message_placeholder = st.empty()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("➕ Add URL Configuration for Web Scraping"):
            st.session_state.url_configs.append({
                "url": "",
                "recordset": "",
                "depth": 2,
                "exclude_paths": ["/en/", "/pl/", "/_ablage-alte-www/", "/site-euv/", "/site-zwe-ikm/"],
                "include_lang_prefixes": ["/de/"]
            })
            try:
                save_url_configs(st.session_state.url_configs)
                message_placeholder.success("✅ Configuration added and saved!")
            except Exception as e:
                message_placeholder.error(f"Failed to save configuration: {e}")
            st.rerun()
    with col2:
        if st.button("➖ Remove Last URL Configuration from List"):
            if st.session_state.url_configs:
                st.session_state.url_configs.pop()
                try:
                    save_url_configs(st.session_state.url_configs)
                    message_placeholder.success("✅ Configuration removed and saved!")
                except Exception as e:
                    message_placeholder.error(f"Failed to save configuration: {e}")
                st.rerun()
    with col3:
        if st.button("💾 Save All Configurations"):
            try:
                save_url_configs(st.session_state.url_configs)
                message_placeholder.success("✅ All configurations saved to database!")
            except Exception as e:
                message_placeholder.error(f"Failed to save configurations: {e}")

    # Render each URL config with improved layout
    if st.session_state.url_configs:
        st.markdown("##### ⚙️ Configuration Details")
    for i, config in enumerate(st.session_state.url_configs):
        # Create a container for each configuration with better visual separation
        with st.expander(f"🔗 URL Configuration {i+1}" + (f" - {config.get('url', 'No URL set')[:50]}..." if config.get('url') else ""), expanded=True):
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

    # Index button section
    st.markdown("---")
    st.markdown("##### 🚀 Start Indexing")
    if st.button("📥 Index All URLs"):
        # reset per-run state
        print("[DEBUG] Clearing visited sets before scraping")
        print(f"[DEBUG] Before clear - visited_norm has {len(visited_norm)} URLs")
        visited_raw.clear()
        visited_norm.clear()
        frontier_seen.clear()
        global base_path, processed_pages_count
        base_path = None  # Reset base path for new session
        processed_pages_count = 0  # Reset processed pages counter
        print("[DEBUG] Visited sets, base path, and processed pages counter cleared")

        try:
            conn = None if dry_run else get_connection()
            
            # Create progress indicators for real-time updates
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                metrics_container = st.container()
                
            # Count total configurations to estimate progress
            total_configs = len([c for c in st.session_state.url_configs if c["url"].strip()])
            current_config = 0
            
            for config in st.session_state.url_configs:
                url = config["url"].strip()
                recordset = config["recordset"].strip()
                depth = config["depth"]
                exclude_paths = config["exclude_paths"]
                include_lang_prefixes = config["include_lang_prefixes"]
                if url:
                    current_config += 1
                    
                    # Update status
                    status_text.write(f"🔄 **Processing URL {current_config}/{total_configs}:** {url}")
                    
                    # Update progress bar
                    progress_bar.progress((current_config - 1) / total_configs)
                    
                    # Update metrics
                    with metrics_container:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("URLs Visited", len(visited_norm))
                        with col2:
                            st.metric("LLM Processed", processed_pages_count)
                        with col3:
                            st.metric("Current Config", f"{current_config}/{total_configs}")
                    
                    # Create a progress callback closure
                    def update_progress():
                        try:
                            # Update status text immediately  
                            status_text.write(f"🔄 **Processing Config {current_config}/{total_configs}:** {url} | Visited: {len(visited_norm)} | LLM Processed: {processed_pages_count}")
                            
                            # Update metrics container
                            metrics_container.empty()
                            with metrics_container.container():
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("URLs Visited", len(visited_norm))
                                with col2:
                                    st.metric("LLM Processed", processed_pages_count)
                                with col3:
                                    st.metric("Current Config", f"{current_config}/{total_configs}")
                        except Exception as e:
                            print(f"[UI UPDATE ERROR] {e}")  # Debug UI errors
                    
                    scrape(
                        url, depth=0, max_depth=depth, recordset=recordset,
                        conn=conn, exclude_paths=exclude_paths, include_lang_prefixes=include_lang_prefixes,
                        keep_query_keys=keep_query_keys, max_urls_per_run=max_urls_per_run, dry_run=dry_run,
                        progress_callback=update_progress
                    )
            
            # Final progress update
            progress_bar.progress(1.0)
            status_text.write("✅ **Completed all URL configurations**")
            
            if not dry_run:
                st.success("Indexing completed.")
            else:
                st.success("Dry run completed. No DB writes or LLM calls were performed.")
        except Exception as e:
            st.error(f"Error during crawl: {e}")
            print(f"[CRAWL ERROR] {e}")
        finally:
            if not dry_run and conn:
                conn.close()

        # Show frontier results in the UI (Dry Run visibility; also useful for normal runs)
        st.markdown("### Crawl Results (this run)")
        
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
                st.metric("Dry Run Mode", "Active", help="No LLM processing or database writes performed")
        
        st.write(f"**Progress:** {len(visited_norm)} / {max_urls_per_run} URLs visited")
        if not dry_run:
            st.write(f"**LLM Processing:** {processed_pages_count} pages successfully processed and saved to database")
            if processed_pages_count > 0:
                st.info(f"✅ {processed_pages_count} pages are ready for vector store synchronization")
        
        if frontier_seen:
            # Show a compact list (avoid overwhelming the app)
            max_show = min(len(frontier_seen), 1000)
            st.text("\n".join(frontier_seen[:max_show]))
            if len(frontier_seen) > max_show:
                st.info(f"...and {len(frontier_seen) - max_show} more.")

    # --- Show current knowledge base entries ---
    # Fetch entries for filters and display
    entries = sorted(get_kb_entries(), key=lambda entry: len(entry[1]))
    recordsets = sorted(set(entry[9] for entry in entries))
    page_types = sorted(set(entry[11] for entry in entries))

    st.markdown("## Knowledge Base Entries")

    # Add summary of pages waiting for vector sync
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM documents WHERE vector_file_id IS NULL")
            pending_vector_sync = cur.fetchone()[0]
        conn.close()
            
        if pending_vector_sync > 0:
            st.warning(f"⏳ **{pending_vector_sync} pages** are waiting for vector store synchronization. "
                      f"These pages have been processed by LLM but haven't been vectorized yet.")
        else:
            st.info("✅ All pages in the database are synchronized with the vector store.")
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
                st.metric("Total Entries", total_entries)
            with col2:
                st.metric("Filtered Results", filtered_entries)
            with col3:
                st.metric("⏳ Non-vectorized", non_vectorized_total)
            with col4:
                st.metric("✅ Vectorized", vectorized_total)
            
            st.markdown("---")
        
        for id, url, title, safe_title, crawl_date, lang, summary, tags, markdown, recordset, vector_file_id, page_type in filtered:
            tags_str = " ".join(f"#{tag}" for tag in tags)
            
            # Create status indicators
            vector_status = "✅ Vectorized" if vector_file_id else "⏳ Waiting for sync"
            vector_id_display = f"`{vector_file_id}`" if vector_file_id else "`None`"
            
            st.markdown(f"**{title or '(no title)'}** (ID {id}) - [{url}]({url}) - {vector_status} {vector_id_display} - **Page Type:** {page_type}")
            with st.expander(f"**{safe_title}.md** - {recordset} ({crawl_date}) (`{tags_str}`)\n\n**Summary:** {summary or '(no summary)'} (**Language:** {lang})"): 
                st.info(markdown or "(no content)")
                
            if st.button(f"🗑️ Delete Record {id}", key=f"delete_button_{id}"):
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
        if st.button(f"🗑️ Delete All Records in '{selected_recordset}' ({recordset_docs_count} docs)"):
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
        if st.button(f"🗑️ Delete All Records ({total_docs_count} docs)"):
            try:
                delete_docs()
                st.success("All documents have been deleted.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to delete documents: {e}")

if authenticated:
    main()
