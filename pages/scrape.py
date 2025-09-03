import re
from datetime import datetime
from urllib.parse import urljoin, urlparse, urlunparse, parse_qsl, urlencode
from bs4 import BeautifulSoup
import requests
from markdownify import markdownify as md
import streamlit as st
import json
from utils import get_connection, get_kb_entries, create_knowledge_base_table, admin_authentication, render_sidebar, compute_sha256

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

def summarize_and_tag_tooluse(
    markdown_content,
    model="openai/gpt-oss-20b", 
    api_url="http://localhost:1234/v1/chat/completions",
    api_key=None
    ):
    MAX_INPUT_CHARS = 4000  # conservative for Llama-3 8B
    
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
            f"{markdown_content[:MAX_INPUT_CHARS]}"
        }
    ]
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    data = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "required",
        "temperature": 0.7,
        "max_tokens": 512
    }
    try:
        response = requests.post(api_url, json=data, headers=headers)
        response.raise_for_status()
    except requests.RequestException as e:
        print("API responded with:", getattr(e.response, "text", str(e)))
        raise
    
    resp_json = response.json()
    try:
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
            print("No valid tool_calls or arguments in response.")
            print(json.dumps(resp_json, indent=2))
            return None
    except (KeyError, IndexError, TypeError) as e:
        print(f"Malformed API response structure: {e}")
        print(json.dumps(resp_json, indent=2))
        return None

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
           dry_run: bool = False):
    """
    Recursively scrape a URL, process its content, and save it to the database.
    Normalizes URLs (drops fragments, strips default ports, lowercases host, removes unwanted query params),
    dedupes by normalized URL, respects canonical, and guards against non-HTML + runaway crawls.

    If dry_run=True: no LLM calls and no DB writes; collects 'frontier_seen' for UI display.
    """
    # Normalize the start URL immediately (drop fragments, normalize query, etc.)
    norm_url = normalize_url(url, "", keep_query=keep_query_keys)

    # Stop recursion if max depth, already visited, or crawl budget exceeded
    if depth > max_depth or len(visited_norm) >= max_urls_per_run:
        return
    if norm_url in visited_norm:
        return

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
                llm_output = summarize_and_tag_tooluse(markdown, api_key="1234")
                summary = llm_output["summary"]
                lang_from_html = get_html_lang(soup)
                lang = (lang_from_html.lower() if lang_from_html else llm_output["detected_language"])
                tags = llm_output["tags"]
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
                   dry_run=dry_run)

# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    # Ensure table exists before any DB operations
    create_knowledge_base_table()

    st.set_page_config(page_title="Knowledge Base", layout="wide")
    st.title("🌐 Knowledge Base")
    st.markdown("## Indexing Tool")

    # Initialize URL configs in session state
    if "url_configs" not in st.session_state:
        st.session_state.url_configs = []

    # Crawl settings (global for a run)
    st.subheader("Crawler Settings")
    colA, colB, colC = st.columns(3)
    with colA:
        dry_run = st.checkbox("Dry run (no DB writes, no LLM calls)", value=True,
                              help="When enabled, the crawler won't write to the database or call the LLM. It will only traverse and show which URLs would be processed.")
    with colB:
        max_urls_per_run = st.number_input("Max URLs per run (crawl budget)",
                                           min_value=100, max_value=100000, value=5000, step=100)
    with colC:
        keep_query_str = st.text_input("Whitelist query keys (comma-separated)",
                                       value="", help="Leave empty to drop ALL query params. Example: page,lang")
    keep_query_keys = set([x.strip() for x in keep_query_str.split(",") if x.strip()]) if keep_query_str else None

    st.markdown("---")

    # Buttons to add/remove URL config blocks
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ Add URL Configuration for Web Scraping"):
            st.session_state.url_configs.append({
                "url": "",
                "recordset": "",
                "depth": 0,
                "exclude_paths": ["/en/", "/pl/", "/_ablage-alte-www/", "/site-euv/", "/site-zwe-ikm/"],
                "include_lang_prefixes": ["/de/"]
            })
            st.rerun()
    with col2:
        if st.button("➖ Remove Last URL Configuration from List"):
            if st.session_state.url_configs:
                st.session_state.url_configs.pop()
                st.rerun()

    # Fetch entries for filters
    entries = sorted(get_kb_entries(), key=lambda entry: len(entry[1]))
    recordsets = sorted(set(entry[9] for entry in entries))
    page_types = sorted(set(entry[11] for entry in entries))

    # Render each URL config
    for i, config in enumerate(st.session_state.url_configs):
        st.markdown(f"### 🔗 URL Configuration {i+1}")
        st.session_state.url_configs[i]["url"] = st.text_input(
            f"Start URL {i+1}", value=config["url"], key=f"start_url_{i}"
        )

        recordset_options = [r for r in recordsets if r]
        recordset_options.append("Create a new one...")
        selected_recordset = st.selectbox(
            f"Available Recordsets {i+1}",
            options=recordset_options,
            index=recordset_options.index(config["recordset"]) if config["recordset"] in recordset_options else (len(recordset_options)-1 if config["recordset"] and config["recordset"] not in recordset_options else 0),
            key=f"recordset_select_{i}"
        )

        if selected_recordset == "Create a new one...":
            custom_recordset = st.text_input(
                f"New Recordset {i+1}",
                value=config["recordset"] if config["recordset"] not in recordset_options else "",
                key=f"recordset_custom_{i}"
            )
            st.session_state.url_configs[i]["recordset"] = custom_recordset
        else:
            st.session_state.url_configs[i]["recordset"] = selected_recordset

        st.session_state.url_configs[i]["depth"] = st.number_input(
            f"Max Scraping Depth {i+1}", min_value=0, max_value=20, value=config["depth"], step=1, key=f"depth_{i}"
        )

        st.session_state.url_configs[i]["exclude_paths"] = st.text_area(
            f"Exclude Paths {i+1} (comma-separated)", 
            value=", ".join(config.get("exclude_paths", [])), 
            key=f"exclude_paths_{i}"
        ).split(", ")

        st.session_state.url_configs[i]["include_lang_prefixes"] = st.text_area(
            f"Include Language Prefixes {i+1} (comma-separated)", 
            value=", ".join(config.get("include_lang_prefixes", [])), 
            key=f"include_lang_prefixes_{i}"
        ).split(", ")

        st.markdown("---")

    # Index button
    if st.button("📥 Index All URLs"):
        # reset per-run state
        visited_raw.clear()
        visited_norm.clear()
        frontier_seen.clear()

        try:
            conn = None if dry_run else get_connection()
            for config in st.session_state.url_configs:
                url = config["url"].strip()
                recordset = config["recordset"].strip()
                depth = config["depth"]
                exclude_paths = config["exclude_paths"]
                include_lang_prefixes = config["include_lang_prefixes"]
                if url:
                    scrape(
                        url, depth=0, max_depth=depth, recordset=recordset,
                        conn=conn, exclude_paths=exclude_paths, include_lang_prefixes=include_lang_prefixes,
                        keep_query_keys=keep_query_keys, max_urls_per_run=max_urls_per_run, dry_run=dry_run
                    )
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
        st.write(f"Visited (normalized) URLs: {len(visited_norm)} / budget {max_urls_per_run}")
        if frontier_seen:
            # Show a compact list (avoid overwhelming the app)
            max_show = min(len(frontier_seen), 1000)
            st.text("\n".join(frontier_seen[:max_show]))
            if len(frontier_seen) > max_show:
                st.info(f"...and {len(frontier_seen) - max_show} more.")

    # --- Show current knowledge base entries ---
    st.markdown("## Knowledge Base Entries")
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

    filtered = entries
    if selected_recordset != "All":
        filtered = [entry for entry in filtered if entry[9] == selected_recordset]
    if selected_page_type != "All":
        filtered = [entry for entry in filtered if entry[11] == selected_page_type]

    try:
        if not filtered:
            st.info("No entries found in the knowledge base.")
        for id, url, title, safe_title, crawl_date, lang, summary, tags, markdown, recordset, vector_file_id, page_type in filtered:
            tags_str = " ".join(f"#{tag}" for tag in tags)
            st.markdown(f"**{title or '(no title)'}** (ID {id}) - [{url}]({url}) - `{vector_file_id}` - **Page Type:** {page_type}")
            with st.expander(f"**{safe_title}.md** - {recordset} ({crawl_date}) (`{tags_str}`)\n\n**Summary:** {summary or '(no summary)'} (**Language:** {lang})"): 
                st.info(markdown or "(no content)")
                
            if st.button(f"🗑️ Delete Record {id}", key=f"delete_button_{id}"):
                try:
                    conn = get_connection()
                    with conn.cursor() as cur:
                        cur.execute("DELETE FROM documents WHERE id = %s", (id,))
                        conn.commit()
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
    if __name__ == "__main__":
        main()
