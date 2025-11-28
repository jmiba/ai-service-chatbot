"""
Citation extraction and rendering functions.
Handles OpenAI Responses API annotations (file_citation, url_citation).
"""
from __future__ import annotations

import re
import html
import json
from typing import Any, Callable
from urllib.parse import quote_plus

from .helpers import get_attr
from utils import get_recordset_label


# Regex patterns for citation processing
_LINK_RE = re.compile(r'(?<!!)\[([^\]]+)\]\((https?://[^)\s]+)\)')
_SEP_ONLY_PARENS = re.compile(
    r"\(\s*(?:[,;:/|·•\-–—]|\s|&nbsp;|&#160;)*\)"
)


def _pick_index(ann: Any, full_len: int) -> int | None:
    """
    Extract position index from an annotation object.
    Tries multiple possible field names for compatibility.
    """
    # Prefer explicit starts
    for key in ("start_index", "start", "index", "offset", "position"):
        v = ann.get(key) if isinstance(ann, dict) else getattr(ann, key, None)
        if v is not None:
            try:
                iv = int(v)
                return max(0, min(full_len, iv))
            except Exception:
                pass
    
    # Some SDKs nest values under the type key
    t = ann.get("type") if isinstance(ann, dict) else getattr(ann, "type", None)
    inner = ann.get(t) if isinstance(ann, dict) else getattr(ann, t, None)
    if isinstance(inner, dict):
        for key in ("start_index", "start", "index", "offset", "position"):
            v = inner.get(key)
            if v is not None:
                try:
                    iv = int(v)
                    return max(0, min(full_len, iv))
                except Exception:
                    pass
    return None


def strip_markdown_links_preserve_md(md_text: str) -> str:
    """
    Strip markdown links from text while preserving the anchor text.
    Skips images and fenced code blocks.
    
    Args:
        md_text: Markdown text with potential links
        
    Returns:
        Text with links removed but anchor text preserved
    """
    if not md_text:
        return md_text

    parts = re.split(r'(```.*?```)', md_text, flags=re.DOTALL)
    for i, part in enumerate(parts):
        if part.startswith("```"):
            continue

        def repl(m: re.Match) -> str:
            anchor = m.group(1)
            looks_like_url = bool(re.match(
                r'^(https?://|www\.|[A-Za-z0-9.-]+\.[A-Za-z]{2,})$', 
                anchor.strip(), re.I
            ))
            dbis_anchor = bool(re.match(r'^\s*dbis\b', anchor.strip(), re.I))
            return "" if (looks_like_url or dbis_anchor) else anchor

        safe = _LINK_RE.sub(repl, part)
        safe = re.sub(r'\(\s*\)', '', safe)
        parts[i] = safe

    return "".join(parts)


def trim_separator_artifacts(s: str) -> str:
    """
    Clean up artifacts left behind by citation processing.
    Removes empty parentheses and dangling commas.
    
    Args:
        s: Text to clean
        
    Returns:
        Cleaned text
    """
    if not s:
        return s

    # Remove empty/sep-only parenthesis groups
    s = _SEP_ONLY_PARENS.sub("", s)
    
    # Remove dangling commas before block ends
    s = re.sub(r",\s*(?=(?:$|\n|\r|</p>|</li>|</div>|</ul>|</ol>))", "", s, flags=re.IGNORECASE)
    
    # Collapse multiple spaces before punctuation
    s = re.sub(r"\s+([,.;:])", r"\1", s)

    return s


def extract_citations_from_annotations_response_dict(
    text_part: Any,
    get_connection: Callable,
    get_urls_and_titles_by_file_ids: Callable,
    cached_filename: Callable[[str], str] | None = None,
) -> tuple[dict[int, dict], list[tuple[int, int]]]:
    """
    Robust extractor for Responses API annotations (dicts or objects).
    
    Args:
        text_part: Response part like {"text": <str>, "annotations": <list>}
        get_connection: Function to get database connection
        get_urls_and_titles_by_file_ids: Function to lookup file metadata
        cached_filename: Optional function to get cached filename for file_id
    
    Returns:
        Tuple of (citation_map, placements) where:
        - citation_map: {n: {number, file_name, file_id, url, title, summary, ...}}
        - placements: [(char_index, n), ...]
    """
    full_text = get_attr(text_part, "text") or ""
    annotations = get_attr(text_part, "annotations") or []

    citation_map: dict[int, dict] = {}
    placements: list[tuple[int, int]] = []
    
    if not annotations:
        return citation_map, placements

    # Normalize each annotation into a simple dict
    norm = []
    for a in annotations:
        a_type = str(get_attr(a, "type") or "").strip().lower()

        if a_type == "file_citation":
            file_id = get_attr(a, "file_id")
            if not file_id:
                fc = get_attr(a, "file_citation")
                if isinstance(fc, dict):
                    file_id = fc.get("file_id")
            if not file_id:
                continue

            filename = get_attr(a, "filename")
            idx = _pick_index(a, len(full_text))
            if idx is None:
                idx = len(full_text)

            norm.append({
                "type": "file_citation",
                "file_id": file_id,
                "filename": filename,
                "index": idx
            })

        elif a_type in ("url_citation", "web_citation"):
            payload = a
            inner = get_attr(a, "url_citation")
            if isinstance(inner, dict):
                payload = inner

            url = get_attr(payload, "url")
            title = get_attr(payload, "title")
            summary = get_attr(payload, "summary")

            idx = _pick_index(payload, len(full_text))
            if idx is None:
                end_idx = get_attr(payload, "end_index")
                if end_idx is not None:
                    try:
                        idx = int(end_idx)
                    except Exception:
                        pass
            if idx is None:
                idx = len(full_text)

            norm.append({
                "type": "web_citation",
                "url": url,
                "title": title,
                "summary": summary,
                "index": idx
            })

    if not norm:
        return citation_map, placements

    # File citations: DB lookup
    file_norm = [n for n in norm if n["type"] == "file_citation"]
    file_ids = list({n["file_id"] for n in file_norm})
    file_data = {}
    if file_ids:
        conn = get_connection()
        file_data = get_urls_and_titles_by_file_ids(conn, file_ids)

    # Build map and placements in order of appearance
    i = 1
    for n in norm:
        if n["type"] == "file_citation":
            file_id = n["file_id"]
            filename = n["filename"]
            if not filename and cached_filename:
                filename = cached_filename(file_id)
            if not filename:
                filename = file_id
                
            idx = max(0, min(len(full_text), n["index"]))
            file_info = file_data.get(file_id, {})
            url = file_info.get("url")
            title = file_info.get("title")
            summary = file_info.get("summary")
            source_config_id = file_info.get("source_config_id")
            doc_id = file_info.get("doc_id")
            
            if not title:
                title = filename.rsplit(".", 1)[0]
            clean_title = html.escape(re.sub(r"^\d+_", "", title).replace("_", " "))
            
            citation_map[i] = {
                "number": i,
                "source": "file",
                "file_name": filename,
                "file_id": file_id,
                "url": url,
                "title": clean_title,
                "summary": summary,
                "source_config_id": source_config_id,
                "doc_id": doc_id,
            }
            placements.append((idx, i))
            i += 1
            
        elif n["type"] == "web_citation":
            idx = max(0, min(len(full_text), n["index"]))
            url = n["url"]
            title = n["title"] or url or "Web Source"
            summary = n["summary"] or ""
            clean_title = html.escape(title)
            
            citation_map[i] = {
                "number": i,
                "source": "web",
                "file_name": None,
                "file_id": None,
                "url": url,
                "title": clean_title,
                "summary": summary,
            }
            placements.append((idx, i))
            i += 1

    return citation_map, placements


def render_with_citations_by_index(
    text_html: str,
    citation_map: dict[int, dict],
    placements: list[tuple[int, int]]
) -> str:
    """
    Insert <sup>[n]</sup> at specified character indices.
    
    Args:
        text_html: The HTML text to annotate
        citation_map: Map of citation number to citation info
        placements: List of (char_index, citation_number) tuples
        
    Returns:
        Text with citation superscripts inserted
    """
    s = text_html or ""
    n = len(s)
    used_nums = set()
    
    # Sort by (index, num) and reverse so we insert at largest index first
    for idx, num in sorted(placements or [], key=lambda x: (x[0], x[1]), reverse=True):
        note = citation_map.get(num)
        if not note:
            continue
        if idx is not None and 0 <= idx <= n:
            sup = f"<sup title='Source: {note['title']}'>[{num}]</sup>"
            i = max(0, min(n, idx))
            s = s[:i] + sup + s[i:]
            used_nums.add(num)
    
    # Append any citations not referenced
    for num, note in citation_map.items():
        if num not in used_nums:
            sup = f"<sup title='Source: {note['title']}'>[{num}]</sup>"
            s += sup
    
    # Ensure space between adjacent supers
    s = re.sub(r"</sup><sup", "</sup> <sup", s)
    return s


def render_sources_list(
    citation_map: dict[int, dict],
    doc_icon_html: str = "",
    web_icon_html: str = "",
    base_url_path: str = "",
) -> str:
    """
    Render a markdown list of sources from the citation map.
    
    Args:
        citation_map: Map of citation number to citation info
        doc_icon_html: HTML for document icon
        web_icon_html: HTML for web icon
        base_url_path: Base URL path for internal document links
        
    Returns:
        Markdown formatted source list
    """
    if not citation_map:
        return ""

    def _icon_for_source(src: str) -> str:
        if src == "file":
            return doc_icon_html
        if src == "web":
            return web_icon_html
        return ""

    lines = []
    for c in citation_map.values():
        title = c["title"] or "Untitled"
        summary = html.escape((c["summary"] or "").replace("\n", " ").strip())
        badge = f"[{c['number']}]"
        url = c.get("url")

        # Get recordset label from source_config_id
        source_config_id = c.get("source_config_id")
        rs_text = get_recordset_label(source_config_id)
        rs_html = html.escape(rs_text) if rs_text else ""

        is_internal_doc = source_config_id == 0 or (url and url.startswith("internal://"))

        # Link rendering
        safe_title = html.escape(title)
        if url and not is_internal_doc:
            if summary:
                link_part = f'<a href="{url}" title="{summary}" target="_blank" rel="noopener noreferrer">{safe_title}</a>'
            else:
                link_part = f"[{safe_title}]({url})"
        elif is_internal_doc:
            target_id = c.get("doc_id") or c.get("file_id")
            if target_id:
                param_name = "doc_id" if c.get("doc_id") else "file_id"
                base_path = base_url_path
                if base_path and not base_path.startswith("/"):
                    base_path = "/" + base_path
                viewer_path = "document_viewer"
                viewer_url = f"{base_path}/{viewer_path}?{param_name}={quote_plus(str(target_id))}"
                link_part = (
                    f'<a href="{viewer_url}" title="{summary}" target="doc-viewer" rel="noopener noreferrer">'
                    f"{safe_title}</a>"
                )
            else:
                link_part = safe_title
        else:
            link_part = safe_title

        icon_html = _icon_for_source(c.get("source"))

        parts = [f"* {badge}"]
        if icon_html:
            parts.append(icon_html)
        if rs_html:
            parts.append(f"{rs_html}:")
        parts.append(link_part)

        lines.append(" ".join(parts))

    return "\n".join(lines)


def replace_filecite_markers_with_sup(
    text: str,
    citation_map: dict[int, dict],
    placements: list[tuple[int, int]],
    annotations: list | None = None
) -> str:
    """
    Replace markers like 'fileciteturn0file1turn0file3' with <sup>[n]</sup>.
    
    Args:
        text: Text containing filecite markers
        citation_map: Map of citation number to citation info
        placements: List of (char_index, citation_number) tuples
        annotations: Optional list of annotation objects for token resolution
        
    Returns:
        Text with markers replaced by superscript citations
    """
    if not text:
        return text

    # Map file_id -> citation number
    fileid_to_num = {}
    for num, info in (citation_map or {}).items():
        fid = info.get("file_id")
        if fid:
            fileid_to_num[str(fid)] = int(num)

    def find_file_id_for_token(token: str) -> str | None:
        if not annotations:
            return None
        for a in annotations:
            for key in ("id", "marker", "token", "file_id", "filename"):
                val = get_attr(a, key)
                if val is None:
                    continue
                try:
                    sval = str(val)
                except Exception:
                    continue
                if token == sval or token in sval:
                    fid = get_attr(a, "file_id")
                    if fid:
                        return str(fid)
            # Search annotation string
            try:
                ann_str = json.dumps(a, ensure_ascii=False) if isinstance(a, (dict, list)) else str(a)
            except Exception:
                ann_str = str(a)
            if token in ann_str:
                fid = get_attr(a, "file_id")
                if fid:
                    return str(fid)
        return None

    # Ordered fallback iterator
    ordered_nums_iter = iter([num for _, num in sorted(placements or [], key=lambda x: x[0])])

    # The special Unicode char used as separator in filecite markers
    SEP = "\ue202"  # Private Use Area character
    
    def _repl(m: re.Match) -> str:
        payload = m.group(1)
        tokens = [t for t in payload.split(SEP) if t]
        parts = []
        for tok in tokens:
            num = None
            fid = find_file_id_for_token(tok)
            if fid:
                num = fileid_to_num.get(fid)
            if num is None:
                try:
                    num = next(ordered_nums_iter)
                except StopIteration:
                    num = None
            if num is None:
                parts.append("<sup>[?]</sup>")
            else:
                note = citation_map.get(num)
                if note and note.get("title"):
                    parts.append(f"<sup title='Source: {note['title']}'>[{int(num)}]</sup>")
                else:
                    parts.append(f"<sup>[{int(num)}]</sup>")
        return "".join(parts)

    # Match filecite followed by content until the next Unicode separator or end
    return re.sub(r'filecite([^\ue202]+(?:\ue202[^\ue202]+)*)', _repl, text)
