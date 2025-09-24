"""MCP server that exposes DBIS endpoints as tools."""

from __future__ import annotations

from typing import Any, Dict, Optional, List, Tuple

import argparse
import os
import sys
import json
import asyncio
import re

import httpx
from fastmcp import FastMCP, Context

DBIS_BASE_URL = "https://dbis.ur.de/api/v1"
DEFAULT_TIMEOUT = httpx.Timeout(10.0, read=30.0)

# Parse optional headers for upstream DBIS API from environment.
# Supports either JSON (e.g. '{"Authorization":"Bearer ..."}')
# or newline/semicolon separated 'Key: Value' pairs. Also supports
# DBIS_API_AUTH or DBIS_API_AUTHORIZATION to set the Authorization header.
def _parse_dbis_headers_from_env() -> Dict[str, str]:
    headers: Dict[str, str] = {}
    raw = os.getenv("DBIS_API_HEADERS") or ""
    auth = os.getenv("DBIS_API_AUTH") or os.getenv("DBIS_API_AUTHORIZATION")
    if raw:
        try:
            maybe = json.loads(raw)
            if isinstance(maybe, dict):
                headers.update({str(k): str(v) for k, v in maybe.items()})
        except Exception:
            # Fallback simple parser for 'Key: Value' lines
            parts: list[str] = []
            for chunk in raw.split("\n"):
                chunk = chunk.strip()
                if chunk:
                    parts.append(chunk)
            if not parts and ";" in raw:
                parts = [p.strip() for p in raw.split(";") if p.strip()]
            for p in parts:
                if ":" in p:
                    k, v = p.split(":", 1)
                    headers[k.strip()] = v.strip()
    if auth and "Authorization" not in headers:
        headers["Authorization"] = str(auth).strip()
    return headers

DBIS_UPSTREAM_HEADERS = _parse_dbis_headers_from_env()

mcp = FastMCP("dbis")
print("DBIS MCP server initialized.", file=sys.stderr, flush=True)

async def _fetch_json(url: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    # Ensure every upstream call uses timeouts and optional headers
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, headers=(DBIS_UPSTREAM_HEADERS or None)) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


DEFAULT_ORG_ID = os.getenv("DBIS_ORGANIZATION_ID")


def _get_header_case_insensitive(headers: Any, key: str) -> Optional[str]:
    try:
        if isinstance(headers, dict):
            # common variants
            return (
                headers.get(key)
                or headers.get(key.lower())
                or headers.get(key.upper())
            )
    except Exception:
        pass
    return None


def _org_from_context_headers(context: Context) -> Optional[str]:
    # Try a few likely locations for headers depending on transport/host
    for attr in ("headers", "meta", "metadata"):
        hdrs = getattr(context, attr, None)
        val = _get_header_case_insensitive(hdrs, "X-DBIS-Organization-Id")
        if val:
            return str(val).strip()
    # Some hosts expose request headers
    req = getattr(context, "request", None)
    if req is not None:
        hdrs = getattr(req, "headers", None)
        val = _get_header_case_insensitive(hdrs, "X-DBIS-Organization-Id")
        if val:
            return str(val).strip()
    return None


def _resolve_org_id(request_org: Optional[str], context: Optional[Context] = None) -> str:
    # Priority: explicit arg -> per-request header -> env default
    org_id = request_org or (_org_from_context_headers(context) if context else None) or DEFAULT_ORG_ID
    if not org_id:
        raise ValueError(
            "DBIS organization_id is required. Pass it explicitly or set DBIS_ORGANIZATION_ID env var."
        )
    return org_id


def _norm_text(v: Any) -> str:
    try:
        s = str(v or "").strip()
    except Exception:
        s = ""
    return s


def _candidate_fields(d: dict, names: List[str]) -> Optional[str]:
    for n in names:
        if n in d and _norm_text(d[n]):
            return _norm_text(d[n])
    return None


async def _get_subjects() -> List[dict[str, Any]]:
    url = f"{DBIS_BASE_URL}/subjects"
    data = await _fetch_json(url)
    # Expecting a list; fallback to []
    if isinstance(data, list):
        return data
    # Sometimes wrapped
    if isinstance(data, dict):
        for key in ("subjects", "items", "data"):
            v = data.get(key)
            if isinstance(v, list):
                return v
    return []


def _find_subject_id(subjects: List[dict[str, Any]], query: str) -> Tuple[Optional[str], Optional[str]]:
    q = _norm_text(query).lower()
    if not q:
        return None, None

    # Try direct ID use (numeric or exact id string)
    for s in subjects:
        sid = _candidate_fields(s, ["id", "subject_id", "code"]) or ""
        if sid and (q == sid.lower()):
            name = _candidate_fields(s, ["name", "title", "label"]) or sid
            return sid, name
    if q.isdigit():
        # Assume it's a subject id
        return q, None

    # Exact name/title match (case-insensitive)
    for s in subjects:
        name = _candidate_fields(s, ["name", "title", "label"]) or ""
        if name and name.lower() == q:
            sid = _candidate_fields(s, ["id", "subject_id", "code"]) or None
            return sid, name

    # Startswith match
    for s in subjects:
        name = _candidate_fields(s, ["name", "title", "label"]) or ""
        if name and name.lower().startswith(q):
            sid = _candidate_fields(s, ["id", "subject_id", "code"]) or None
            return sid, name

    # Contains match
    for s in subjects:
        name = _candidate_fields(s, ["name", "title", "label"]) or ""
        if name and q in name.lower():
            sid = _candidate_fields(s, ["id", "subject_id", "code"]) or None
            return sid, name

    return None, None


def _extract_resource_title(res: dict[str, Any]) -> str:
    return (
        _candidate_fields(res, [
            "title", "name", "label", "short_title", "shortTitle", "display_title",
        ])
        or _candidate_fields(res.get("resource", {}), ["title", "name"])  # nested fallback
        or "Untitled resource"
    )


def _extract_resource_urls(res: dict[str, Any]) -> List[str]:
    urls: List[str] = []

    def _add(u: Optional[str]):
        v = _norm_text(u)
        if v and v.startswith("http"):
            urls.append(v)

    # common direct fields
    for k in ["url", "portalUrl", "portal_url", "homepage", "providerUrl", "provider_url", "link", "website"]:
        _add(res.get(k))

    # nested possibilities under resource
    resource = res.get("resource")
    if isinstance(resource, dict):
        for k in ["url", "portalUrl", "homepage", "website", "link"]:
            _add(resource.get(k))
        links = resource.get("links")
        if isinstance(links, list):
            for lk in links:
                if isinstance(lk, dict):
                    _add(lk.get("href") or lk.get("url"))
                elif isinstance(lk, str):
                    _add(lk)

    # DBIS API often places access URLs under licenses -> accesses
    licenses = res.get("licenses")
    if isinstance(licenses, list):
        for lic in licenses:
            if not isinstance(lic, dict):
                continue
            accesses = lic.get("accesses")
            if isinstance(accesses, list):
                for acc in accesses:
                    if not isinstance(acc, dict):
                        continue
                    for key in ("accessUrl", "manualUrl", "_404Url"):
                        _add(acc.get(key))

    # api_urls may also contain useful links
    api_urls = res.get("api_urls")
    if isinstance(api_urls, list):
        for au in api_urls:
            if isinstance(au, str):
                _add(au)
            elif isinstance(au, dict):
                # in case of objects with href/url
                _add(au.get("href") or au.get("url"))

    # instructions field might embed HTML with links
    instr = res.get("instructions") or (resource.get("instructions") if isinstance(resource, dict) else None)
    if isinstance(instr, str):
        for match in re.findall(r'href=["\'](https?://[^"\']+)["\']', instr, flags=re.IGNORECASE):
            _add(match)

    # Deduplicate, preserve order
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out


def _strip_html(s: str) -> str:
    if not s:
        return s
    return re.sub(r"<[^>]+>", "", s)


def _shorten(s: str, limit: int = 160) -> str:
    s = _norm_text(s)
    if not s:
        return s
    if len(s) <= limit:
        return s
    cut = s[:limit].rsplit(" ", 1)[0]
    return cut + "…"


def _extract_resource_description(res: dict[str, Any]) -> str:
    # Try common fields for a short description/summary
    for path in [
        ("description",),
        ("short_description",),
        ("summary",),
        ("abstract",),
        ("resource", "description"),
        ("resource", "short_description"),
        ("resource", "summary"),
        ("resource", "abstract"),
    ]:
        cur: Any = res
        try:
            for key in path:
                if isinstance(cur, dict):
                    cur = cur.get(key)
                else:
                    cur = None
                    break
            text = _norm_text(cur)
            if text:
                # strip HTML, collapse whitespace, and shorten
                text = _strip_html(text)
                text = re.sub(r"\s+", " ", text).strip()
                return _shorten(text, 600)
        except Exception:
            continue
    return ""


def _dbis_fallback_url(resource_id: str, org_id: str) -> str:
    # Best-effort construction of the DBIS resource description URL
    # Known pattern in DBIS: detail page includes bib_id (org) and titel_id/resource id.
    rid = _norm_text(resource_id)
    oid = _norm_text(org_id)
    if rid and oid:
        return f"https://dbis.ur.de/dbinfo/detail.php?bib_id={oid}&titel_id={rid}"
    # Fallback to API URL if data is missing
    return f"{DBIS_BASE_URL}/resource/{rid or 'UNKNOWN'}/organization/{oid or 'UNKNOWN'}"


def _is_dbis_url(u: str) -> bool:
    return "dbis.ur.de" in (u or "")


def _derive_links(urls: List[str], resource_id: str, org_id: str) -> tuple[Optional[str], str, List[str]]:
    """Return (vendor_link, dbis_link, ordered_urls).
    - vendor_link: first non-DBIS URL if present, else None
    - dbis_link: existing DBIS URL or constructed fallback detail.php
    - ordered_urls: vendor first (if any), then DBIS, then others (deduped)
    """
    vendor_link: Optional[str] = None
    dbis_link: Optional[str] = None

    for u in urls:
        if _is_dbis_url(u) and not dbis_link:
            dbis_link = u
        if (not _is_dbis_url(u)) and not vendor_link:
            vendor_link = u

    if not dbis_link:
        dbis_link = _dbis_fallback_url(resource_id, org_id)

    ordered: List[str] = []
    if vendor_link:
        ordered.append(vendor_link)
    if dbis_link and dbis_link not in ordered:
        ordered.append(dbis_link)
    # append remaining unique
    seen = set(ordered)
    for u in urls:
        if u not in seen:
            ordered.append(u)
            seen.add(u)
    return vendor_link, dbis_link, ordered


async def _fetch_text(url: str) -> str:
    async with httpx.AsyncClient(timeout=httpx.Timeout(6.0, read=6.0), headers=(DBIS_UPSTREAM_HEADERS or None)) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.text or ""


async def _extract_vendor_from_dbis_html(dbis_detail_url: str) -> Optional[str]:
    try:
        html = await _fetch_text(dbis_detail_url)
        # find external http(s) links; prefer the first that is not dbis.ur.de
        candidates = re.findall(r'href=["\'](https?://[^"\']+)["\']', html, flags=re.IGNORECASE)
        for u in candidates:
            if not _is_dbis_url(u):
                return u
    except Exception:
        pass
    return None


async def _fetch_resource_detail(resource_id: str, org_id: str) -> dict[str, Any]:
    url = f"{DBIS_BASE_URL}/resource/{resource_id}/organization/{org_id}"
    try:
        data = await _fetch_json(url)
        # Ensure dict
        if not isinstance(data, dict):
            data = {"data": data}
        title = _extract_resource_title(data)
        urls = _extract_resource_urls(data)
        vendor_link, dbis_link, _ = _derive_links(urls, resource_id, org_id)
        # If no vendor link from API, try to extract one from the DBIS detail page HTML
        if not vendor_link and dbis_link:
            vendor_link = await _extract_vendor_from_dbis_html(dbis_link)
        desc = _extract_resource_description(data)
        # Compose a single markdown line that keeps vendor inline and DBIS as a markdown link
        parts: List[str] = [f"{vendor_link}"]
        if dbis_link:
            parts.append(f"([DBIS: {title}]({dbis_link}))")
        md_line = " ".join(parts)
        item = {
            "id": resource_id,
            "title": title,
            # Intentionally omit raw 'dbis_link' to avoid models printing it inline
            # 'md_line' provides the intended presentation
            "annotation": md_line,
            "description": desc,
        }
        return item
    except httpx.TimeoutException:
        return {"id": resource_id, "error": "timeout"}
    except Exception as e:
        return {"id": resource_id, "error": str(e)}


def _build_markdown(items: List[dict[str, Any]]) -> str:
    lines: List[str] = []
    for it in items:
        # Prefer prebuilt line to keep formatting deterministic
        if "annotation" in it and it["annotation"]:
            lines.append(str(it["annotation"]))
            continue
        title = _norm_text(it.get("title")) or "Untitled resource"
        vendor = _norm_text(it.get("vendor_link"))
        dbis = _norm_text(it.get("dbis_link"))  # may be missing by design
        parts: List[str] = [f"{vendor}"]
        if dbis:
            parts.append(f"([DBIS: ]({dbis}))")
        lines.append(" ".join(parts))
    return "\n".join(lines)


@mcp.tool(
    name="dbis_top_resources",
    description=(
        "Top DBIS resources for a subject. Params: subject_name, limit (default 6). "
        "Returns compact items with vendor_link and DBIS link; default org from env."
    ),
)
async def top_resources(
    context: Context,
    subject_name: str,
    limit: Optional[int] = 6,
    organization_id: Optional[str] = None,
    format: Optional[str] = "markdown",
) -> dict[str, Any]:
    print("[MCP] dbis_top_resources: start", file=sys.stderr, flush=True)
    try:
        org_id = _resolve_org_id(organization_id, context)
        subjects = await _get_subjects()
        sid, resolved_name = _find_subject_id(subjects, subject_name)
        if not sid:
            msg = f"Subject not found for query: {subject_name}"
            print(f"[MCP] dbis_top_resources: {msg}", file=sys.stderr, flush=True)
            return {"error": msg, "query": subject_name}

        # List resource IDs by subject
        url_ids = f"{DBIS_BASE_URL}/resourceIdsBySubject/{sid}/organization/{org_id}"
        ids_payload = await _fetch_json(url_ids)
        if isinstance(ids_payload, dict):
            resource_ids = ids_payload.get("resource_ids") or ids_payload.get("ids") or []
        else:
            resource_ids = ids_payload or []
        resource_ids = [str(r) for r in resource_ids if r is not None]
        if limit is None or not isinstance(limit, int) or limit <= 0:
            limit = 6
        resource_ids = resource_ids[:limit]

        # Fetch details concurrently
        tasks = [
            _fetch_resource_detail(rid, org_id)
            for rid in resource_ids
        ]
        items = await asyncio.gather(*tasks)

        # Build output
        md = _build_markdown(items)
        result = {
            "organization_id": org_id,
            "subject": {"id": sid, "name": resolved_name or subject_name},
            "count": len(items),
            "items": items,
        }
        if (format or "").lower() == "markdown":
            result["md"] = md
        print("[MCP] dbis_top_resources: ok", file=sys.stderr, flush=True)
        return result
    except httpx.TimeoutException:
        msg = "Timeout while fetching DBIS top resources."
        print(f"[MCP] dbis_top_resources: {msg}", file=sys.stderr, flush=True)
        return {"error": msg}
    except Exception as e:
        print(f"[MCP] dbis_top_resources: error: {e}", file=sys.stderr, flush=True)
        return {"error": str(e)}


@mcp.tool(
    name="dbis_list_resource_ids",
    description="List DBIS resource IDs for an organization."
)
async def list_resource_ids(context: Context, organization_id: Optional[str] = None) -> dict[str, Any]:
    print("[MCP] dbis_list_resource_ids: start", file=sys.stderr, flush=True)
    try:
        org_id = _resolve_org_id(organization_id, context)
        url = f"{DBIS_BASE_URL}/resourceIds/organization/{org_id}"
        data = await _fetch_json(url)
        result = {"organization_id": org_id, "resource_ids": data}
        print("[MCP] dbis_list_resource_ids: ok", file=sys.stderr, flush=True)
        return result
    except httpx.TimeoutException:
        msg = "Timeout while contacting DBIS for resource IDs."
        print(f"[MCP] dbis_list_resource_ids: {msg}", file=sys.stderr, flush=True)
        return {"error": msg}
    except Exception as e:
        print(f"[MCP] dbis_list_resource_ids: error: {e}", file=sys.stderr, flush=True)
        return {"error": str(e)}


@mcp.tool(
    name="dbis_get_resource",
    description="Fetch a single DBIS resource for a given organization."
)
async def get_resource(context: Context, resource_id: str, organization_id: Optional[str] = None) -> dict[str, Any]:
    print("[MCP] dbis_get_resource: start", file=sys.stderr, flush=True)
    try:
        org_id = _resolve_org_id(organization_id, context)
        url = f"{DBIS_BASE_URL}/resource/{resource_id}/organization/{org_id}"
        data = await _fetch_json(url)
        print("[MCP] dbis_get_resource: ok", file=sys.stderr, flush=True)
        return data
    except httpx.TimeoutException:
        msg = "Timeout while fetching DBIS resource."
        print(f"[MCP] dbis_get_resource: {msg}", file=sys.stderr, flush=True)
        return {"error": msg, "resource_id": resource_id}
    except Exception as e:
        print(f"[MCP] dbis_get_resource: error: {e}", file=sys.stderr, flush=True)
        return {"error": str(e), "resource_id": resource_id}


@mcp.tool(
    name="dbis_list_subjects",
    description="Return the list of DBIS subjects (disciplines)."
)
async def list_subjects(context: Context) -> dict[str, Any]:
    print("[MCP] dbis_list_subjects: start", file=sys.stderr, flush=True)
    try:
        url = f"{DBIS_BASE_URL}/subjects"
        data = await _fetch_json(url)
        print("[MCP] dbis_list_subjects: ok", file=sys.stderr, flush=True)
        return {"subjects": data}
    except httpx.TimeoutException:
        msg = "Timeout while fetching DBIS subjects."
        print(f"[MCP] dbis_list_subjects: {msg}", file=sys.stderr, flush=True)
        return {"error": msg}
    except Exception as e:
        print(f"[MCP] dbis_list_subjects: error: {e}", file=sys.stderr, flush=True)
        return {"error": str(e)}


@mcp.tool(
    name="dbis_list_resource_ids_by_subject",
    description="List resource IDs by subject for an organization."
)
async def list_resource_ids_by_subject(
    context: Context, subject_id: str, organization_id: Optional[str] = None
) -> dict[str, Any]:
    print("[MCP] dbis_list_resource_ids_by_subject: start", file=sys.stderr, flush=True)
    try:
        org_id = _resolve_org_id(organization_id, context)
        url = f"{DBIS_BASE_URL}/resourceIdsBySubject/{subject_id}/organization/{org_id}"
        data = await _fetch_json(url)
        result = {
            "subject_id": subject_id,
            "organization_id": org_id,
            "resource_ids": data,
        }
        print("[MCP] dbis_list_resource_ids_by_subject: ok", file=sys.stderr, flush=True)
        return result
    except httpx.TimeoutException:
        msg = "Timeout while fetching DBIS resource IDs by subject."
        print(f"[MCP] dbis_list_resource_ids_by_subject: {msg}", file=sys.stderr, flush=True)
        return {"error": msg, "subject_id": subject_id}
    except Exception as e:
        print(f"[MCP] dbis_list_resource_ids_by_subject: error: {e}", file=sys.stderr, flush=True)
        return {"error": str(e), "subject_id": subject_id}


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run the DBIS MCP server.")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http", "sse", "streamable-http"],
        default="stdio",
        help="Transport protocol to expose (default: stdio).",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("DBIS_MCP_HOST", "127.0.0.1"),
        help="Bind host when using an HTTP-based transport (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("DBIS_MCP_PORT", "8765")),
        help="Bind port when using an HTTP-based transport (default: 8765).",
    )
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Suppress the startup banner emitted by FastMCP.",
    )

    args = parser.parse_args(argv)

    transport_kwargs: Dict[str, Any] = {}
    if args.transport != "stdio":
        transport_kwargs["host"] = args.host
        transport_kwargs["port"] = args.port

    mcp.run(
        transport=args.transport,
        show_banner=not args.no_banner,
        **transport_kwargs,
    )


if __name__ == "__main__":
    main()
