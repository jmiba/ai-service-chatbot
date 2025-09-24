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


def _resolve_org_id(request_org: Optional[str]) -> str:
    org_id = request_org or DEFAULT_ORG_ID
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
    # common direct fields
    for k in ["url", "portalUrl", "portal_url", "homepage", "providerUrl", "provider_url", "link", "website"]:
        v = _norm_text(res.get(k))
        if v and v.startswith("http"):
            urls.append(v)
    # nested possibilities
    resource = res.get("resource")
    if isinstance(resource, dict):
        for k in ["url", "portalUrl", "homepage", "website", "link"]:
            v = _norm_text(resource.get(k))
            if v and v.startswith("http"):
                urls.append(v)
        links = resource.get("links")
        if isinstance(links, list):
            for lk in links:
                if isinstance(lk, dict):
                    v = _norm_text(lk.get("href") or lk.get("url"))
                    if v and v.startswith("http"):
                        urls.append(v)
                elif isinstance(lk, str) and lk.startswith("http"):
                    urls.append(lk)
    # Deduplicate, preserve order
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out


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
                # collapse whitespace, keep it short
                text = re.sub(r"\s+", " ", text).strip()
                return text
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


def _prefer_dbis_link(urls: List[str], resource_id: str, org_id: str) -> Tuple[str, List[str]]:
    # If any candidate is a DBIS page, prefer it; otherwise, add a constructed DBIS URL first
    for u in urls:
        if "dbis.ur.de" in u:
            # Put DBIS first
            reordered = [u] + [x for x in urls if x != u]
            return u, reordered
    constructed = _dbis_fallback_url(resource_id, org_id)
    if constructed not in urls:
        return constructed, [constructed] + urls
    return constructed, urls


async def _fetch_resource_detail(resource_id: str, org_id: str) -> dict[str, Any]:
    url = f"{DBIS_BASE_URL}/resource/{resource_id}/organization/{org_id}"
    try:
        data = await _fetch_json(url)
        # Ensure dict
        if not isinstance(data, dict):
            data = {"data": data}
        title = _extract_resource_title(data)
        urls = _extract_resource_urls(data)
        dbis_link, urls_pref = _prefer_dbis_link(urls, resource_id, org_id)
        desc = _extract_resource_description(data)
        item = {
            "id": resource_id,
            "title": title,
            "urls": urls_pref,
            "link": dbis_link,
            "description": desc,
            "api_url": url,
        }
        return item
    except httpx.TimeoutException:
        return {"id": resource_id, "error": "timeout", "api_url": url}
    except Exception as e:
        return {"id": resource_id, "error": str(e), "api_url": url}


def _build_markdown(items: List[dict[str, Any]]) -> str:
    lines: List[str] = []
    for it in items:
        title = _norm_text(it.get("title")) or "Untitled resource"
        link = _norm_text(it.get("link"))
        desc = _norm_text(it.get("description"))
        if link:
            line = f"- [{title}]({link})"
        else:
            line = f"- {title}"
        if desc:
            # add a short em dash + description
            line += f" — {desc}"
        lines.append(line)
    return "\n".join(lines)


@mcp.tool(
    name="dbis_top_resources",
    description=(
        "Return a concise, opinionated top list of DBIS resources for the given subject name or id. "
        "Defaults to organization from DBIS_ORGANIZATION_ID. Accepts: subject_name (str), limit (int, default 6). "
        "Prefer DBIS resource description links over vendor links and include brief descriptions when available. "
        "Do not ask follow-up questions; assume Viadrina/EUV by default."
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
        org_id = _resolve_org_id(organization_id)
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
        org_id = _resolve_org_id(organization_id)
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
        org_id = _resolve_org_id(organization_id)
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
        org_id = _resolve_org_id(organization_id)
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
