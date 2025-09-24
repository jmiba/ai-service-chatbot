"""MCP server that exposes DBIS endpoints as tools."""

from __future__ import annotations

from typing import Any, Dict, Optional

import argparse
import os
import sys
import json

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
