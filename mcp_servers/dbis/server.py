"""MCP server that exposes DBIS endpoints as tools."""

from __future__ import annotations

from typing import Any, Dict, Optional

import argparse
import os
import sys

import httpx
from fastmcp import FastMCP, Context

DBIS_BASE_URL = "https://dbis.ur.de/api/v1"
DEFAULT_TIMEOUT = httpx.Timeout(10.0, read=30.0)

mcp = FastMCP("dbis")
print("DBIS MCP server initialized.", file=sys.stderr, flush=True)

async def _fetch_json(url: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
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
    org_id = _resolve_org_id(organization_id)
    url = f"{DBIS_BASE_URL}/resourceIds/organization/{org_id}"
    data = await _fetch_json(url)
    return {"organization_id": org_id, "resource_ids": data}


@mcp.tool(
    name="dbis_get_resource",
    description="Fetch a single DBIS resource for a given organization."
)
async def get_resource(context: Context, resource_id: str, organization_id: Optional[str] = None) -> dict[str, Any]:
    org_id = _resolve_org_id(organization_id)
    url = f"{DBIS_BASE_URL}/resource/{resource_id}/organization/{org_id}"
    data = await _fetch_json(url)
    return data


@mcp.tool(
    name="dbis_list_subjects",
    description="Return the list of DBIS subjects (disciplines)."
)
async def list_subjects(context: Context) -> dict[str, Any]:
    url = f"{DBIS_BASE_URL}/subjects"
    data = await _fetch_json(url)
    return {"subjects": data}


@mcp.tool(
    name="dbis_list_resource_ids_by_subject",
    description="List resource IDs by subject for an organization."
)
async def list_resource_ids_by_subject(
    context: Context, subject_id: str, organization_id: Optional[str] = None
) -> dict[str, Any]:
    org_id = _resolve_org_id(organization_id)
    url = f"{DBIS_BASE_URL}/resourceIdsBySubject/{subject_id}/organization/{org_id}"
    data = await _fetch_json(url)
    return {
        "subject_id": subject_id,
        "organization_id": org_id,
        "resource_ids": data,
    }


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
