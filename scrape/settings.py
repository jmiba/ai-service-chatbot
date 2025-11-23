from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class ScrapeSettings:
    max_depth: int = 3
    max_urls_per_run: int = 5000
    keep_query_keys: set[str] | None = None
    exclude_paths: list[str] | None = None
    include_lang_prefixes: list[str] | None = None
    dry_run: bool = False
    recordset: str | None = None
    source_config_id: int | None = None
    headers: dict | None = None

    @classmethod
    def from_config(
        cls,
        *,
        max_depth: int,
        budget: int,
        keep_query: Iterable[str] | None = None,
        exclude_paths: list[str] | None = None,
        include_lang_prefixes: list[str] | None = None,
        dry_run: bool = False,
        recordset: str | None = None,
        source_config_id: int | None = None,
        headers: dict | None = None,
    ) -> "ScrapeSettings":
        return cls(
            max_depth=max_depth,
            max_urls_per_run=budget,
            keep_query_keys=set(keep_query) if keep_query else None,
            exclude_paths=exclude_paths or None,
            include_lang_prefixes=include_lang_prefixes or None,
            dry_run=dry_run,
            recordset=recordset,
            source_config_id=source_config_id,
            headers=headers,
        )


def build_headers(admin_email: str | None = None) -> dict:
    ua = "Mozilla/5.0 (compatible; Viadrina-Indexer/1.0"
    if admin_email:
        ua += f"; +mailto:{admin_email}"
    ua += ")"
    return {"User-Agent": ua}
