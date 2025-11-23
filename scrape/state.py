from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict


@dataclass
class CrawlerState:
    """Mutable crawl state for a single run (shared across configs)."""

    visited_raw: set[str] = field(default_factory=set)
    visited_norm: set[str] = field(default_factory=set)
    frontier_seen: list[str] = field(default_factory=list)
    recordset_latest_urls: DefaultDict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    base_path: str | None = None
    processed_pages_count: int = 0
    dry_run_llm_eligible_count: int = 0
    llm_analysis_results: list[dict] = field(default_factory=list)

    def reset(self):
        self.visited_raw.clear()
        self.visited_norm.clear()
        self.frontier_seen.clear()
        self.recordset_latest_urls.clear()
        self.base_path = None
        self.processed_pages_count = 0
        self.dry_run_llm_eligible_count = 0
        self.llm_analysis_results.clear()
