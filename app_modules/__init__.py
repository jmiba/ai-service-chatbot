"""
app_modules - Refactored modules extracted from app.py to reduce code duplication.
"""

from .helpers import (
    get_attr,
    redact_ids,
    safe_output_text,
    humanize_debug_text,
    extract_output_json,
    extract_first_json_object,
    iter_content_items,
    iter_content_parts,
    get_usage_from_final,
    coerce_text_part,
)

from .prompts import (
    load_prompt_config,
    build_eval_response_format,
    format_prompt,
)

from .citations import (
    extract_citations_from_annotations_response_dict,
    render_with_citations_by_index,
    render_sources_list,
    replace_filecite_markers_with_sup,
    strip_markdown_links_preserve_md,
    trim_separator_artifacts,
)

__all__ = [
    # helpers
    "get_attr",
    "redact_ids",
    "safe_output_text",
    "humanize_debug_text",
    "extract_output_json",
    "extract_first_json_object",
    "iter_content_items",
    "iter_content_parts",
    "get_usage_from_final",
    "coerce_text_part",
    # prompts
    "load_prompt_config",
    "build_eval_response_format",
    "format_prompt",
    # citations
    "extract_citations_from_annotations_response_dict",
    "render_with_citations_by_index",
    "render_sources_list",
    "replace_filecite_markers_with_sup",
    "strip_markdown_links_preserve_md",
    "trim_separator_artifacts",
]
