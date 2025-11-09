# utils/__init__.py

# Import key functions or classes from submodules for convenient access

from .utils import (
    get_connection,
    create_database_if_not_exists,
    create_log_table,
    create_prompt_versions_table,
    create_url_configs_table,
    create_knowledge_base_table,
    initialize_default_prompt_if_empty,
    get_latest_prompt,
    admin_authentication,
    render_sidebar,
    render_save_chat_button,
    build_chat_markdown,
    load_css,
    compute_sha256,
    get_kb_entries,
    save_url_configs,
    load_url_configs,
    initialize_default_url_configs,
    create_llm_settings_table,
    save_llm_settings,
    get_llm_settings,
    get_document_by_identifier,
    get_available_openai_models,
    supports_reasoning_effort,
    get_supported_verbosity_options,
    supports_full_verbosity,
    create_filter_settings_table,
    save_filter_settings,
    get_filter_settings,
    estimate_cost_usd,
    normalize_tags_for_storage,
    normalize_existing_document_tags,
    # Export request classification helpers
    create_request_classifications_table,
    get_request_classifications,
    save_request_classifications,
    get_document_status_counts,
    get_document_metrics,
    is_job_locked,
    release_job_lock,
    show_blocking_overlay,
    hide_blocking_overlay,
    render_log_output,
)

from .cli_runner import CLIJob, CLIJobError, launch_cli_job
from .vector_status import read_vector_status, write_vector_status
from .vector_details import read_vector_store_details, write_vector_store_details
from .vector_dirty import mark_vector_store_dirty, clear_vector_store_dirty, is_vector_store_dirty
from .error_codes import load_error_code_labels, human_error_label, format_error_code_legend
from .scraper_schedule import read_scraper_schedule, write_scraper_schedule, update_last_scrape_run

# Now you can import from utils directly, e.g.:
# from utils import get_connection, load_css
