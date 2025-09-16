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
    compute_sha256,
    get_kb_entries,
    save_url_configs,
    load_url_configs,
    initialize_default_url_configs,
    create_llm_settings_table,
    save_llm_settings,
    get_llm_settings,
    get_available_openai_models,
    supports_reasoning_effort,
    get_supported_verbosity_options,
    supports_full_verbosity,
    create_filter_settings_table,
    save_filter_settings,
    get_filter_settings,
    estimate_cost_usd,
    # Export request classification helpers
    create_request_classifications_table,
    get_request_classifications,
    save_request_classifications,
    get_document_status_counts,
)

# Now you can import from utils directly, e.g.:
# from utils import get_connection, load_css
