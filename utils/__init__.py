# utils/__init__.py

# Import key functions or classes from submodules for convenient access

from .utils import (
    get_connection,
    create_database_if_not_exists,
    create_log_table,
    create_prompt_versions_table,
    initialize_default_prompt_if_empty,
    get_latest_prompt,
    create_knowledge_base_table,
    admin_authentication,
    render_sidebar,
    compute_sha256,
    get_kb_entries
)

# Now you can import from utils directly, e.g.:
# from utils import get_connection, load_css
