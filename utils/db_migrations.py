"""Simple versioned database migrations."""
from __future__ import annotations

from typing import Callable, Dict

import psycopg2
from psycopg2.extensions import connection as PGConnection

MigrationFunc = Callable[[PGConnection], None]


def _migration_1(conn: PGConnection) -> None:
    """Initial documents table (idempotent)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                url TEXT UNIQUE NOT NULL,
                title TEXT,
                safe_title TEXT,
                crawl_date DATE,
                lang TEXT,
                summary TEXT,
                tags TEXT[],
                markdown_content TEXT,
                markdown_hash TEXT,
                recordset TEXT,
                vector_file_id VARCHAR(255),
                old_file_id VARCHAR(255),
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                page_type VARCHAR(50),
                no_upload BOOLEAN DEFAULT FALSE,
                is_stale BOOLEAN DEFAULT FALSE
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_documents_recordset
            ON documents(recordset)
            """
        )


def _migration_2(conn: PGConnection) -> None:
    """Ensure is_stale column exists (legacy compatibility)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            ALTER TABLE documents
            ADD COLUMN IF NOT EXISTS is_stale BOOLEAN DEFAULT FALSE
            """
        )


def _migration_3(conn: PGConnection) -> None:
    """Ensure job_locks table exists (used by CLI)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS job_locks (
                name TEXT PRIMARY KEY,
                acquired_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )


def _migration_4(conn: PGConnection) -> None:
    """Ensure prompt_versions history table exists."""
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS prompt_versions (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                prompt TEXT NOT NULL,
                edited_by TEXT,
                note TEXT
            )
            """
        )


def _migration_5(conn: PGConnection) -> None:
    """Ensure log_table exists with expected columns and indexes."""
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS log_table (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                session_id VARCHAR(36),
                user_input TEXT NOT NULL,
                assistant_response TEXT NOT NULL,
                error_code VARCHAR(10),
                citation_count INTEGER DEFAULT 0,
                citations JSONB,
                confidence DECIMAL(3,2) DEFAULT 0.0,
                request_classification VARCHAR(50),
                evaluation_notes TEXT,
                model VARCHAR(100),
                usage_input_tokens INTEGER,
                usage_output_tokens INTEGER,
                usage_total_tokens INTEGER,
                usage_reasoning_tokens INTEGER,
                api_cost_usd NUMERIC(12,6),
                response_time_ms INTEGER
            )
            """
        )
        # Backfill any missing columns for legacy deployments.
        for column_def in (
            "session_id VARCHAR(36)",
            "error_code VARCHAR(10)",
            "citation_count INTEGER DEFAULT 0",
            "citations JSONB",
            "confidence DECIMAL(3,2) DEFAULT 0.0",
            "request_classification VARCHAR(50)",
            "evaluation_notes TEXT",
            "model VARCHAR(100)",
            "usage_input_tokens INTEGER",
            "usage_output_tokens INTEGER",
            "usage_total_tokens INTEGER",
            "usage_reasoning_tokens INTEGER",
            "api_cost_usd NUMERIC(12,6)",
            "response_time_ms INTEGER",
        ):
            cur.execute(f"ALTER TABLE log_table ADD COLUMN IF NOT EXISTS {column_def}")
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_log_table_session_id
            ON log_table(session_id)
            """
        )


def _migration_6(conn: PGConnection) -> None:
    """Ensure url_configs table exists for scraper settings."""
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS url_configs (
                id SERIAL PRIMARY KEY,
                url TEXT NOT NULL,
                recordset TEXT NOT NULL DEFAULT '',
                depth INTEGER NOT NULL DEFAULT 2,
                exclude_paths TEXT[] DEFAULT ARRAY['/en/', '/pl/', '/_ablage-alte-www/', '/site-euv/', '/site-zwe-ikm/'],
                include_lang_prefixes TEXT[] DEFAULT ARRAY['/de/'],
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )


def _migration_7(conn: PGConnection) -> None:
    """Ensure llm_settings table exists with a default row."""
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_settings (
                id SERIAL PRIMARY KEY,
                model VARCHAR(100) NOT NULL DEFAULT 'gpt-4o-mini',
                parallel_tool_calls BOOLEAN DEFAULT TRUE,
                reasoning_effort VARCHAR(20) DEFAULT 'medium',
                text_verbosity VARCHAR(20) DEFAULT 'medium',
                updated_by VARCHAR(100),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )
        cur.execute(
            """
            INSERT INTO llm_settings (model, parallel_tool_calls, reasoning_effort, text_verbosity, updated_by)
            SELECT 'gpt-4o-mini', TRUE, 'medium', 'medium', 'system'
            WHERE NOT EXISTS (SELECT 1 FROM llm_settings)
            """
        )


def _migration_8(conn: PGConnection) -> None:
    """Ensure filter_settings table exists with expected columns and seed row."""
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS filter_settings (
                id SERIAL PRIMARY KEY,
                web_search_enabled BOOLEAN DEFAULT TRUE,
                web_locale TEXT DEFAULT 'de-DE',
                web_domains TEXT[] DEFAULT ARRAY[]::TEXT[],
                web_domains_mode VARCHAR(16) DEFAULT 'include',
                web_userloc_type TEXT DEFAULT 'approximate',
                web_userloc_country TEXT,
                web_userloc_city TEXT,
                web_userloc_region TEXT,
                web_userloc_timezone TEXT,
                updated_by VARCHAR(100),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )
        for column_def in (
            "web_search_enabled BOOLEAN DEFAULT TRUE",
            "web_locale TEXT DEFAULT 'de-DE'",
            "web_domains TEXT[] DEFAULT ARRAY[]::TEXT[]",
            "web_domains_mode VARCHAR(16) DEFAULT 'include'",
            "web_userloc_type TEXT DEFAULT 'approximate'",
            "web_userloc_country TEXT",
            "web_userloc_city TEXT",
            "web_userloc_region TEXT",
            "web_userloc_timezone TEXT",
            "updated_by VARCHAR(100)",
            "updated_at TIMESTAMPTZ DEFAULT NOW()",
        ):
            cur.execute(f"ALTER TABLE filter_settings ADD COLUMN IF NOT EXISTS {column_def}")
        cur.execute(
            """
            INSERT INTO filter_settings (
                web_search_enabled,
                web_locale,
                web_domains,
                web_domains_mode,
                web_userloc_type,
                web_userloc_country,
                web_userloc_city,
                web_userloc_region,
                web_userloc_timezone,
                updated_by
            )
            SELECT TRUE, 'de-DE', ARRAY[]::TEXT[], 'include', 'approximate', NULL, NULL, NULL, NULL, 'system'
            WHERE NOT EXISTS (SELECT 1 FROM filter_settings)
            """
        )


def _migration_9(conn: PGConnection) -> None:
    """Ensure request_classifications table exists with default categories."""
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS request_classifications (
                id SERIAL PRIMARY KEY,
                categories TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
                updated_by TEXT,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        cur.execute(
            """
            INSERT INTO request_classifications (categories, updated_by)
            SELECT ARRAY[
                'library hours',
                'book search',
                'research help',
                'account info',
                'facility info',
                'policy question',
                'technical support',
                'other'
            ]::TEXT[], 'system'
            WHERE NOT EXISTS (SELECT 1 FROM request_classifications)
            """
        )


def _migration_10(conn: PGConnection) -> None:
    """Add DBIS MCP tool settings to filter_settings."""
    with conn.cursor() as cur:
        cur.execute(
            """
            ALTER TABLE filter_settings
            ADD COLUMN IF NOT EXISTS dbis_mcp_enabled BOOLEAN DEFAULT TRUE,
            ADD COLUMN IF NOT EXISTS dbis_org_id TEXT
            """
        )
        # Seed defaults if row exists but values are NULL
        cur.execute(
            """
            UPDATE filter_settings
            SET dbis_mcp_enabled = COALESCE(dbis_mcp_enabled, TRUE)
            """
        )


MIGRATIONS: Dict[int, MigrationFunc] = {
    1: _migration_1,
    2: _migration_2,
    3: _migration_3,
    4: _migration_4,
    5: _migration_5,
    6: _migration_6,
    7: _migration_7,
    8: _migration_8,
    9: _migration_9,
    10: _migration_10,
}


def run_migrations(connection_factory: Callable[[], PGConnection]) -> int:
    """Run pending migrations and return the current schema version."""
    conn = connection_factory()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS schema_version (
                        version INTEGER PRIMARY KEY,
                        applied_at TIMESTAMPTZ DEFAULT NOW()
                    )
                    """
                )
                cur.execute("SELECT COALESCE(MAX(version), 0) FROM schema_version")
                row = cur.fetchone()
                current_version = int(row[0]) if row and row[0] is not None else 0

        for version in sorted(MIGRATIONS):
            migration = MIGRATIONS[version]
            try:
                with conn:
                    migration(conn)
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO schema_version (version) VALUES (%s)
                            ON CONFLICT (version) DO NOTHING
                            """,
                            (version,)
                        )
                current_version = max(current_version, version)
            except psycopg2.Error:
                conn.rollback()
                raise

        return current_version
    finally:
        conn.close()
