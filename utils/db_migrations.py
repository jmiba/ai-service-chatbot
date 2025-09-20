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


MIGRATIONS: Dict[int, MigrationFunc] = {
    1: _migration_1,
    2: _migration_2,
    3: _migration_3,
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
            if version <= current_version:
                continue
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
                current_version = version
            except psycopg2.Error:
                conn.rollback()
                raise

        return current_version
    finally:
        conn.close()
