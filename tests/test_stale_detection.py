import datetime
from collections import defaultdict
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pages.scrape as scrape


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self._result = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        self.conn.executed.append((sql, params))
        params = params or ()
        if "FROM documents WHERE recordset" in sql:
            recordset = params[0]
            self._result = list(self.conn.recordset_rows.get(recordset, []))
        elif "UPDATE documents SET is_stale = FALSE" in sql:
            ids = list(params[0]) if params else []
            if "updated_at" in sql:
                # this is the TRUE branch; handled separately
                pass
            if ids:
                self.conn.updated_false_ids.extend(ids)
            self._result = []
        elif "SET is_stale = TRUE" in sql:
            ids = list(params[0]) if params else []
            if ids:
                self.conn.updated_true_ids.extend(ids)
            self._result = []
        else:
            self._result = []

    def fetchall(self):
        return list(self._result)

    def fetchone(self):
        return self._result[0] if self._result else None


class FakeConnection:
    def __init__(self, recordset_rows):
        self.recordset_rows = recordset_rows
        self.executed = []
        self.updated_true_ids = []
        self.updated_false_ids = []

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        pass


@pytest.fixture(autouse=True)
def reset_globals():
    prev = scrape.recordset_latest_urls
    scrape.recordset_latest_urls = defaultdict(set)
    try:
        yield
    finally:
        scrape.recordset_latest_urls = prev


def test_update_stale_documents_handles_deleted_and_retained(monkeypatch):
    scrape.recordset_latest_urls["rs"] = {
        "https://example.com/kept",
    }

    doc_rows = {
        "rs": [
            (1, "rs", "https://example.com/kept", "Kept", datetime.date(2024, 1, 1)),
            (2, "rs", "https://example.com/missing", "Missing", datetime.date(2024, 1, 2)),
            (3, "rs", "https://example.com/unlinked", "Unlinked", datetime.date(2024, 1, 3)),
        ]
    }
    conn = FakeConnection(doc_rows)

    def fake_verify(url, log_callback=None):
        if url.endswith("missing"):
            return True, "HTTP 404"
        if url.endswith("unlinked"):
            return False, "reachable"
        return False, "reachable"

    monkeypatch.setattr(scrape, "verify_url_deleted", fake_verify)

    stale_candidates = scrape.update_stale_documents(conn, dry_run=False)

    assert len(stale_candidates) == 1
    assert stale_candidates[0]["url"] == "https://example.com/missing"
    assert conn.updated_true_ids == [2]
    # Seen page and retained page are marked fresh
    assert 1 in conn.updated_false_ids
    assert 3 in conn.updated_false_ids
