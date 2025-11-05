import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scrape import maintenance


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


def test_update_stale_documents_handles_deleted_and_retained():
    recordset_latest_urls = {
        "rs": {
            "https://example.com/kept",
        }
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

    stale_candidates = maintenance.update_stale_documents(
        conn,
        dry_run=False,
        recordset_latest_urls=recordset_latest_urls,
        verify_url_deleted=fake_verify,
    )

    assert len(stale_candidates) == 1
    assert stale_candidates[0]["url"] == "https://example.com/missing"
    assert conn.updated_true_ids == [2]
    # Seen page and retained page are marked fresh
    assert 1 in conn.updated_false_ids
    assert 3 in conn.updated_false_ids
