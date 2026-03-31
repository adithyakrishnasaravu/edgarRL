"""
Tests for db.py — DuckDB schema, CRUD helpers, and analytics queries.
"""

import pytest

from db import (
    get_connection,
    upsert_filing,
    list_filings,
    start_episode,
    record_step,
    close_episode,
    upsert_extracted_value,
    insert_golden,
    get_golden,
)


# ---------------------------------------------------------------------------
# Schema / connection
# ---------------------------------------------------------------------------

class TestGetConnection:
    def test_creates_tables(self, db_connection):
        tables = db_connection.execute("SHOW TABLES").fetchall()
        table_names = {t[0] for t in tables}
        expected = {"filings", "episodes", "extraction_steps",
                    "extracted_values", "golden_set", "action_stats"}
        assert expected.issubset(table_names)

    def test_creates_sequence(self, db_connection):
        # step_id_seq should exist
        result = db_connection.execute("SELECT nextval('step_id_seq')").fetchone()
        assert result[0] >= 1

    def test_idempotent(self, tmp_path):
        """Calling get_connection twice on same DB should not error."""
        db_path = tmp_path / "test.duckdb"
        conn1 = get_connection(db_path)
        conn1.close()
        conn2 = get_connection(db_path)
        tables = conn2.execute("SHOW TABLES").fetchall()
        assert len(tables) >= 5
        conn2.close()


# ---------------------------------------------------------------------------
# Filing helpers
# ---------------------------------------------------------------------------

class TestUpsertFiling:
    def test_insert_new(self, db_connection):
        meta = {
            "filing_id": "0000320193/0000320193-24-000123",
            "cik": "0000320193",
            "accession": "0000320193-24-000123",
            "ticker": "AAPL",
            "company_name": "Apple Inc",
            "sic_code": 3571,
            "fiscal_year": 2024,
        }
        upsert_filing(db_connection, meta)
        rows = db_connection.execute(
            "SELECT * FROM filings WHERE filing_id = ?",
            [meta["filing_id"]]
        ).fetchall()
        assert len(rows) == 1

    def test_upsert_updates_existing(self, db_connection):
        meta = {
            "filing_id": "test/filing",
            "cik": "0000000001",
            "ticker": "TEST",
        }
        upsert_filing(db_connection, meta)

        # Update ticker
        meta["ticker"] = "UPDATED"
        upsert_filing(db_connection, meta)

        rows = db_connection.execute(
            "SELECT ticker FROM filings WHERE filing_id = ?",
            [meta["filing_id"]]
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "UPDATED"

    def test_minimal_fields(self, db_connection):
        """Only filing_id and cik are truly required by NOT NULL."""
        meta = {"filing_id": "minimal/test", "cik": "0000000002"}
        upsert_filing(db_connection, meta)
        rows = db_connection.execute(
            "SELECT * FROM filings WHERE filing_id = ?",
            [meta["filing_id"]]
        ).fetchall()
        assert len(rows) == 1


class TestListFilings:
    def test_empty(self, db_connection):
        result = list_filings(db_connection)
        assert result == []

    def test_returns_dicts(self, db_connection):
        upsert_filing(db_connection, {
            "filing_id": "test/1", "cik": "0000000001", "ticker": "A",
        })
        upsert_filing(db_connection, {
            "filing_id": "test/2", "cik": "0000000002", "ticker": "B",
        })
        result = list_filings(db_connection)
        assert len(result) == 2
        assert isinstance(result[0], dict)
        assert "filing_id" in result[0]
        assert "ticker" in result[0]


# ---------------------------------------------------------------------------
# Episode helpers
# ---------------------------------------------------------------------------

class TestEpisodeHelpers:
    def _seed_filing(self, conn):
        upsert_filing(conn, {"filing_id": "f1", "cik": "0000000001"})

    def test_start_episode(self, db_connection):
        self._seed_filing(db_connection)
        start_episode(db_connection, "ep1", "f1", "revenue", 100.0)
        rows = db_connection.execute(
            "SELECT * FROM episodes WHERE episode_id = 'ep1'"
        ).fetchall()
        assert len(rows) == 1

    def test_start_episode_idempotent(self, db_connection):
        self._seed_filing(db_connection)
        start_episode(db_connection, "ep1", "f1", "revenue", 100.0)
        start_episode(db_connection, "ep1", "f1", "revenue", 100.0)  # ON CONFLICT DO NOTHING
        rows = db_connection.execute(
            "SELECT * FROM episodes WHERE episode_id = 'ep1'"
        ).fetchall()
        assert len(rows) == 1

    def test_record_step(self, db_connection):
        self._seed_filing(db_connection)
        start_episode(db_connection, "ep1", "f1", "revenue", 100.0)
        record_step(
            db_connection,
            episode_id="ep1",
            step_index=0,
            action=0,
            action_name="xbrl_direct",
            extracted_value=100.0,
            confidence=0.95,
            reward=1.0,
            latency_ms=5.0,
            xbrl_tag="us-gaap:Revenues",
        )
        rows = db_connection.execute(
            "SELECT * FROM extraction_steps WHERE episode_id = 'ep1'"
        ).fetchall()
        assert len(rows) == 1

    def test_record_multiple_steps(self, db_connection):
        self._seed_filing(db_connection)
        start_episode(db_connection, "ep1", "f1", "revenue", 100.0)
        for i in range(3):
            record_step(
                db_connection,
                episode_id="ep1",
                step_index=i,
                action=i,
                action_name=f"action_{i}",
                extracted_value=float(i * 10),
                confidence=0.5,
                reward=0.3,
                latency_ms=1.0,
            )
        rows = db_connection.execute(
            "SELECT * FROM extraction_steps WHERE episode_id = 'ep1'"
        ).fetchall()
        assert len(rows) == 3

    def test_close_episode(self, db_connection):
        self._seed_filing(db_connection)
        start_episode(db_connection, "ep1", "f1", "revenue", 100.0)
        close_episode(
            db_connection,
            episode_id="ep1",
            n_steps=2,
            total_reward=1.5,
            best_action=0,
            best_value=100.0,
            best_confidence=0.95,
        )
        row = db_connection.execute(
            "SELECT terminated, n_steps, total_reward, best_action FROM episodes WHERE episode_id = 'ep1'"
        ).fetchone()
        assert row[0] is True   # terminated
        assert row[1] == 2      # n_steps
        assert row[2] == 1.5    # total_reward
        assert row[3] == 0      # best_action


# ---------------------------------------------------------------------------
# Extracted values
# ---------------------------------------------------------------------------

class TestExtractedValues:
    def test_upsert_new(self, db_connection):
        upsert_extracted_value(
            db_connection,
            filing_id="f1",
            field_name="revenue",
            value=100.0,
            unit="USD",
            confidence=0.95,
            source_action=0,
            source_name="xbrl_direct",
        )
        rows = db_connection.execute(
            "SELECT * FROM extracted_values WHERE filing_id = 'f1' AND field_name = 'revenue'"
        ).fetchall()
        assert len(rows) == 1

    def test_upsert_updates(self, db_connection):
        upsert_extracted_value(
            db_connection, "f1", "revenue", 100.0, "USD", 0.5, 0, "xbrl",
        )
        upsert_extracted_value(
            db_connection, "f1", "revenue", 200.0, "USD", 0.9, 1, "html",
        )
        row = db_connection.execute(
            "SELECT value, source_name FROM extracted_values WHERE filing_id = 'f1' AND field_name = 'revenue'"
        ).fetchone()
        assert row[0] == 200.0
        assert row[1] == "html"


# ---------------------------------------------------------------------------
# Golden set
# ---------------------------------------------------------------------------

class TestGoldenSet:
    def test_insert_and_get(self, db_connection):
        insert_golden(db_connection, "f1", "revenue", 100.0, unit="USD")
        result = get_golden(db_connection, "f1")
        assert result == {"revenue": 100.0}

    def test_upsert_golden(self, db_connection):
        insert_golden(db_connection, "f1", "revenue", 100.0)
        insert_golden(db_connection, "f1", "revenue", 200.0)  # update
        result = get_golden(db_connection, "f1")
        assert result == {"revenue": 200.0}

    def test_multiple_fields(self, db_connection):
        insert_golden(db_connection, "f1", "revenue", 100.0)
        insert_golden(db_connection, "f1", "net_income", 50.0)
        result = get_golden(db_connection, "f1")
        assert result == {"revenue": 100.0, "net_income": 50.0}

    def test_get_empty(self, db_connection):
        result = get_golden(db_connection, "nonexistent")
        assert result == {}


# ---------------------------------------------------------------------------
# Full workflow integration
# ---------------------------------------------------------------------------

class TestFullWorkflow:
    def test_end_to_end(self, db_connection):
        """Simulate a complete episode lifecycle."""
        # 1. Register filing
        upsert_filing(db_connection, {
            "filing_id": "f1", "cik": "0000320193",
            "ticker": "AAPL", "fiscal_year": 2024,
        })

        # 2. Start episode
        start_episode(db_connection, "ep1", "f1", "revenue", 394_328_000_000.0)

        # 3. Record steps
        record_step(
            db_connection, "ep1", 0, 0, "xbrl_direct",
            394_328_000_000.0, 0.95, 1.0, 5.0,
            xbrl_tag="us-gaap:Revenues",
        )

        # 4. Close episode
        close_episode(db_connection, "ep1", 1, 1.0, 0, 394_328_000_000.0, 0.95)

        # 5. Store extracted value
        upsert_extracted_value(
            db_connection, "f1", "revenue",
            394_328_000_000.0, "USD", 0.95, 0, "xbrl_direct",
        )

        # 6. Add golden set entry
        insert_golden(db_connection, "f1", "revenue", 394_328_000_000.0)

        # Verify everything
        filings = list_filings(db_connection)
        assert len(filings) == 1

        golden = get_golden(db_connection, "f1")
        assert golden["revenue"] == 394_328_000_000.0

        episodes = db_connection.execute("SELECT * FROM episodes").fetchall()
        assert len(episodes) == 1

        steps = db_connection.execute("SELECT * FROM extraction_steps").fetchall()
        assert len(steps) == 1
