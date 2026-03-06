"""
db.py — DuckDB schema and helpers for the EDGAR RL extraction system.

Tables:
  filings          — metadata for each downloaded 10-K
  episodes         — one row per (filing, field) RL episode
  extraction_steps — one row per action taken within an episode
  extracted_values — final best extraction per (filing, field)
  golden_set       — hand-verified ground truth for reward calibration
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import duckdb

DB_PATH = Path(__file__).parent / "edgar_rl.duckdb"


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

DDL = """
-- Raw filing metadata
CREATE TABLE IF NOT EXISTS filings (
    filing_id       VARCHAR PRIMARY KEY,   -- "{cik}/{accession}"
    cik             VARCHAR NOT NULL,
    accession       VARCHAR,
    ticker          VARCHAR,
    company_name    VARCHAR,
    sic_code        INTEGER,
    fiscal_year     INTEGER,
    fiscal_year_end DATE,
    form_type       VARCHAR DEFAULT '10-K',
    filed_date      DATE,
    xbrl_path       VARCHAR,
    html_path       VARCHAR,
    txt_path        VARCHAR,
    created_at      TIMESTAMP DEFAULT now()
);

-- One RL episode = one (filing, field) pair
CREATE TABLE IF NOT EXISTS episodes (
    episode_id      VARCHAR PRIMARY KEY,   -- uuid or "{filing_id}:{field}:{ts}"
    filing_id       VARCHAR NOT NULL REFERENCES filings(filing_id),
    field_name      VARCHAR NOT NULL,
    ground_truth    DOUBLE,                -- XBRL oracle value (may be NULL)
    n_steps         INTEGER DEFAULT 0,
    total_reward    DOUBLE DEFAULT 0.0,
    terminated      BOOLEAN DEFAULT false,
    best_action     INTEGER,
    best_value      DOUBLE,
    best_confidence DOUBLE,
    started_at      TIMESTAMP DEFAULT now(),
    ended_at        TIMESTAMP
);

-- Individual extractor attempts within an episode
CREATE TABLE IF NOT EXISTS extraction_steps (
    step_id         BIGINT PRIMARY KEY,    -- auto-increment via sequence
    episode_id      VARCHAR NOT NULL REFERENCES episodes(episode_id),
    step_index      INTEGER NOT NULL,
    action          INTEGER NOT NULL,      -- 0-5
    action_name     VARCHAR,
    extracted_value DOUBLE,
    confidence      DOUBLE,
    reward          DOUBLE,
    latency_ms      DOUBLE,
    xbrl_tag        VARCHAR,
    error           VARCHAR,
    created_at      TIMESTAMP DEFAULT now()
);

-- Final best extraction per (filing, field) — populated after episode ends
CREATE TABLE IF NOT EXISTS extracted_values (
    filing_id       VARCHAR NOT NULL,
    field_name      VARCHAR NOT NULL,
    value           DOUBLE,
    unit            VARCHAR,
    confidence      DOUBLE,
    source_action   INTEGER,
    source_name     VARCHAR,
    xbrl_tag        VARCHAR,
    period          VARCHAR,
    created_at      TIMESTAMP DEFAULT now(),
    PRIMARY KEY (filing_id, field_name)
);

-- Hand-verified golden set for calibration
CREATE TABLE IF NOT EXISTS golden_set (
    filing_id       VARCHAR NOT NULL,
    field_name      VARCHAR NOT NULL,
    verified_value  DOUBLE NOT NULL,
    unit            VARCHAR,
    period          VARCHAR,
    notes           VARCHAR,
    verified_by     VARCHAR DEFAULT 'manual',
    verified_at     TIMESTAMP DEFAULT now(),
    PRIMARY KEY (filing_id, field_name)
);

-- Per-action running performance statistics
CREATE TABLE IF NOT EXISTS action_stats (
    field_name      VARCHAR NOT NULL,
    action          INTEGER NOT NULL,
    n_attempts      INTEGER DEFAULT 0,
    n_success       INTEGER DEFAULT 0,
    avg_reward      DOUBLE DEFAULT 0.0,
    avg_latency_ms  DOUBLE DEFAULT 0.0,
    updated_at      TIMESTAMP DEFAULT now(),
    PRIMARY KEY (field_name, action)
);
"""

SEQUENCES = """
CREATE SEQUENCE IF NOT EXISTS step_id_seq START 1;
"""


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

def get_connection(db_path: Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    """Return a connection to the DuckDB file, creating schema if needed."""
    conn = duckdb.connect(str(db_path))
    conn.execute(SEQUENCES)
    conn.execute(DDL)
    return conn


# ---------------------------------------------------------------------------
# Filing helpers
# ---------------------------------------------------------------------------

def upsert_filing(conn: duckdb.DuckDBPyConnection, meta: dict) -> None:
    """Insert or replace a filing metadata record."""
    conn.execute("""
        INSERT INTO filings (
            filing_id, cik, accession, ticker, company_name,
            sic_code, fiscal_year, fiscal_year_end, form_type,
            filed_date, xbrl_path, html_path, txt_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (filing_id) DO UPDATE SET
            ticker          = EXCLUDED.ticker,
            company_name    = EXCLUDED.company_name,
            sic_code        = EXCLUDED.sic_code,
            fiscal_year     = EXCLUDED.fiscal_year,
            fiscal_year_end = EXCLUDED.fiscal_year_end,
            xbrl_path       = EXCLUDED.xbrl_path,
            html_path       = EXCLUDED.html_path,
            txt_path        = EXCLUDED.txt_path
    """, [
        meta.get("filing_id"),
        meta.get("cik"),
        meta.get("accession"),
        meta.get("ticker"),
        meta.get("company_name"),
        meta.get("sic_code"),
        meta.get("fiscal_year"),
        meta.get("fiscal_year_end"),
        meta.get("form_type", "10-K"),
        meta.get("filed_date"),
        meta.get("xbrl_path"),
        meta.get("html_path"),
        meta.get("txt_path"),
    ])


def list_filings(conn: duckdb.DuckDBPyConnection) -> list[dict]:
    rows = conn.execute("SELECT * FROM filings ORDER BY filed_date DESC").fetchall()
    cols = [d[0] for d in conn.description]
    return [dict(zip(cols, row)) for row in rows]


# ---------------------------------------------------------------------------
# Episode helpers
# ---------------------------------------------------------------------------

def start_episode(
    conn: duckdb.DuckDBPyConnection,
    episode_id: str,
    filing_id: str,
    field_name: str,
    ground_truth: Optional[float],
) -> None:
    conn.execute("""
        INSERT INTO episodes (episode_id, filing_id, field_name, ground_truth)
        VALUES (?, ?, ?, ?)
        ON CONFLICT DO NOTHING
    """, [episode_id, filing_id, field_name, ground_truth])


def record_step(
    conn: duckdb.DuckDBPyConnection,
    episode_id: str,
    step_index: int,
    action: int,
    action_name: str,
    extracted_value: Optional[float],
    confidence: float,
    reward: float,
    latency_ms: float,
    xbrl_tag: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    conn.execute("""
        INSERT INTO extraction_steps (
            step_id, episode_id, step_index, action, action_name,
            extracted_value, confidence, reward, latency_ms, xbrl_tag, error
        ) VALUES (nextval('step_id_seq'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        episode_id, step_index, action, action_name,
        extracted_value, confidence, reward, latency_ms, xbrl_tag, error,
    ])


def close_episode(
    conn: duckdb.DuckDBPyConnection,
    episode_id: str,
    n_steps: int,
    total_reward: float,
    best_action: Optional[int],
    best_value: Optional[float],
    best_confidence: float,
) -> None:
    conn.execute("""
        UPDATE episodes SET
            n_steps         = ?,
            total_reward    = ?,
            terminated      = true,
            best_action     = ?,
            best_value      = ?,
            best_confidence = ?,
            ended_at        = now()
        WHERE episode_id = ?
    """, [n_steps, total_reward, best_action, best_value, best_confidence, episode_id])


# ---------------------------------------------------------------------------
# Extracted values
# ---------------------------------------------------------------------------

def upsert_extracted_value(
    conn: duckdb.DuckDBPyConnection,
    filing_id: str,
    field_name: str,
    value: Optional[float],
    unit: Optional[str],
    confidence: float,
    source_action: int,
    source_name: str,
    xbrl_tag: Optional[str] = None,
    period: Optional[str] = None,
) -> None:
    conn.execute("""
        INSERT INTO extracted_values (
            filing_id, field_name, value, unit, confidence,
            source_action, source_name, xbrl_tag, period
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (filing_id, field_name) DO UPDATE SET
            value         = EXCLUDED.value,
            unit          = EXCLUDED.unit,
            confidence    = EXCLUDED.confidence,
            source_action = EXCLUDED.source_action,
            source_name   = EXCLUDED.source_name,
            xbrl_tag      = EXCLUDED.xbrl_tag,
            period        = EXCLUDED.period,
            created_at    = now()
    """, [
        filing_id, field_name, value, unit, confidence,
        source_action, source_name, xbrl_tag, period,
    ])


# ---------------------------------------------------------------------------
# Golden set
# ---------------------------------------------------------------------------

def insert_golden(
    conn: duckdb.DuckDBPyConnection,
    filing_id: str,
    field_name: str,
    verified_value: float,
    unit: Optional[str] = None,
    period: Optional[str] = None,
    notes: Optional[str] = None,
) -> None:
    conn.execute("""
        INSERT INTO golden_set (filing_id, field_name, verified_value, unit, period, notes)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (filing_id, field_name) DO UPDATE SET
            verified_value = EXCLUDED.verified_value,
            unit           = EXCLUDED.unit,
            period         = EXCLUDED.period,
            notes          = EXCLUDED.notes,
            verified_at    = now()
    """, [filing_id, field_name, verified_value, unit, period, notes])


def get_golden(conn: duckdb.DuckDBPyConnection, filing_id: str) -> dict[str, float]:
    """Return {field_name: verified_value} for a filing."""
    rows = conn.execute(
        "SELECT field_name, verified_value FROM golden_set WHERE filing_id = ?",
        [filing_id]
    ).fetchall()
    return {r[0]: r[1] for r in rows}


# ---------------------------------------------------------------------------
# Analytics queries
# ---------------------------------------------------------------------------

def field_accuracy_report(conn: duckdb.DuckDBPyConnection) -> None:
    """Print accuracy by field vs golden set."""
    print("\n=== Field Accuracy vs Golden Set ===\n")
    rows = conn.execute("""
        SELECT
            g.field_name,
            COUNT(*) as n,
            AVG(ABS(ev.value - g.verified_value) / NULLIF(ABS(g.verified_value), 0)) as mean_rel_err,
            SUM(CASE WHEN ABS(ev.value - g.verified_value) / NULLIF(ABS(g.verified_value), 0) <= 0.01 THEN 1 ELSE 0 END) as n_exact
        FROM golden_set g
        JOIN extracted_values ev ON g.filing_id = ev.filing_id AND g.field_name = ev.field_name
        GROUP BY g.field_name
        ORDER BY mean_rel_err ASC
    """).fetchall()
    print(f"{'Field':<25} {'N':>5} {'MeanRelErr':>12} {'ExactCount':>12}")
    print("-" * 58)
    for row in rows:
        print(f"{row[0]:<25} {row[1]:>5} {(row[2] or 0)*100:>11.2f}% {row[3]:>12}")


def action_performance_report(conn: duckdb.DuckDBPyConnection) -> None:
    """Print per-action average reward and usage count."""
    print("\n=== Action Performance ===\n")
    rows = conn.execute("""
        SELECT
            action,
            action_name,
            COUNT(*) as n_uses,
            AVG(reward) as avg_reward,
            AVG(latency_ms) as avg_latency_ms
        FROM extraction_steps
        GROUP BY action, action_name
        ORDER BY action
    """).fetchall()
    print(f"{'Action':>6} {'Name':<20} {'Uses':>6} {'AvgReward':>10} {'AvgLatencyMs':>14}")
    print("-" * 60)
    for row in rows:
        print(f"{row[0]:>6} {(row[1] or '?'):<20} {row[2]:>6} {(row[3] or 0):>+10.3f} {(row[4] or 0):>14.1f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    conn = get_connection()
    print(f"DuckDB initialized at: {DB_PATH}")
    tables = conn.execute("SHOW TABLES").fetchall()
    print(f"Tables: {[t[0] for t in tables]}")
    conn.close()
    print("Done.")
