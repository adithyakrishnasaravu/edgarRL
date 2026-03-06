"""
html_extractor.py — Action 1: Extract financial values from HTML income statement tables.

Strategy:
  1. Find the income statement section by heading keywords.
  2. Parse all <table> elements within that section.
  3. Score each table by how many known income-statement labels it contains.
  4. In the best table, find the column for the most recent annual period.
  5. Walk rows looking for label matches → return the cell value.

Returns (value: float | None, confidence: float).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

FIELD_CONFIG_PATH = Path(__file__).parent / "field_config.json"
with open(FIELD_CONFIG_PATH) as f:
    _FIELD_CONFIG = json.load(f)

FIELDS: dict = _FIELD_CONFIG["fields"]

# ---------------------------------------------------------------------------
# Label matching — each field has a list of expected row labels in HTML tables
# ---------------------------------------------------------------------------

# Additional table-level scoring keywords (presence → higher table score)
INCOME_STATEMENT_SIGNALS = [
    "revenue", "net sales", "gross profit", "operating income",
    "net income", "earnings per share", "cost of revenue",
]


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for fuzzy matching."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _label_matches(cell_text: str, field_name: str) -> float:
    """
    Return a match score [0, 1] for how well a table cell label matches a field.
    Uses the html_labels list from field_config.json.
    """
    norm = _normalize(cell_text)
    labels = FIELDS.get(field_name, {}).get("html_labels", [])
    best = 0.0
    for label in labels:
        nl = _normalize(label)
        if norm == nl:
            best = max(best, 1.0)
        elif nl in norm:
            # Partial match — penalize by ratio
            best = max(best, len(nl) / max(len(norm), 1) * 0.85)
    return best


# ---------------------------------------------------------------------------
# Value parsing
# ---------------------------------------------------------------------------

_MULTIPLIER_RE = re.compile(
    r"(in\s+)?(thousands|millions|billions)", re.IGNORECASE
)
_VALUE_RE = re.compile(r"[\$]?\s*\(?([\d,]+(?:\.\d+)?)\)?")


def _parse_numeric(text: str) -> Optional[float]:
    """Extract a numeric value from a table cell string, handling parentheses for negatives."""
    text = text.strip()
    if not text or text in ("—", "–", "-", "N/A", "n/a", "*"):
        return None
    negative = text.startswith("(") and text.endswith(")")
    m = _VALUE_RE.search(text)
    if not m:
        return None
    try:
        val = float(m.group(1).replace(",", ""))
        return -val if negative else val
    except ValueError:
        return None


def _detect_scale(soup) -> float:
    """
    Detect the reporting scale from page-level text (thousands / millions / billions).
    Returns the multiplier to apply to raw table values.
    """
    text = soup.get_text(" ", strip=True)[:5000]
    m = _MULTIPLIER_RE.search(text)
    if m:
        word = m.group(2).lower()
        return {"thousands": 1_000, "millions": 1_000_000, "billions": 1_000_000_000}[word]
    return 1.0  # assume units already in USD if not specified


# ---------------------------------------------------------------------------
# Table scoring and column selection
# ---------------------------------------------------------------------------

def _score_table(rows: list[list[str]]) -> float:
    """Score a parsed table by income-statement signal density."""
    all_text = " ".join(cell for row in rows for cell in row).lower()
    hits = sum(1 for sig in INCOME_STATEMENT_SIGNALS if sig in all_text)
    return hits / len(INCOME_STATEMENT_SIGNALS)


def _pick_annual_column(header_row: list[str]) -> int:
    """
    Pick the column index most likely to be the most recent annual period.
    Prefers cells containing a 4-digit year; falls back to column 1.
    """
    best_col = 1  # default: first data column (col 0 is usually labels)
    best_year = 0
    for i, cell in enumerate(header_row):
        m = re.search(r"(20\d{2}|19\d{2})", cell)
        if m:
            year = int(m.group(1))
            if year > best_year:
                best_year = year
                best_col = i
    return best_col


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

def extract_from_html(
    html_path: Path,
    field_name: str,
) -> tuple[Optional[float], float]:
    """
    Parse an HTML 10-K filing and extract a single financial field.

    Returns:
        (value, confidence)  where confidence ∈ [0, 1].
        Returns (None, 0.0) on failure.
    """
    try:
        from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
        import warnings
        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    except ImportError:
        raise ImportError("beautifulsoup4 is required: pip install beautifulsoup4 lxml")

    if not html_path.exists():
        return None, 0.0

    try:
        raw = html_path.read_bytes()
        soup = BeautifulSoup(raw, "lxml")
    except Exception as e:
        logger.warning("HTML parse error for %s: %s", html_path, e)
        return None, 0.0

    scale = _detect_scale(soup)

    # Remove script/style noise
    for tag in soup(["script", "style", "footer", "header"]):
        tag.decompose()

    tables = soup.find_all("table")
    if not tables:
        return None, 0.0

    # --- Score every table, keep top candidates ---
    scored: list[tuple[float, list[list[str]], object]] = []
    for table in tables:
        rows_raw = table.find_all("tr")
        rows: list[list[str]] = []
        for tr in rows_raw:
            cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
            if cells:
                rows.append(cells)
        if len(rows) < 3:
            continue
        score = _score_table(rows)
        if score > 0.0:
            scored.append((score, rows, table))

    if not scored:
        return None, 0.0

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # Try the top-3 tables
    for table_score, rows, _ in scored[:3]:
        result = _extract_field_from_rows(rows, field_name, scale, table_score)
        if result[0] is not None:
            return result

    return None, 0.0


def _extract_field_from_rows(
    rows: list[list[str]],
    field_name: str,
    scale: float,
    table_score: float,
) -> tuple[Optional[float], float]:
    """
    Given a parsed table (list of row/cell strings), find the target field value.
    """
    if not rows:
        return None, 0.0

    # Assume first row (or first few rows) are headers
    header_row = rows[0] if rows else []
    data_col = _pick_annual_column(header_row)

    best_value: Optional[float] = None
    best_match_score: float = 0.0

    for row in rows[1:]:
        if not row:
            continue
        label_cell = row[0]
        match_score = _label_matches(label_cell, field_name)
        if match_score < 0.5:
            continue

        # Try the preferred annual column first, then adjacent columns
        cols_to_try = [data_col] + [c for c in range(1, min(len(row), 5)) if c != data_col]
        for col in cols_to_try:
            if col >= len(row):
                continue
            val = _parse_numeric(row[col])
            if val is not None:
                if match_score > best_match_score:
                    best_match_score = match_score
                    best_value = val * scale
                break

    if best_value is None:
        return None, 0.0

    # Confidence: product of label match quality × table income-statement score
    confidence = min(0.95, best_match_score * (0.5 + table_score * 0.5))
    return best_value, confidence


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python html_extractor.py <filing.htm> <field_name>")
        print(f"Fields: {list(FIELDS.keys())}")
        sys.exit(1)

    html_path = Path(sys.argv[1])
    field = sys.argv[2]
    value, conf = extract_from_html(html_path, field)
    if value is not None:
        print(f"{field}: {value:,.2f}  (confidence={conf:.2f})")
    else:
        print(f"{field}: not found")
