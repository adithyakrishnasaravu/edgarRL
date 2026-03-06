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

    Scoring rules:
    • Exact match                                         → 1.0
    • Label is a substring of cell (cell is more verbose) → 0.85
    • Cell is a substring of label (label is more verbose) → penalised by length ratio
    • Short label (≤ 12 chars) that appears in cell        → boost: score = 0.7
      (handles "Basic" / "Diluted" at end of long EPS labels)
    """
    norm = _normalize(cell_text)
    labels = FIELDS.get(field_name, {}).get("html_labels", [])
    best = 0.0
    for label in labels:
        nl = _normalize(label)
        if norm == nl:
            best = max(best, 1.0)
        elif nl in norm:
            # Label is a substring of the (longer) cell text.
            if len(nl) <= 12:
                # Short keyword label (e.g. "basic") present in cell — strong signal
                best = max(best, 0.70)
            elif norm.endswith(nl) or norm.startswith(nl):
                # Label anchored to start/end of cell (e.g. "...—basic and diluted")
                best = max(best, 0.75)
            else:
                best = max(best, len(nl) / max(len(norm), 1) * 0.85)
        elif norm in nl:
            # Cell text is a substring of the (longer) label.
            # Only give a meaningful score if the cell is long enough to be specific
            # (short cell fragments like "net" appear in many labels → false positives)
            if len(norm) >= 8:
                best = max(best, len(norm) / max(len(nl), 1) * 0.75)
    return best


# ---------------------------------------------------------------------------
# Value parsing
# ---------------------------------------------------------------------------

_MULTIPLIER_RE = re.compile(
    r"\(\s*(?:dollars?\s+in\s+|\$\s*in\s+|amounts?\s+in\s+|in\s+)?(thousands|millions|billions)\b",
    re.IGNORECASE,
)
# Fallback for unparenthesized scale notes (e.g. "dollars in millions" in table headers)
_MULTIPLIER_LOOSE_RE = re.compile(
    r"(?:dollars?\s+in\s+|\$\s*in\s+|amounts?\s+in\s+|in\s+)(thousands|millions|billions)\b",
    re.IGNORECASE,
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


def _detect_scale_from_ixbrl(raw_html: str) -> Optional[float]:
    """
    Parse iXBRL inline tags to find the dominant USD scale used in the filing.
    Returns the multiplier (e.g. 1_000_000 for scale=6) or None if not found.

    Modern 10-K HTML filings are inline XBRL; the `scale` attribute on
    <ix:nonFraction> encodes 10^scale (6 = millions, 3 = thousands, 9 = billions).
    We find the most common scale across all USD-denominated numeric tags.
    """
    from collections import Counter
    scales: list[int] = []
    for attrs in re.finditer(r"<ix:nonFraction\s+([^>]+)>", raw_html, re.IGNORECASE):
        a = attrs.group(1)
        unit_m = re.search(r'unitRef="([^"]+)"', a, re.IGNORECASE)
        scale_m = re.search(r'scale="(-?\d+)"', a, re.IGNORECASE)
        if unit_m and scale_m:
            unit = unit_m.group(1).lower()
            # Only dollar-denominated (not per-share, not pure count)
            if unit == "usd":
                scales.append(int(scale_m.group(1)))
    if not scales:
        return None
    dominant_scale = Counter(scales).most_common(1)[0][0]
    return float(10 ** dominant_scale)


def _detect_scale(soup, raw_html: str = "") -> float:
    """
    Detect the reporting scale (thousands / millions / billions).
    Returns the multiplier to apply to raw table values.

    Priority:
      1. iXBRL scale attributes (authoritative for inline XBRL filings).
      2. Parenthesized scale note near the income statement section.
      3. Full-document text search with the parenthesized pattern.
    """
    # 1. iXBRL — most reliable
    if raw_html:
        ixbrl_scale = _detect_scale_from_ixbrl(raw_html)
        if ixbrl_scale is not None:
            return ixbrl_scale

    _SCALE_MAP = {"thousands": 1_000, "millions": 1_000_000, "billions": 1_000_000_000}
    text = soup.get_text(" ", strip=True)

    # 2. Parenthesized scale note near the income statement section
    for anchor_pat in [
        r"consolidated\s+statements?\s+of\s+(operations|income|earnings)",
        r"statements?\s+of\s+(operations|income|earnings)",
        r"results\s+of\s+operations",
    ]:
        m = re.search(anchor_pat, text, re.IGNORECASE)
        if m:
            region = text[max(0, m.start() - 500): m.start() + 5000]
            for pat in (_MULTIPLIER_RE, _MULTIPLIER_LOOSE_RE):
                sm = pat.search(region)
                if sm:
                    word = sm.group(1).lower()
                    return _SCALE_MAP[word]

    # 3. Full-document fallback with strict (parenthesized) pattern only
    for pat in (_MULTIPLIER_RE, _MULTIPLIER_LOOSE_RE):
        m = pat.search(text)
        if m:
            word = m.group(1).lower()
            return _SCALE_MAP[word]

    return 1.0  # assume units already in USD if not specified


# ---------------------------------------------------------------------------
# Table scoring and column selection
# ---------------------------------------------------------------------------

def _score_table(rows: list[list[str]]) -> float:
    """
    Score a parsed table by income-statement signal density.
    Penalizes percentage-only tables (e.g. 'as a % of revenue' breakdowns)
    that contain no dollar values.
    """
    all_text = " ".join(cell for row in rows for cell in row)
    all_lower = all_text.lower()
    hits = sum(1 for sig in INCOME_STATEMENT_SIGNALS if sig in all_lower)
    base_score = hits / len(INCOME_STATEMENT_SIGNALS)

    # Penalise tables that look like percentage breakdowns: many "%" but no "$"
    pct_count = all_text.count("%")
    dollar_count = all_text.count("$")
    if pct_count >= 3 and dollar_count == 0:
        base_score *= 0.2  # heavy penalty — this is almost certainly a % table

    return base_score


def _pick_annual_column(rows: list[list[str]]) -> tuple[int, int]:
    """
    Return (header_col, header_width) for the most recent annual period.

    Scans the first 4 rows for year-labelled cells.  The caller should use
    header_width and the actual data-row width to compute the real column
    index per row (handles colspan expansion where each table is different).
    """
    best_col = 1
    best_year = 0
    best_header_width = 1

    for row in rows[:4]:
        for i, cell in enumerate(row):
            m = re.search(r"(20\d{2}|19\d{2})", cell)
            if m:
                year = int(m.group(1))
                if year > best_year:
                    best_year = year
                    best_col = i
                    best_header_width = len(row)

    return best_col, best_header_width


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
        raw_str = raw.decode("utf-8", errors="replace")
        soup = BeautifulSoup(raw, "lxml")
    except Exception as e:
        logger.warning("HTML parse error for %s: %s", html_path, e)
        return None, 0.0

    scale = _detect_scale(soup, raw_html=raw_str)

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

    # Per-share fields (EPS) are reported in USD/share, never scaled
    field_unit = FIELDS.get(field_name, {}).get("unit", "USD")
    effective_scale = 1.0 if field_unit == "USD/shares" else scale

    # Try the top-3 tables
    for table_score, rows, _ in scored[:3]:
        result = _extract_field_from_rows(rows, field_name, effective_scale, table_score)
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

    # header_col is in the header row's coordinate space;
    # actual data column depends on each row's width (colspan expansion varies per row)
    header_col, header_width = _pick_annual_column(rows)

    best_value: Optional[float] = None
    best_match_score: float = 0.0

    for row in rows[1:]:
        if not row:
            continue
        label_cell = row[0]
        match_score = _label_matches(label_cell, field_name)
        if match_score < 0.5:
            continue

        # Per-row colspan expansion mapping:
        #   Some rows have more cells than the header due to colspan.
        #   Re-compute the target column for this specific row's width.
        row_width = len(row)
        if header_width > 1 and row_width > 1:
            expansion = (row_width - 1) / max(header_width - 1, 1)
            data_col = round(1 + (header_col - 1) * expansion)
        else:
            data_col = header_col

        # Search order: annual column, then +1 (handles $-sign as separate cell),
        # then all other columns as fallback
        anchor_cols = [data_col, data_col + 1]
        other_cols = [c for c in range(1, row_width) if c not in anchor_cols]
        cols_to_try = anchor_cols + other_cols

        for col in cols_to_try:
            if col >= row_width:
                continue
            val = _parse_numeric(row[col])
            if val is not None:
                if match_score > best_match_score:
                    best_match_score = match_score
                    best_value = val * scale
                break

    if best_value is None:
        return None, 0.0

    # --- Guardrails ---
    field_cfg = FIELDS.get(field_name, {})
    field_sign = field_cfg.get("sign", "positive")

    # Fields with sign="positive" should never be negative after extraction.
    # EDGAR HTML often shows costs parenthesised as deductions; absolute value fixes it.
    if field_sign == "positive" and best_value < 0:
        best_value = abs(best_value)

    # EPS sanity: scale should NOT be applied to per-share values.
    # (Already handled at the table level, but guard against edge cases.)
    if field_cfg.get("unit") == "USD/shares" and abs(best_value) > 1_000:
        # Looks like scale was incorrectly applied; reverse it
        best_value = best_value / scale if scale != 1.0 else best_value

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
