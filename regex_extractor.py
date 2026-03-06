"""
regex_extractor.py — Action 2: Extract financial values via regex on raw TXT filings.

The SEC full-submission TXT file is a flat SGML/text dump of the entire filing.
It's messy but consistent enough for regex: labels appear near dollar amounts
in predictable patterns from the income statement section.

Strategy:
  1. Find the income statement section by anchor text patterns.
  2. Within a window around each label match, find the nearest dollar amount.
  3. Score by proximity and label confidence.
  4. Detect scale (thousands/millions) from surrounding context.

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
# Income statement section anchors
# These are searched (case-insensitive) to find where to focus regex extraction
# ---------------------------------------------------------------------------

INCOME_STMT_ANCHORS = [
    r"consolidated\s+statements?\s+of\s+(operations|income|earnings)",
    r"statements?\s+of\s+(operations|income|earnings)",
    r"results\s+of\s+operations",
    r"income\s+from\s+operations",
]

# Window size (characters) around section anchor to search for values
SECTION_WINDOW = 15_000

# Window size around a label match to look for the adjacent value
LABEL_VALUE_WINDOW = 400


# ---------------------------------------------------------------------------
# Scale detection
# ---------------------------------------------------------------------------

_SCALE_RE = re.compile(
    r"in\s+(thousands|millions|billions)",
    re.IGNORECASE,
)

def _detect_scale(text: str) -> float:
    m = _SCALE_RE.search(text[:3000])
    if m:
        return {"thousands": 1_000, "millions": 1_000_000, "billions": 1_000_000_000}[
            m.group(1).lower()
        ]
    return 1.0


# ---------------------------------------------------------------------------
# Value regex — matches: $1,234  (1,234)  1,234.5  (1,234.5)
# ---------------------------------------------------------------------------

_VALUE_PATTERN = re.compile(
    r"[\$\(]?\s*([\d]{1,3}(?:,\d{3})*(?:\.\d+)?)\s*\)?"
)

_NEGATIVE_PATTERN = re.compile(
    r"\(\s*([\d]{1,3}(?:,\d{3})*(?:\.\d+)?)\s*\)"
)


def _parse_value(text: str) -> Optional[float]:
    """Extract a numeric value from a matched text fragment."""
    text = text.strip()
    if not text or text in ("-", "—", "–", "*", "N/A"):
        return None

    # Negative in parentheses
    m_neg = _NEGATIVE_PATTERN.fullmatch(text.strip())
    if m_neg:
        try:
            return -float(m_neg.group(1).replace(",", ""))
        except ValueError:
            pass

    m = _VALUE_PATTERN.search(text)
    if m:
        is_negative = text.strip().startswith("(") and ")" in text
        try:
            val = float(m.group(1).replace(",", ""))
            return -val if is_negative else val
        except ValueError:
            pass
    return None


# ---------------------------------------------------------------------------
# Section finder
# ---------------------------------------------------------------------------

def _find_income_stmt_section(text: str) -> str:
    """
    Return the substring of text most likely to be the income statement.
    Falls back to the full text if no anchor is found.
    """
    for pattern in INCOME_STMT_ANCHORS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            start = m.start()
            return text[start: start + SECTION_WINDOW]
    # Fallback: search everywhere (slower, noisier)
    return text


# ---------------------------------------------------------------------------
# Label → value extraction
# ---------------------------------------------------------------------------

def _build_label_patterns(field_name: str) -> list[re.Pattern]:
    """Build ordered list of regex patterns for a field's HTML labels."""
    labels = FIELDS.get(field_name, {}).get("html_labels", [])
    patterns = []
    for label in labels:
        # Escape and allow flexible whitespace between words
        escaped = re.sub(r"\s+", r"\\s+", re.escape(label))
        patterns.append(re.compile(escaped, re.IGNORECASE))
    return patterns


def _find_value_near_label(
    section: str,
    label_match: re.Match,
    scale: float,
) -> tuple[Optional[float], float]:
    """
    Given a label match in `section`, search a window after the match
    for the nearest dollar value. Returns (value, proximity_score).
    """
    start = label_match.end()
    window = section[start: start + LABEL_VALUE_WINDOW]

    # Collect all value candidates with their position
    candidates: list[tuple[int, float]] = []

    for m in _VALUE_PATTERN.finditer(window):
        raw = window[max(0, m.start() - 5): m.end() + 5]
        val = _parse_value(raw)
        if val is not None and abs(val) > 0:
            candidates.append((m.start(), val))

    if not candidates:
        return None, 0.0

    # Prefer the closest value (fewest characters from label end)
    pos, val = min(candidates, key=lambda x: x[0])
    proximity = max(0.0, 1.0 - pos / LABEL_VALUE_WINDOW)

    return val * scale, proximity


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

def extract_from_txt(
    txt_path: Path,
    field_name: str,
    max_file_mb: float = 50.0,
) -> tuple[Optional[float], float]:
    """
    Extract a financial field value from a raw SEC TXT submission file.

    Returns:
        (value, confidence)  where confidence ∈ [0, 1].
        Returns (None, 0.0) on failure.
    """
    if not txt_path.exists():
        return None, 0.0

    # Guard against huge files
    size_mb = txt_path.stat().st_size / 1_048_576
    if size_mb > max_file_mb:
        logger.warning("TXT file too large (%.1f MB), reading first 50MB only", size_mb)

    try:
        # Read up to 50MB; most 10-K TXT files are 1–10MB
        with txt_path.open("r", errors="replace") as f:
            text = f.read(50 * 1_048_576)
    except Exception as e:
        logger.warning("Could not read %s: %s", txt_path, e)
        return None, 0.0

    scale = _detect_scale(text)
    section = _find_income_stmt_section(text)
    patterns = _build_label_patterns(field_name)

    if not patterns:
        return None, 0.0

    best_value: Optional[float] = None
    best_confidence: float = 0.0

    for rank, pattern in enumerate(patterns):
        # Rank 0 = preferred label, higher rank = fallback
        label_weight = max(0.4, 1.0 - rank * 0.15)

        for m in pattern.finditer(section):
            val, proximity = _find_value_near_label(section, m, scale)
            if val is None:
                continue

            # Skip obviously wrong values (e.g. page numbers, dates)
            if abs(val) < 1:
                continue

            confidence = label_weight * (0.4 + proximity * 0.6)
            if confidence > best_confidence:
                best_confidence = confidence
                best_value = val

        # If we found a high-confidence match on the preferred label, stop early
        if best_confidence >= 0.75:
            break

    return best_value, min(best_confidence, 0.90)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python regex_extractor.py <filing.txt> <field_name>")
        print(f"Fields: {list(FIELDS.keys())}")
        sys.exit(1)

    txt_path = Path(sys.argv[1])
    field = sys.argv[2]
    value, conf = extract_from_txt(txt_path, field)
    if value is not None:
        print(f"{field}: {value:,.2f}  (confidence={conf:.2f})")
    else:
        print(f"{field}: not found")
