"""
llm_extractor.py — Action 3: Extract financial values via Claude (LLM fallback).

This is the most expensive extractor — used when XBRL, HTML table, and regex all fail
or return low-confidence results. The RL agent should learn to call this sparingly.

Strategy:
  1. Find the income statement section in the filing (same as regex_extractor).
  2. Truncate to a context window (to control token cost).
  3. Send to Claude with a field-specific structured prompt.
  4. Parse the JSON response → (value, confidence).

Cost control:
  - Uses claude-haiku-4-5 by default (cheapest, fast enough for structured extraction).
  - The RL reward function applies a -0.2 latency penalty for slow calls (>5s).
  - Confidence is derived from Claude's self-reported certainty in the JSON output.

Returns (value: float | None, confidence: float).
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

FIELD_CONFIG_PATH = Path(__file__).parent / "field_config.json"
with open(FIELD_CONFIG_PATH) as f:
    _FIELD_CONFIG = json.load(f)

FIELDS: dict = _FIELD_CONFIG["fields"]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "claude-haiku-4-5-20251001"   # cheapest; upgrade to sonnet if needed
MAX_CONTEXT_CHARS = 8_000                      # ~2k tokens; keeps cost low
SYSTEM_PROMPT = """\
You are a financial data extraction assistant. You will be given a passage from an SEC 10-K filing \
and asked to extract one specific financial figure. Reply ONLY with valid JSON — no commentary, \
no markdown fences. If you cannot find the value with high confidence, return null for the value field.
"""

EXTRACTION_PROMPT_TEMPLATE = """\
Extract the following financial field from the passage below.

Field: {field_label}
Description: {description}
Expected unit: {unit}
Expected sign: {sign}

Passage:
---
{passage}
---

Reply with this exact JSON structure:
{{
  "value": <number or null>,
  "unit": "<USD | USD/shares | null>",
  "period": "<fiscal year end date YYYY-MM-DD or null>",
  "confidence": <0.0 to 1.0>,
  "reasoning": "<one sentence>"
}}
"""

# ---------------------------------------------------------------------------
# Passage extraction (reuse section finder from regex_extractor)
# ---------------------------------------------------------------------------

INCOME_STMT_ANCHORS = [
    r"consolidated\s+statements?\s+of\s+(operations|income|earnings)",
    r"statements?\s+of\s+(operations|income|earnings)",
    r"results\s+of\s+operations",
]


def _get_relevant_passage(doc_path: Path, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """
    Read a filing (TXT or HTML) and extract the income statement section.
    Strips HTML tags if the file is HTML.
    """
    suffix = doc_path.suffix.lower()
    try:
        raw = doc_path.read_text(errors="replace")
    except Exception as e:
        logger.warning("Could not read %s: %s", doc_path, e)
        return ""

    # Strip HTML if needed
    if suffix in (".htm", ".html"):
        try:
            from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
            import warnings
            warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
            raw = BeautifulSoup(raw, "lxml").get_text(" ", strip=True)
        except Exception:
            raw = re.sub(r"<[^>]+>", " ", raw)

    # Find income statement section
    for pattern in INCOME_STMT_ANCHORS:
        m = re.search(pattern, raw, re.IGNORECASE)
        if m:
            start = max(0, m.start() - 200)
            return raw[start: start + max_chars]

    # Fallback: return the middle of the document (most likely to have IS)
    mid = len(raw) // 3
    return raw[mid: mid + max_chars]


# ---------------------------------------------------------------------------
# Claude API call
# ---------------------------------------------------------------------------

def _call_claude(prompt: str, model: str = DEFAULT_MODEL) -> dict:
    """
    Call the Anthropic API and return the parsed JSON response.
    Raises on API error or JSON parse failure.
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic is required: pip install anthropic")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Add it to a .env file or export it."
        )

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=256,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = message.content[0].text.strip()

    # Strip markdown fences if present (model occasionally adds them)
    raw_text = re.sub(r"^```(?:json)?\n?", "", raw_text)
    raw_text = re.sub(r"\n?```$", "", raw_text)

    return json.loads(raw_text)


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

def extract_with_claude(
    doc_path: Path,
    field_name: str,
    model: str = DEFAULT_MODEL,
) -> tuple[Optional[float], float]:
    """
    Extract a financial field using Claude as a last-resort LLM extractor.

    Args:
        doc_path:   Path to the filing TXT or HTML file.
        field_name: One of the 12 fields defined in field_config.json.
        model:      Claude model ID to use.

    Returns:
        (value, confidence)  where confidence ∈ [0, 1].
        Returns (None, 0.0) on any failure.
    """
    field_cfg = FIELDS.get(field_name)
    if field_cfg is None:
        logger.warning("Unknown field: %s", field_name)
        return None, 0.0

    passage = _get_relevant_passage(doc_path)
    if not passage.strip():
        logger.warning("No usable passage from %s", doc_path)
        return None, 0.0

    prompt = EXTRACTION_PROMPT_TEMPLATE.format(
        field_label=field_cfg["label"],
        description=f"US-GAAP tag: {field_cfg['preferred_tag']}",
        unit=field_cfg["unit"],
        sign=field_cfg["sign"],
        passage=passage,
    )

    try:
        result = _call_claude(prompt, model=model)
    except json.JSONDecodeError as e:
        logger.warning("Claude returned non-JSON for %s: %s", field_name, e)
        return None, 0.0
    except Exception as e:
        logger.warning("Claude API error for %s: %s", field_name, e)
        return None, 0.0

    raw_value = result.get("value")
    confidence = float(result.get("confidence", 0.0))

    if raw_value is None:
        return None, 0.0

    try:
        value = float(raw_value)
    except (ValueError, TypeError):
        logger.warning("Claude returned non-numeric value for %s: %r", field_name, raw_value)
        return None, 0.0

    # Cap confidence — LLM self-reports are optimistic
    confidence = min(confidence, 0.85)

    logger.debug(
        "Claude extracted %s=%.0f (conf=%.2f, reasoning=%s)",
        field_name, value, confidence, result.get("reasoning", "")
    )
    return value, confidence


# ---------------------------------------------------------------------------
# Batch extraction (for golden set creation)
# ---------------------------------------------------------------------------

def extract_all_fields_with_claude(
    doc_path: Path,
    model: str = DEFAULT_MODEL,
) -> dict[str, tuple[Optional[float], float]]:
    """
    Extract all 12 fields from a single filing in one pass.
    More efficient than calling extract_with_claude 12 times because
    we only fetch the passage once.

    Returns {field_name: (value, confidence)}.
    """
    passage = _get_relevant_passage(doc_path)
    if not passage.strip():
        return {f: (None, 0.0) for f in FIELDS}

    # Build a combined prompt for all 12 fields
    fields_list = "\n".join(
        f'  "{name}": {{"label": "{cfg["label"]}", "tag": "{cfg["preferred_tag"]}"}}'
        for name, cfg in FIELDS.items()
    )

    batch_prompt = f"""\
Extract ALL of the following financial fields from this 10-K income statement passage.
For each field, return the most recent annual value in USD (or USD/share for EPS).

Fields to extract:
{{
{fields_list}
}}

Passage:
---
{passage}
---

Reply with ONLY valid JSON in this exact structure:
{{
  "revenue":          {{"value": <number or null>, "confidence": <0-1>}},
  "cogs":             {{"value": <number or null>, "confidence": <0-1>}},
  "gross_profit":     {{"value": <number or null>, "confidence": <0-1>}},
  "operating_income": {{"value": <number or null>, "confidence": <0-1>}},
  "net_income":       {{"value": <number or null>, "confidence": <0-1>}},
  "total_expenses":   {{"value": <number or null>, "confidence": <0-1>}},
  "interest_expense": {{"value": <number or null>, "confidence": <0-1>}},
  "income_tax_expense":{{"value": <number or null>, "confidence": <0-1>}},
  "eps_basic":        {{"value": <number or null>, "confidence": <0-1>}},
  "eps_diluted":      {{"value": <number or null>, "confidence": <0-1>}},
  "rd_expense":       {{"value": <number or null>, "confidence": <0-1>}},
  "sga_expense":      {{"value": <number or null>, "confidence": <0-1>}}
}}
"""

    try:
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set")

        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": batch_prompt}],
        )
        raw = message.content[0].text.strip()
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        parsed = json.loads(raw)
    except Exception as e:
        logger.error("Batch Claude extraction failed: %s", e)
        return {f: (None, 0.0) for f in FIELDS}

    results = {}
    for field_name in FIELDS:
        entry = parsed.get(field_name, {})
        raw_val = entry.get("value")
        conf = min(float(entry.get("confidence", 0.0)), 0.85)
        try:
            val = float(raw_val) if raw_val is not None else None
        except (ValueError, TypeError):
            val = None
        results[field_name] = (val, conf)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv()

    if len(sys.argv) < 2:
        print("Usage: python llm_extractor.py <filing.txt|filing.htm> [field_name]")
        print(f"       If field_name omitted, extracts all 12 fields.")
        sys.exit(1)

    doc_path = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        field = sys.argv[2]
        value, conf = extract_with_claude(doc_path, field)
        if value is not None:
            print(f"{field}: {value:,.2f}  (confidence={conf:.2f})")
        else:
            print(f"{field}: not found")
    else:
        results = extract_all_fields_with_claude(doc_path)
        print(f"\n{'Field':<25} {'Value':>20} {'Confidence':>12}")
        print("-" * 60)
        for fname, (val, conf) in results.items():
            val_str = f"{val:,.2f}" if val is not None else "—"
            print(f"{fname:<25} {val_str:>20} {conf:>12.2f}")
