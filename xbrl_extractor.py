"""
xbrl_extractor.py — Ground truth oracle for EDGAR 10-K fields.

Uses arelle to parse XBRL instance documents and extract values
for the 12 target fields defined in field_config.json.

This is ACTION 0 (xbrl_direct) in the RL action space and also
the source of reward signals for all other extractors.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FIELD_CONFIG_PATH = Path(__file__).parent / "field_config.json"

with open(FIELD_CONFIG_PATH) as f:
    _FIELD_CONFIG = json.load(f)

FIELDS: dict = _FIELD_CONFIG["fields"]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ExtractionResult:
    field_name: str
    value: Optional[float]
    unit: Optional[str]
    period: Optional[str]          # e.g. "2023-12-31" (instant) or "2023-01-01/2023-12-31"
    xbrl_tag: Optional[str]        # which tag actually matched
    confidence: float              # 0.0–1.0
    source: str = "xbrl_direct"
    error: Optional[str] = None


@dataclass
class FilingXBRL:
    """Parsed representation of a single 10-K filing's XBRL data."""
    accession: str
    cik: str
    fiscal_year_end: Optional[str]
    facts: dict[str, list[dict]]   # tag → list of {value, unit, period, decimals}


# ---------------------------------------------------------------------------
# Parser (arelle-based)
# ---------------------------------------------------------------------------

def _strip_prefix(tag: str) -> str:
    """'us-gaap:Revenues' → 'Revenues'"""
    return tag.split(":")[-1] if ":" in tag else tag


def load_xbrl_filing(xbrl_path: Path) -> Optional[FilingXBRL]:
    """
    Parse an XBRL instance document (.xml or .htm) using arelle.
    Returns a FilingXBRL with all numeric facts, or None on failure.
    """
    try:
        from arelle import Cntlr, ModelXbrl, XbrlConst  # noqa: F401
    except ImportError:
        raise ImportError("arelle-release is required: pip install arelle-release")

    try:
        ctrl = Cntlr.Cntlr(logFileName="logToStdErr")
        model = ctrl.modelManager.load(str(xbrl_path))

        if model is None:
            logger.warning("arelle returned None for %s", xbrl_path)
            return None

        facts: dict[str, list[dict]] = {}

        for fact in model.facts:
            if fact.concept is None or fact.value is None:
                continue
            # Only numeric facts
            if not fact.concept.isNumeric:
                continue

            clark_name = fact.concept.qname.clarkNotation  # {namespace}LocalName
            # Build a us-gaap:LocalName style key
            local = fact.concept.qname.localName
            ns = fact.concept.qname.namespaceURI or ""
            if "us-gaap" in ns:
                tag_key = f"us-gaap:{local}"
            elif "dei" in ns:
                tag_key = f"dei:{local}"
            else:
                tag_key = local

            try:
                numeric_value = float(fact.value)
            except (ValueError, TypeError):
                continue

            # Period
            ctx = fact.context
            period_str: Optional[str] = None
            if ctx is not None:
                if ctx.isInstantPeriod:
                    period_str = str(ctx.instantDatetime.date()) if ctx.instantDatetime else None
                elif ctx.isStartEndPeriod:
                    start = str(ctx.startDatetime.date()) if ctx.startDatetime else "?"
                    end = str(ctx.endDatetime.date()) if ctx.endDatetime else "?"
                    period_str = f"{start}/{end}"

            unit_str: Optional[str] = None
            if fact.unit is not None:
                unit_str = str(fact.unit)

            entry = {
                "value": numeric_value,
                "unit": unit_str,
                "period": period_str,
                "decimals": fact.decimals,
                "clark_name": clark_name,
            }
            facts.setdefault(tag_key, []).append(entry)

        # Best-effort accession / CIK from path
        accession = _infer_accession(xbrl_path)
        cik = _infer_cik(xbrl_path)

        ctrl.close()
        return FilingXBRL(
            accession=accession,
            cik=cik,
            fiscal_year_end=None,  # populated later if needed
            facts=facts,
        )

    except Exception as exc:
        logger.error("Failed to parse XBRL %s: %s", xbrl_path, exc)
        return None


# ---------------------------------------------------------------------------
# Fallback: parse XBRL JSON (EDGAR company-facts API)
# ---------------------------------------------------------------------------

def load_xbrl_from_company_facts(
    facts_json: dict,
    cik: str,
    accession: Optional[str] = None,
    fiscal_year: Optional[int] = None,
) -> FilingXBRL:
    """
    Parse the EDGAR company-facts JSON (from
    https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json)
    into a FilingXBRL object.

    If accession is provided, only facts from that filing are included.
    If fiscal_year is provided (and accession is not), facts from that FY are included.
    Without either filter, all 10-K facts are included (multi-year, prefer most recent).
    """
    us_gaap = facts_json.get("facts", {}).get("us-gaap", {})
    facts: dict[str, list[dict]] = {}

    # Normalize accession for comparison (EDGAR uses dashes)
    accn_filter: Optional[str] = None
    if accession:
        # Ensure dashed format: 0000320193-25-000079
        accn_filter = accession if "-" in accession else (
            f"{accession[:10]}-{accession[10:12]}-{accession[12:]}"
            if len(accession) == 18 else accession
        )

    for concept_name, concept_data in us_gaap.items():
        tag_key = f"us-gaap:{concept_name}"
        units_map: dict = concept_data.get("units", {})
        for unit_label, entries in units_map.items():
            for entry in entries:
                form = entry.get("form", "")
                if "10-K" not in form:
                    continue

                # Accession filter
                if accn_filter and entry.get("accn") != accn_filter:
                    continue

                # Fiscal year filter (only when no accession filter)
                if not accn_filter and fiscal_year:
                    if entry.get("fy") != fiscal_year:
                        continue

                try:
                    numeric_value = float(entry["val"])
                except (KeyError, ValueError, TypeError):
                    continue

                period_str: Optional[str] = None
                start = entry.get("start")
                end = entry.get("end")
                if start and end:
                    period_str = f"{start}/{end}"
                elif end:
                    period_str = end

                facts.setdefault(tag_key, []).append({
                    "value": numeric_value,
                    "unit": unit_label,
                    "period": period_str,
                    "decimals": entry.get("decimals"),
                    "accession": entry.get("accn"),
                    "fiscal_year": entry.get("fy"),
                    "fiscal_period": entry.get("fp"),
                    "form": form,
                })

    return FilingXBRL(
        accession=accn_filter or "",
        cik=cik,
        fiscal_year_end=None,
        facts=facts,
    )


def load_xbrl_from_facts_file(
    json_path: Path,
    cik: str,
    accession: Optional[str] = None,
    fiscal_year: Optional[int] = None,
) -> Optional[FilingXBRL]:
    """Load a company-facts JSON file from disk and parse it."""
    try:
        with open(json_path) as f:
            data = json.load(f)
        return load_xbrl_from_company_facts(data, cik, accession=accession, fiscal_year=fiscal_year)
    except Exception as e:
        logger.error("Failed to load company-facts JSON %s: %s", json_path, e)
        return None


# ---------------------------------------------------------------------------
# Field extractor
# ---------------------------------------------------------------------------

def _pick_best_fact(
    candidates: list[dict],
    target_period: Optional[str] = None,
    prefer_annual: bool = True,
) -> Optional[dict]:
    """
    From a list of fact entries for one tag, pick the most relevant one.
    Priority: matching period > annual duration (≈365 days) > most recent.
    """
    if not candidates:
        return None

    if target_period:
        exact = [c for c in candidates if c.get("period") == target_period]
        if exact:
            return exact[-1]

    # Prefer annual duration facts (duration ≈ 365 days)
    annual = []
    for c in candidates:
        period = c.get("period", "")
        if "/" in period:
            try:
                from datetime import date
                parts = period.split("/")
                start = date.fromisoformat(parts[0])
                end = date.fromisoformat(parts[1])
                days = (end - start).days
                if 330 <= days <= 400:
                    annual.append(c)
            except ValueError:
                pass

    pool = annual if annual else candidates
    # Return the most recent (last in list, or by period end date)
    return pool[-1]


def extract_field(
    filing: FilingXBRL,
    field_name: str,
    target_period: Optional[str] = None,
) -> ExtractionResult:
    """
    Extract a single field from a parsed FilingXBRL using the tag priority
    list defined in field_config.json.

    Returns ExtractionResult — check .error for failures.
    """
    if field_name not in FIELDS:
        return ExtractionResult(
            field_name=field_name,
            value=None,
            unit=None,
            period=None,
            xbrl_tag=None,
            confidence=0.0,
            error=f"Unknown field: {field_name}",
        )

    cfg = FIELDS[field_name]
    tags: list[str] = cfg["xbrl_tags"]

    for tag in tags:
        candidates = filing.facts.get(tag, [])
        if not candidates:
            # Also try without namespace prefix (some arelle outputs)
            local = _strip_prefix(tag)
            candidates = filing.facts.get(local, [])

        if not candidates:
            continue

        best = _pick_best_fact(candidates, target_period=target_period)
        if best is None:
            continue

        value = best["value"]
        # Confidence: preferred tag = 1.0, fallback tags decay slightly
        tag_rank = tags.index(tag)
        confidence = max(0.6, 1.0 - tag_rank * 0.1)

        return ExtractionResult(
            field_name=field_name,
            value=value,
            unit=best.get("unit"),
            period=best.get("period"),
            xbrl_tag=tag,
            confidence=confidence,
        )

    return ExtractionResult(
        field_name=field_name,
        value=None,
        unit=None,
        period=None,
        xbrl_tag=None,
        confidence=0.0,
        error=f"No XBRL tags found for field '{field_name}'",
    )


def extract_all_fields(
    filing: FilingXBRL,
    target_period: Optional[str] = None,
) -> dict[str, ExtractionResult]:
    """Extract all 12 fields from a filing. Returns {field_name: ExtractionResult}."""
    return {
        field_name: extract_field(filing, field_name, target_period=target_period)
        for field_name in FIELDS
    }


# ---------------------------------------------------------------------------
# Reward helper — used by env.py
# ---------------------------------------------------------------------------

REWARD_CFG = _FIELD_CONFIG["reward"]


def compute_reward(
    extracted: Optional[float],
    ground_truth: Optional[float],
    action_index: int,
    latency_seconds: float = 0.0,
) -> float:
    """
    Score an extracted value against the XBRL ground truth.

    Returns a scalar reward in roughly [-1.0, 1.0].
    action_index 3 (llm_claude) incurs a latency penalty if slow.
    """
    if ground_truth is None:
        # Can't score — neutral
        return 0.0

    if extracted is None:
        return REWARD_CFG["null_or_missing"]

    if not isinstance(extracted, (int, float)):
        return REWARD_CFG["type_mismatch"]

    if ground_truth == 0:
        rel_error = abs(extracted - ground_truth)
    else:
        rel_error = abs(extracted - ground_truth) / abs(ground_truth)

    if rel_error == 0.0:
        reward = REWARD_CFG["exact_match"]
    elif rel_error <= 0.01:
        reward = REWARD_CFG["within_1pct"]
    elif _same_order_of_magnitude(extracted, ground_truth):
        reward = REWARD_CFG["correct_magnitude"]
    else:
        reward = REWARD_CFG["null_or_missing"]

    # Latency penalty for slow LLM calls (action 3)
    if action_index == 3 and latency_seconds > 5.0:
        reward += REWARD_CFG["high_latency_penalty"]

    return reward


def _same_order_of_magnitude(a: float, b: float) -> bool:
    import math
    if a == 0 or b == 0:
        return False
    ratio = abs(a) / abs(b)
    return 0.1 <= ratio <= 10.0


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _infer_accession(path: Path) -> str:
    """Try to parse accession number from path like .../0000320193-23-000077/..."""
    for part in path.parts:
        m = re.match(r"(\d{10}-\d{2}-\d{6})", part)
        if m:
            return m.group(1)
    return path.stem


def _infer_cik(path: Path) -> str:
    for part in path.parts:
        m = re.match(r"^(\d{7,10})$", part)
        if m:
            return m.group(1)
    return ""


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python xbrl_extractor.py <path/to/filing.xml>")
        print("       python xbrl_extractor.py --test-reward")
        sys.exit(1)

    if sys.argv[1] == "--test-reward":
        tests = [
            (100_000_000, 100_000_000, 0, 0.0, "exact"),
            (100_000_000, 100_500_000, 0, 0.0, "within 1%"),
            (95_000_000,  100_000_000, 0, 0.0, "correct OOM"),
            (None,        100_000_000, 0, 0.0, "null"),
            (1_000,       100_000_000, 0, 0.0, "way off"),
        ]
        for extracted, gt, action, latency, label in tests:
            r = compute_reward(extracted, gt, action, latency)
            print(f"  [{label:20s}] reward = {r:+.2f}")
        sys.exit(0)

    xbrl_path = Path(sys.argv[1])
    filing = load_xbrl_filing(xbrl_path)
    if filing is None:
        print("Failed to load filing.")
        sys.exit(1)

    results = extract_all_fields(filing)
    print(f"\nFiling: {xbrl_path.name}  (CIK={filing.cik})\n")
    print(f"{'Field':<25} {'Value':>20} {'Tag':<55} {'Conf':>5}")
    print("-" * 110)
    for fname, res in results.items():
        val_str = f"{res.value:,.0f}" if res.value is not None else "—"
        tag_str = res.xbrl_tag or (f"ERROR: {res.error}" if res.error else "—")
        print(f"{fname:<25} {val_str:>20} {tag_str:<55} {res.confidence:>5.2f}")
