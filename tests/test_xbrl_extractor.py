"""
Tests for xbrl_extractor.py — ground truth oracle, reward function, and field extraction.
"""

import math
from datetime import date
from pathlib import Path

import pytest

from xbrl_extractor import (
    ExtractionResult,
    FilingXBRL,
    FIELDS,
    _strip_prefix,
    _infer_accession,
    _infer_cik,
    _pick_best_fact,
    _same_order_of_magnitude,
    extract_field,
    extract_all_fields,
    compute_reward,
    load_xbrl_from_company_facts,
    load_xbrl_from_facts_file,
)


# ---------------------------------------------------------------------------
# ExtractionResult dataclass
# ---------------------------------------------------------------------------

class TestExtractionResult:
    def test_construction(self):
        r = ExtractionResult(
            field_name="revenue",
            value=100.0,
            unit="USD",
            period="2024-01-01/2024-12-31",
            xbrl_tag="us-gaap:Revenues",
            confidence=0.95,
        )
        assert r.field_name == "revenue"
        assert r.value == 100.0
        assert r.source == "xbrl_direct"
        assert r.error is None

    def test_defaults(self):
        r = ExtractionResult(
            field_name="cogs", value=None, unit=None, period=None,
            xbrl_tag=None, confidence=0.0, error="not found",
        )
        assert r.source == "xbrl_direct"
        assert r.error == "not found"


# ---------------------------------------------------------------------------
# FilingXBRL dataclass
# ---------------------------------------------------------------------------

class TestFilingXBRL:
    def test_construction(self, sample_filing_xbrl):
        assert sample_filing_xbrl.accession == "0000320193-24-000123"
        assert sample_filing_xbrl.cik == "0000320193"
        assert len(sample_filing_xbrl.facts) > 0

    def test_empty(self, empty_filing_xbrl):
        assert empty_filing_xbrl.facts == {}
        assert empty_filing_xbrl.fiscal_year_end is None


# ---------------------------------------------------------------------------
# _strip_prefix
# ---------------------------------------------------------------------------

class TestStripPrefix:
    def test_us_gaap(self):
        assert _strip_prefix("us-gaap:Revenues") == "Revenues"

    def test_dei(self):
        assert _strip_prefix("dei:EntityName") == "EntityName"

    def test_no_prefix(self):
        assert _strip_prefix("Revenues") == "Revenues"

    def test_multiple_colons(self):
        assert _strip_prefix("a:b:c") == "c"


# ---------------------------------------------------------------------------
# _infer_accession
# ---------------------------------------------------------------------------

class TestInferAccession:
    def test_with_accession_in_path(self):
        p = Path("/data/filings/0000320193/0000320193-23-000077/filing.xml")
        assert _infer_accession(p) == "0000320193-23-000077"

    def test_without_accession(self):
        p = Path("/data/filings/some_file.xml")
        assert _infer_accession(p) == "some_file"

    def test_stem_fallback(self):
        p = Path("report.xml")
        assert _infer_accession(p) == "report"


# ---------------------------------------------------------------------------
# _infer_cik
# ---------------------------------------------------------------------------

class TestInferCik:
    def test_with_cik_in_path(self):
        p = Path("/data/filings/0000320193/filing.xml")
        assert _infer_cik(p) == "0000320193"

    def test_without_cik(self):
        p = Path("/data/filings/report.xml")
        assert _infer_cik(p) == ""

    def test_short_number_not_cik(self):
        # CIK must be 7-10 digits
        p = Path("/data/12345/file.xml")
        assert _infer_cik(p) == ""

    def test_seven_digit_cik(self):
        p = Path("/data/1234567/file.xml")
        assert _infer_cik(p) == "1234567"


# ---------------------------------------------------------------------------
# _pick_best_fact
# ---------------------------------------------------------------------------

class TestPickBestFact:
    def test_empty_candidates(self):
        assert _pick_best_fact([]) is None

    def test_exact_period_match(self):
        candidates = [
            {"value": 100, "period": "2023-01-01/2023-12-31"},
            {"value": 200, "period": "2024-01-01/2024-12-31"},
        ]
        result = _pick_best_fact(candidates, target_period="2024-01-01/2024-12-31")
        assert result["value"] == 200

    def test_annual_duration_preferred(self):
        candidates = [
            {"value": 50, "period": "2024-01-01/2024-03-31"},   # quarterly (~90 days)
            {"value": 200, "period": "2024-01-01/2024-12-31"},  # annual (~365 days)
            {"value": 75, "period": "2024-07-01/2024-09-30"},   # quarterly
        ]
        result = _pick_best_fact(candidates)
        assert result["value"] == 200

    def test_no_annual_returns_last(self):
        candidates = [
            {"value": 50, "period": "2024-01-01/2024-03-31"},
            {"value": 75, "period": "2024-07-01/2024-09-30"},
        ]
        result = _pick_best_fact(candidates)
        assert result["value"] == 75  # last in list

    def test_instant_period(self):
        candidates = [
            {"value": 100, "period": "2024-12-31"},
        ]
        result = _pick_best_fact(candidates)
        assert result["value"] == 100

    def test_no_period(self):
        candidates = [{"value": 42}]
        result = _pick_best_fact(candidates)
        assert result["value"] == 42

    def test_malformed_period_ignored(self):
        candidates = [
            {"value": 100, "period": "bad/date"},
            {"value": 200, "period": "2024-01-01/2024-12-31"},
        ]
        result = _pick_best_fact(candidates)
        assert result["value"] == 200


# ---------------------------------------------------------------------------
# _same_order_of_magnitude
# ---------------------------------------------------------------------------

class TestSameOrderOfMagnitude:
    def test_same_value(self):
        assert _same_order_of_magnitude(100.0, 100.0) is True

    def test_within_range(self):
        assert _same_order_of_magnitude(50.0, 100.0) is True  # ratio 0.5
        assert _same_order_of_magnitude(500.0, 100.0) is True  # ratio 5.0

    def test_boundary_low(self):
        assert _same_order_of_magnitude(10.0, 100.0) is True   # ratio 0.1
        assert _same_order_of_magnitude(9.9, 100.0) is False   # ratio 0.099

    def test_boundary_high(self):
        assert _same_order_of_magnitude(1000.0, 100.0) is True  # ratio 10.0
        assert _same_order_of_magnitude(1001.0, 100.0) is False  # ratio 10.01

    def test_zero_a(self):
        assert _same_order_of_magnitude(0.0, 100.0) is False

    def test_zero_b(self):
        assert _same_order_of_magnitude(100.0, 0.0) is False

    def test_both_zero(self):
        assert _same_order_of_magnitude(0.0, 0.0) is False

    def test_negative_values(self):
        assert _same_order_of_magnitude(-50.0, 100.0) is True
        assert _same_order_of_magnitude(-50.0, -100.0) is True


# ---------------------------------------------------------------------------
# extract_field
# ---------------------------------------------------------------------------

class TestExtractField:
    def test_known_field_with_data(self, sample_filing_xbrl):
        result = extract_field(sample_filing_xbrl, "revenue")
        # Fixture has two annual revenue entries; _pick_best_fact returns pool[-1]
        # which is the 2023 entry (last in the candidates list).
        assert result.value is not None
        assert result.value in (394_328_000_000.0, 383_285_000_000.0)
        # Tag is at index 1 in xbrl_tags list (preferred is us-gaap:Revenues at index 0)
        # so confidence = max(0.6, 1.0 - 1*0.1) = 0.9
        assert result.confidence > 0.0
        assert result.xbrl_tag is not None
        assert result.error is None

    def test_known_field_no_data(self, empty_filing_xbrl):
        result = extract_field(empty_filing_xbrl, "revenue")
        assert result.value is None
        assert result.confidence == 0.0
        assert result.error is not None

    def test_unknown_field(self, sample_filing_xbrl):
        result = extract_field(sample_filing_xbrl, "nonexistent_field")
        assert result.value is None
        assert result.error == "Unknown field: nonexistent_field"

    def test_all_twelve_fields_present(self, sample_filing_xbrl):
        """Sample filing has data for most fields."""
        for field_name in ["revenue", "cogs", "gross_profit", "operating_income",
                           "net_income", "eps_basic", "eps_diluted",
                           "rd_expense", "sga_expense"]:
            result = extract_field(sample_filing_xbrl, field_name)
            assert result.value is not None, f"{field_name} should have a value"

    def test_confidence_decay_for_fallback_tags(self):
        """When the preferred tag is missing but a fallback tag matches, confidence decays."""
        filing = FilingXBRL(
            accession="test", cik="test", fiscal_year_end=None,
            facts={
                # Revenue: preferred tag is index 0, this is index 1
                "us-gaap:RevenueFromContractWithCustomerIncludingAssessedTax": [
                    {"value": 100.0, "unit": "USD", "period": "2024-01-01/2024-12-31"},
                ],
            },
        )
        result = extract_field(filing, "revenue")
        assert result.value == 100.0
        # Tag at index 2 in the list → confidence = max(0.6, 1.0 - 2*0.1) = 0.8
        assert result.confidence < 1.0

    def test_target_period_filter(self, sample_filing_xbrl):
        result = extract_field(
            sample_filing_xbrl, "revenue",
            target_period="2023-10-01/2024-09-28",
        )
        assert result.value == 394_328_000_000.0

    def test_strips_prefix_fallback(self):
        """If tag is stored without namespace prefix, still matches."""
        filing = FilingXBRL(
            accession="test", cik="test", fiscal_year_end=None,
            facts={
                "RevenueFromContractWithCustomerExcludingAssessedTax": [
                    {"value": 999.0, "unit": "USD", "period": "2024-01-01/2024-12-31"},
                ],
            },
        )
        result = extract_field(filing, "revenue")
        assert result.value == 999.0


# ---------------------------------------------------------------------------
# extract_all_fields
# ---------------------------------------------------------------------------

class TestExtractAllFields:
    def test_returns_all_fields(self, sample_filing_xbrl):
        results = extract_all_fields(sample_filing_xbrl)
        assert len(results) == len(FIELDS)
        for field_name in FIELDS:
            assert field_name in results
            assert isinstance(results[field_name], ExtractionResult)

    def test_empty_filing(self, empty_filing_xbrl):
        results = extract_all_fields(empty_filing_xbrl)
        for field_name, result in results.items():
            assert result.value is None


# ---------------------------------------------------------------------------
# compute_reward
# ---------------------------------------------------------------------------

class TestComputeReward:
    def test_exact_match(self):
        r = compute_reward(100_000_000.0, 100_000_000.0, action_index=0)
        assert r == 1.0

    def test_within_1_percent(self):
        r = compute_reward(100_500_000.0, 100_000_000.0, action_index=0)
        assert r == 0.8

    def test_correct_magnitude(self):
        r = compute_reward(95_000_000.0, 100_000_000.0, action_index=0)
        assert r == 0.3

    def test_null_extracted(self):
        r = compute_reward(None, 100_000_000.0, action_index=0)
        assert r == -0.5

    def test_type_mismatch(self):
        r = compute_reward("not a number", 100_000_000.0, action_index=0)
        assert r == -1.0

    def test_ground_truth_none(self):
        r = compute_reward(100.0, None, action_index=0)
        assert r == 0.0

    def test_ground_truth_zero(self):
        # When ground_truth is 0, uses absolute error
        r = compute_reward(0.0, 0.0, action_index=0)
        assert r == 1.0  # exact match

    def test_ground_truth_zero_with_extracted_nonzero(self):
        r = compute_reward(100.0, 0.0, action_index=0)
        # rel_error = abs(100 - 0) = 100, not within 1%, not same OOM (0 returns False)
        assert r == -0.5

    def test_way_off(self):
        r = compute_reward(1_000.0, 100_000_000.0, action_index=0)
        assert r == -0.5  # not same order of magnitude

    def test_llm_latency_penalty(self):
        r = compute_reward(100_000_000.0, 100_000_000.0, action_index=3, latency_seconds=6.0)
        assert r == 1.0 + (-0.2)  # exact match minus penalty

    def test_llm_no_penalty_fast(self):
        r = compute_reward(100_000_000.0, 100_000_000.0, action_index=3, latency_seconds=4.0)
        assert r == 1.0  # no penalty

    def test_non_llm_no_latency_penalty(self):
        r = compute_reward(100_000_000.0, 100_000_000.0, action_index=0, latency_seconds=10.0)
        assert r == 1.0  # only action 3 gets penalty

    def test_negative_values_exact(self):
        r = compute_reward(-50_000.0, -50_000.0, action_index=0)
        assert r == 1.0


# ---------------------------------------------------------------------------
# load_xbrl_from_company_facts
# ---------------------------------------------------------------------------

class TestLoadXbrlFromCompanyFacts:
    @pytest.fixture
    def company_facts_json(self):
        return {
            "facts": {
                "us-gaap": {
                    "RevenueFromContractWithCustomerExcludingAssessedTax": {
                        "units": {
                            "USD": [
                                {
                                    "val": 394_328_000_000,
                                    "form": "10-K",
                                    "fy": 2024,
                                    "fp": "FY",
                                    "accn": "0000320193-24-000123",
                                    "start": "2023-10-01",
                                    "end": "2024-09-28",
                                },
                                {
                                    "val": 383_285_000_000,
                                    "form": "10-K",
                                    "fy": 2023,
                                    "fp": "FY",
                                    "accn": "0000320193-23-000077",
                                    "start": "2022-10-01",
                                    "end": "2023-09-30",
                                },
                                {
                                    "val": 50_000_000,
                                    "form": "10-Q",
                                    "fy": 2024,
                                    "fp": "Q1",
                                    "accn": "0000320193-24-000050",
                                    "start": "2023-10-01",
                                    "end": "2023-12-31",
                                },
                            ]
                        }
                    },
                }
            }
        }

    def test_no_filter_all_10k(self, company_facts_json):
        filing = load_xbrl_from_company_facts(company_facts_json, cik="0000320193")
        tag = "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
        assert tag in filing.facts
        # Should include both 10-K entries but NOT the 10-Q
        assert len(filing.facts[tag]) == 2

    def test_accession_filter(self, company_facts_json):
        filing = load_xbrl_from_company_facts(
            company_facts_json, cik="0000320193",
            accession="0000320193-24-000123",
        )
        tag = "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
        assert len(filing.facts[tag]) == 1
        assert filing.facts[tag][0]["value"] == 394_328_000_000

    def test_fiscal_year_filter(self, company_facts_json):
        filing = load_xbrl_from_company_facts(
            company_facts_json, cik="0000320193",
            fiscal_year=2023,
        )
        tag = "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
        assert len(filing.facts[tag]) == 1
        assert filing.facts[tag][0]["value"] == 383_285_000_000

    def test_10q_excluded(self, company_facts_json):
        filing = load_xbrl_from_company_facts(company_facts_json, cik="0000320193")
        tag = "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
        values = [f["value"] for f in filing.facts[tag]]
        assert 50_000_000 not in values

    def test_accession_normalization(self, company_facts_json):
        # Accession without dashes should be normalized
        filing = load_xbrl_from_company_facts(
            company_facts_json, cik="0000320193",
            accession="000032019324000123",
        )
        tag = "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
        assert len(filing.facts[tag]) == 1

    def test_empty_facts(self):
        filing = load_xbrl_from_company_facts({"facts": {}}, cik="0000000000")
        assert filing.facts == {}

    def test_period_construction(self, company_facts_json):
        filing = load_xbrl_from_company_facts(
            company_facts_json, cik="0000320193",
            accession="0000320193-24-000123",
        )
        tag = "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
        fact = filing.facts[tag][0]
        assert fact["period"] == "2023-10-01/2024-09-28"


# ---------------------------------------------------------------------------
# load_xbrl_from_facts_file
# ---------------------------------------------------------------------------

class TestLoadXbrlFromFactsFile:
    def test_valid_file(self, tmp_path):
        import json
        data = {
            "facts": {
                "us-gaap": {
                    "NetIncomeLoss": {
                        "units": {
                            "USD": [
                                {"val": 93_736_000_000, "form": "10-K", "fy": 2024,
                                 "accn": "0000320193-24-000123",
                                 "start": "2023-10-01", "end": "2024-09-28"},
                            ]
                        }
                    }
                }
            }
        }
        path = tmp_path / "facts.json"
        path.write_text(json.dumps(data))
        filing = load_xbrl_from_facts_file(path, cik="0000320193")
        assert filing is not None
        assert "us-gaap:NetIncomeLoss" in filing.facts

    def test_invalid_file(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json")
        result = load_xbrl_from_facts_file(path, cik="test")
        assert result is None

    def test_nonexistent_file(self, tmp_path):
        path = tmp_path / "missing.json"
        result = load_xbrl_from_facts_file(path, cik="test")
        assert result is None
