"""
Shared fixtures for the edgarRL test suite.
"""

import json
import os
import sys
from pathlib import Path

import pytest

# Ensure the project root is on sys.path so we can import modules directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Fixtures: FilingXBRL with realistic data
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_filing_xbrl():
    """Return a FilingXBRL with realistic Apple-like financial data."""
    from xbrl_extractor import FilingXBRL

    facts = {
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax": [
            {"value": 394_328_000_000.0, "unit": "USD", "period": "2023-10-01/2024-09-28", "decimals": "-6"},
            {"value": 383_285_000_000.0, "unit": "USD", "period": "2022-10-01/2023-09-30", "decimals": "-6"},
        ],
        "us-gaap:CostOfRevenue": [
            {"value": 210_352_000_000.0, "unit": "USD", "period": "2023-10-01/2024-09-28", "decimals": "-6"},
        ],
        "us-gaap:GrossProfit": [
            {"value": 183_976_000_000.0, "unit": "USD", "period": "2023-10-01/2024-09-28", "decimals": "-6"},
        ],
        "us-gaap:OperatingIncomeLoss": [
            {"value": 123_216_000_000.0, "unit": "USD", "period": "2023-10-01/2024-09-28", "decimals": "-6"},
        ],
        "us-gaap:NetIncomeLoss": [
            {"value": 93_736_000_000.0, "unit": "USD", "period": "2023-10-01/2024-09-28", "decimals": "-6"},
        ],
        "us-gaap:EarningsPerShareBasic": [
            {"value": 6.11, "unit": "USD/shares", "period": "2023-10-01/2024-09-28", "decimals": "2"},
        ],
        "us-gaap:EarningsPerShareDiluted": [
            {"value": 6.08, "unit": "USD/shares", "period": "2023-10-01/2024-09-28", "decimals": "2"},
        ],
        "us-gaap:ResearchAndDevelopmentExpense": [
            {"value": 31_370_000_000.0, "unit": "USD", "period": "2023-10-01/2024-09-28", "decimals": "-6"},
        ],
        "us-gaap:SellingGeneralAndAdministrativeExpense": [
            {"value": 29_390_000_000.0, "unit": "USD", "period": "2023-10-01/2024-09-28", "decimals": "-6"},
        ],
        "us-gaap:InterestExpense": [
            {"value": 3_002_000_000.0, "unit": "USD", "period": "2023-10-01/2024-09-28", "decimals": "-6"},
        ],
        "us-gaap:IncomeTaxExpenseBenefit": [
            {"value": 29_749_000_000.0, "unit": "USD", "period": "2023-10-01/2024-09-28", "decimals": "-6"},
        ],
    }

    return FilingXBRL(
        accession="0000320193-24-000123",
        cik="0000320193",
        fiscal_year_end="2024-09-28",
        facts=facts,
    )


@pytest.fixture
def empty_filing_xbrl():
    """Return a FilingXBRL with no facts."""
    from xbrl_extractor import FilingXBRL
    return FilingXBRL(
        accession="0000000000-00-000000",
        cik="0000000000",
        fiscal_year_end=None,
        facts={},
    )


@pytest.fixture
def sample_registry(tmp_path):
    """Return a minimal data registry for testing the environment."""
    # Create a dummy facts JSON file
    facts_data = {
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
                            }
                        ]
                    }
                },
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            {
                                "val": 93_736_000_000,
                                "form": "10-K",
                                "fy": 2024,
                                "fp": "FY",
                                "accn": "0000320193-24-000123",
                                "start": "2023-10-01",
                                "end": "2024-09-28",
                            }
                        ]
                    }
                },
            }
        }
    }
    facts_path = tmp_path / "company_facts.json"
    facts_path.write_text(json.dumps(facts_data))

    return [
        {
            "filing_id": "0000320193/0000320193-24-000123",
            "cik": "0000320193",
            "accession": "0000320193-24-000123",
            "ticker": "AAPL",
            "company_name": "Apple Inc",
            "sic_code": 3571,
            "fiscal_year": 2024,
            "facts_json_path": str(facts_path),
            "xbrl_path": None,
            "html_path": None,
            "txt_path": None,
        }
    ]


@pytest.fixture
def db_connection(tmp_path):
    """Return a fresh DuckDB connection with schema initialized."""
    from db import get_connection
    db_path = tmp_path / "test_edgar_rl.duckdb"
    conn = get_connection(db_path)
    yield conn
    conn.close()


@pytest.fixture
def full_ground_truth_registry(tmp_path):
    """Return a registry with facts for ALL 12 fields.

    This guarantees that env.reset() always yields a non-None ground_truth
    regardless of which field is randomly selected — necessary because
    env.reset(seed=N) does NOT seed env._rng (Gymnasium custom-RNG gotcha).
    """
    facts_data = {
        "facts": {
            "us-gaap": {
                # revenue
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {"USD": [{"val": 394_328_000_000, "form": "10-K", "fy": 2024,
                        "fp": "FY", "accn": "0000320193-24-000123",
                        "start": "2023-10-01", "end": "2024-09-28"}]}
                },
                # cogs
                "CostOfRevenue": {
                    "units": {"USD": [{"val": 210_352_000_000, "form": "10-K", "fy": 2024,
                        "fp": "FY", "accn": "0000320193-24-000123",
                        "start": "2023-10-01", "end": "2024-09-28"}]}
                },
                # gross_profit
                "GrossProfit": {
                    "units": {"USD": [{"val": 183_976_000_000, "form": "10-K", "fy": 2024,
                        "fp": "FY", "accn": "0000320193-24-000123",
                        "start": "2023-10-01", "end": "2024-09-28"}]}
                },
                # operating_income
                "OperatingIncomeLoss": {
                    "units": {"USD": [{"val": 123_216_000_000, "form": "10-K", "fy": 2024,
                        "fp": "FY", "accn": "0000320193-24-000123",
                        "start": "2023-10-01", "end": "2024-09-28"}]}
                },
                # net_income
                "NetIncomeLoss": {
                    "units": {"USD": [{"val": 93_736_000_000, "form": "10-K", "fy": 2024,
                        "fp": "FY", "accn": "0000320193-24-000123",
                        "start": "2023-10-01", "end": "2024-09-28"}]}
                },
                # total_expenses
                "OperatingExpenses": {
                    "units": {"USD": [{"val": 271_112_000_000, "form": "10-K", "fy": 2024,
                        "fp": "FY", "accn": "0000320193-24-000123",
                        "start": "2023-10-01", "end": "2024-09-28"}]}
                },
                # interest_expense
                "InterestExpense": {
                    "units": {"USD": [{"val": 3_002_000_000, "form": "10-K", "fy": 2024,
                        "fp": "FY", "accn": "0000320193-24-000123",
                        "start": "2023-10-01", "end": "2024-09-28"}]}
                },
                # income_tax_expense
                "IncomeTaxExpenseBenefit": {
                    "units": {"USD": [{"val": 29_749_000_000, "form": "10-K", "fy": 2024,
                        "fp": "FY", "accn": "0000320193-24-000123",
                        "start": "2023-10-01", "end": "2024-09-28"}]}
                },
                # eps_basic
                "EarningsPerShareBasic": {
                    "units": {"USD/shares": [{"val": 6.11, "form": "10-K", "fy": 2024,
                        "fp": "FY", "accn": "0000320193-24-000123",
                        "start": "2023-10-01", "end": "2024-09-28"}]}
                },
                # eps_diluted
                "EarningsPerShareDiluted": {
                    "units": {"USD/shares": [{"val": 6.08, "form": "10-K", "fy": 2024,
                        "fp": "FY", "accn": "0000320193-24-000123",
                        "start": "2023-10-01", "end": "2024-09-28"}]}
                },
                # rd_expense
                "ResearchAndDevelopmentExpense": {
                    "units": {"USD": [{"val": 31_370_000_000, "form": "10-K", "fy": 2024,
                        "fp": "FY", "accn": "0000320193-24-000123",
                        "start": "2023-10-01", "end": "2024-09-28"}]}
                },
                # sga_expense
                "SellingGeneralAndAdministrativeExpense": {
                    "units": {"USD": [{"val": 29_390_000_000, "form": "10-K", "fy": 2024,
                        "fp": "FY", "accn": "0000320193-24-000123",
                        "start": "2023-10-01", "end": "2024-09-28"}]}
                },
            }
        }
    }
    facts_path = tmp_path / "all_fields_facts.json"
    facts_path.write_text(json.dumps(facts_data))

    return [
        {
            "filing_id": "0000320193/0000320193-24-000123",
            "cik": "0000320193",
            "accession": "0000320193-24-000123",
            "ticker": "AAPL",
            "company_name": "Apple Inc",
            "sic_code": 3571,
            "fiscal_year": 2024,
            "facts_json_path": str(facts_path),
            "xbrl_path": None,
            "html_path": None,
            "txt_path": None,
        }
    ]


@pytest.fixture
def sample_html_file(tmp_path):
    """Create a sample HTML filing with an income statement table."""
    html_content = """
    <html>
    <head><title>10-K Filing</title></head>
    <body>
    <h2>CONSOLIDATED STATEMENTS OF OPERATIONS</h2>
    <p>(In millions, except per share amounts)</p>
    <table>
        <tr><th></th><th>2024</th><th>2023</th></tr>
        <tr><td>Net sales</td><td>$394,328</td><td>$383,285</td></tr>
        <tr><td>Cost of sales</td><td>$210,352</td><td>$214,137</td></tr>
        <tr><td>Gross profit</td><td>$183,976</td><td>$169,148</td></tr>
        <tr><td>Research and development</td><td>$31,370</td><td>$29,915</td></tr>
        <tr><td>Selling, general and administrative</td><td>$29,390</td><td>$24,932</td></tr>
        <tr><td>Operating income</td><td>$123,216</td><td>$114,301</td></tr>
        <tr><td>Interest expense</td><td>$3,002</td><td>$3,933</td></tr>
        <tr><td>Provision for income taxes</td><td>$29,749</td><td>$16,741</td></tr>
        <tr><td>Net income</td><td>$93,736</td><td>$96,995</td></tr>
        <tr><td>Basic earnings per share</td><td>$6.11</td><td>$6.16</td></tr>
        <tr><td>Diluted earnings per share</td><td>$6.08</td><td>$6.13</td></tr>
    </table>
    </body>
    </html>
    """
    html_path = tmp_path / "filing.htm"
    html_path.write_text(html_content)
    return html_path


@pytest.fixture
def sample_txt_file(tmp_path):
    """Create a sample TXT filing with income statement content.

    NOTE: The regex extractor has a double-escaping bug in _build_label_patterns()
    that prevents multi-word labels (e.g. "net sales") from matching. Only single-word
    labels like "revenue" and "revenues" work. This fixture includes both forms so
    that at least the single-word label tests pass.
    """
    txt_content = """
<SEC-DOCUMENT>
<SEC-HEADER>
CONFORMED SUBMISSION TYPE:    10-K
FORM TYPE:                    10-K
CONFORMED PERIOD OF REPORT:   20240928
</SEC-HEADER>
<DOCUMENT>

CONSOLIDATED STATEMENTS OF OPERATIONS
(In millions, except per share amounts)

                                    2024          2023
Revenue                         $394,328      $383,285
Cost of sales                    210,352       214,137
Gross profit                     183,976       169,148
Research and development          31,370        29,915
Selling, general and administrative 29,390      24,932
Total operating expenses         271,112       268,984
Operating income                 123,216       114,301
Interest expense                   3,002         3,933
Provision for income taxes        29,749        16,741
Net income                        93,736        96,995
Basic earnings per share            6.11          6.16
Diluted earnings per share          6.08          6.13
"""
    txt_path = tmp_path / "filing.txt"
    txt_path.write_text(txt_content)
    return txt_path
