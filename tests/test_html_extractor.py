"""
Comprehensive tests for html_extractor.py
"""

import re
from pathlib import Path

import pytest

from html_extractor import (
    _normalize,
    _label_matches,
    _parse_numeric,
    _detect_scale_from_ixbrl,
    _detect_scale,
    _score_table,
    _pick_annual_column,
    extract_from_html,
    INCOME_STATEMENT_SIGNALS,
)


# ===================================================================
# _normalize()
# ===================================================================

class TestNormalize:
    def test_lowercase(self):
        assert _normalize("Net Income") == "net income"

    def test_collapse_whitespace(self):
        assert _normalize("net   \t  income") == "net income"

    def test_strip_punctuation(self):
        assert _normalize("net income (loss)") == "net income loss"

    def test_combined(self):
        assert _normalize("  Net   Income—Basic! ") == "net income basic"

    def test_empty(self):
        assert _normalize("") == ""

    def test_dollar_sign_stripped(self):
        assert _normalize("$1,234") == "1 234"

    def test_already_normal(self):
        assert _normalize("revenue") == "revenue"


# ===================================================================
# _label_matches()
# ===================================================================

class TestLabelMatches:
    def test_exact_match(self):
        """Exact match with a known html_label should yield 1.0."""
        score = _label_matches("revenue", "revenue")
        assert score == 1.0

    def test_exact_match_net_income(self):
        score = _label_matches("net income", "net_income")
        assert score == 1.0

    def test_label_substring_of_cell(self):
        """Label appears inside a longer cell → ~0.85 range."""
        # "net sales" is a label for revenue; cell has extra text
        score = _label_matches("Total net sales for the year", "revenue")
        assert 0.5 < score <= 0.85

    def test_short_label_in_cell(self):
        """Short label (<=12 chars) present in cell → 0.70."""
        # "basic" is a label for eps_basic and len("basic") <= 12
        score = _label_matches("Earnings per share—basic and diluted", "eps_basic")
        assert score >= 0.70

    def test_cell_substring_of_label(self):
        """Cell is substring of label (long enough cell) → partial score."""
        # "net income" (len normalized = 10, >= 8) is substring of
        # "net income attributable" label for net_income
        score = _label_matches("net income attributable", "net_income")
        # Should score well — "net income attributable" is itself a label
        assert score >= 0.5

    def test_no_match(self):
        """Completely unrelated text → 0.0."""
        score = _label_matches("total assets", "revenue")
        assert score == 0.0

    def test_case_insensitivity(self):
        """Matching should be case insensitive."""
        score = _label_matches("NET INCOME", "net_income")
        assert score == 1.0

    def test_unknown_field(self):
        """Unknown field_name → no labels → 0.0."""
        score = _label_matches("revenue", "nonexistent_field_xyz")
        assert score == 0.0


# ===================================================================
# _parse_numeric()
# ===================================================================

class TestParseNumeric:
    def test_dollar_with_commas(self):
        assert _parse_numeric("$1,234") == 1234.0

    def test_negative_parentheses(self):
        assert _parse_numeric("(1,234)") == -1234.0

    def test_decimal_with_commas(self):
        assert _parse_numeric("1,234.56") == 1234.56

    def test_dollar_decimal(self):
        assert _parse_numeric("$1,234.56") == 1234.56

    def test_empty_string(self):
        assert _parse_numeric("") is None

    def test_dash(self):
        assert _parse_numeric("—") is None

    def test_en_dash(self):
        assert _parse_numeric("–") is None

    def test_hyphen_dash(self):
        assert _parse_numeric("-") is None

    def test_na(self):
        assert _parse_numeric("N/A") is None

    def test_na_lower(self):
        assert _parse_numeric("n/a") is None

    def test_asterisk(self):
        assert _parse_numeric("*") is None

    def test_plain_number(self):
        assert _parse_numeric("42") == 42.0

    def test_whitespace_padding(self):
        assert _parse_numeric("  $1,234  ") == 1234.0

    def test_negative_dollar_parentheses(self):
        assert _parse_numeric("($5,678)") == -5678.0

    def test_zero(self):
        assert _parse_numeric("0") == 0.0


# ===================================================================
# _detect_scale_from_ixbrl()
# ===================================================================

class TestDetectScaleFromIxbrl:
    def test_scale_6_millions(self):
        html = """
        <html><body>
        <ix:nonFraction unitRef="usd" scale="6" decimals="-6" name="us-gaap:Revenue">394328</ix:nonFraction>
        <ix:nonFraction unitRef="usd" scale="6" decimals="-6" name="us-gaap:NetIncomeLoss">93736</ix:nonFraction>
        <ix:nonFraction unitRef="usd" scale="6" decimals="-6" name="us-gaap:CostOfRevenue">210352</ix:nonFraction>
        </body></html>
        """
        assert _detect_scale_from_ixbrl(html) == 1_000_000

    def test_scale_3_thousands(self):
        html = """
        <html><body>
        <ix:nonFraction unitRef="usd" scale="3" decimals="-3" name="us-gaap:Revenue">394328</ix:nonFraction>
        <ix:nonFraction unitRef="usd" scale="3" decimals="-3" name="us-gaap:NetIncomeLoss">93736</ix:nonFraction>
        </body></html>
        """
        assert _detect_scale_from_ixbrl(html) == 1_000

    def test_no_ix_tags(self):
        html = "<html><body><p>Just some text</p></body></html>"
        assert _detect_scale_from_ixbrl(html) is None

    def test_ignores_per_share_unit(self):
        """Tags with unitRef other than 'usd' should be ignored."""
        html = """
        <ix:nonFraction unitRef="usdPerShare" scale="0" name="us-gaap:EPS">6.11</ix:nonFraction>
        """
        assert _detect_scale_from_ixbrl(html) is None

    def test_scale_9_billions(self):
        html = """
        <ix:nonFraction unitRef="usd" scale="9" decimals="-9" name="us-gaap:Revenue">394</ix:nonFraction>
        <ix:nonFraction unitRef="usd" scale="9" decimals="-9" name="us-gaap:NetIncome">93</ix:nonFraction>
        """
        assert _detect_scale_from_ixbrl(html) == 1_000_000_000

    def test_dominant_scale(self):
        """If mixed scales, the most common one wins."""
        html = """
        <ix:nonFraction unitRef="usd" scale="6" name="a">1</ix:nonFraction>
        <ix:nonFraction unitRef="usd" scale="6" name="b">2</ix:nonFraction>
        <ix:nonFraction unitRef="usd" scale="6" name="c">3</ix:nonFraction>
        <ix:nonFraction unitRef="usd" scale="3" name="d">4</ix:nonFraction>
        """
        assert _detect_scale_from_ixbrl(html) == 1_000_000


# ===================================================================
# _detect_scale()
# ===================================================================

class TestDetectScale:
    def _make_soup(self, html: str):
        from bs4 import BeautifulSoup
        return BeautifulSoup(html, "lxml")

    def test_in_millions(self):
        html = "<html><body><p>(In millions)</p></body></html>"
        soup = self._make_soup(html)
        assert _detect_scale(soup) == 1_000_000

    def test_in_thousands(self):
        html = "<html><body><p>(In thousands)</p></body></html>"
        soup = self._make_soup(html)
        assert _detect_scale(soup) == 1_000

    def test_in_billions(self):
        html = "<html><body><p>(In billions)</p></body></html>"
        soup = self._make_soup(html)
        assert _detect_scale(soup) == 1_000_000_000

    def test_no_scale_indicator(self):
        html = "<html><body><p>Just text, no scale.</p></body></html>"
        soup = self._make_soup(html)
        assert _detect_scale(soup) == 1.0

    def test_ixbrl_takes_priority(self):
        """If raw_html has iXBRL scale tags, they take priority."""
        html = """
        <html><body>
        <p>(In thousands)</p>
        <ix:nonFraction unitRef="usd" scale="6" name="us-gaap:Revenue">394328</ix:nonFraction>
        <ix:nonFraction unitRef="usd" scale="6" name="us-gaap:NI">93736</ix:nonFraction>
        </body></html>
        """
        soup = self._make_soup(html)
        # iXBRL scale=6 (millions) overrides text "(In thousands)"
        assert _detect_scale(soup, raw_html=html) == 1_000_000

    def test_dollars_in_millions(self):
        html = "<html><body><p>(dollars in millions)</p></body></html>"
        soup = self._make_soup(html)
        assert _detect_scale(soup) == 1_000_000

    def test_near_income_statement_anchor(self):
        """Scale note near 'Consolidated Statements of Operations' header."""
        html = """
        <html><body>
        <h2>Consolidated Statements of Operations</h2>
        <p>(In millions, except per share amounts)</p>
        <table><tr><td>Revenue</td><td>$100</td></tr></table>
        </body></html>
        """
        soup = self._make_soup(html)
        assert _detect_scale(soup) == 1_000_000


# ===================================================================
# _score_table()
# ===================================================================

class TestScoreTable:
    def test_high_score_income_statement(self):
        """Table with many income statement signals → high score."""
        rows = [
            ["", "2024", "2023"],
            ["Revenue", "$100,000", "$90,000"],
            ["Cost of revenue", "$60,000", "$55,000"],
            ["Gross profit", "$40,000", "$35,000"],
            ["Operating income", "$25,000", "$20,000"],
            ["Net income", "$15,000", "$12,000"],
            ["Earnings per share", "$2.50", "$2.00"],
            ["Net sales", "$100,000", "$90,000"],
        ]
        score = _score_table(rows)
        assert score > 0.5

    def test_zero_score_no_signals(self):
        """Table with no income statement signals → 0.0."""
        rows = [
            ["", "2024", "2023"],
            ["Total assets", "$500,000", "$450,000"],
            ["Total liabilities", "$300,000", "$280,000"],
            ["Stockholders equity", "$200,000", "$170,000"],
        ]
        score = _score_table(rows)
        assert score == 0.0

    def test_percentage_table_penalized(self):
        """Table with many % but no $ → heavy penalty."""
        rows = [
            ["", "2024", "2023"],
            ["Revenue", "100%", "100%"],
            ["Gross profit", "40%", "38%"],
            ["Operating income", "25%", "22%"],
            ["Net income", "15%", "13%"],
        ]
        score = _score_table(rows)
        # Even though it has income statement signals, the penalty reduces score
        dollar_rows = [
            ["", "2024", "2023"],
            ["Revenue", "$100,000", "$90,000"],
            ["Gross profit", "$40,000", "$35,000"],
            ["Operating income", "$25,000", "$20,000"],
            ["Net income", "$15,000", "$12,000"],
        ]
        dollar_score = _score_table(dollar_rows)
        assert score < dollar_score

    def test_empty_table(self):
        score = _score_table([])
        assert score == 0.0


# ===================================================================
# _pick_annual_column()
# ===================================================================

class TestPickAnnualColumn:
    def test_picks_most_recent_year(self):
        rows = [
            ["", "2022", "2023", "2024"],
            ["Revenue", "$80,000", "$90,000", "$100,000"],
        ]
        col, width = _pick_annual_column(rows)
        assert col == 3  # 2024 is most recent, at index 3

    def test_two_years(self):
        rows = [
            ["", "2023", "2024"],
            ["Revenue", "$90,000", "$100,000"],
        ]
        col, width = _pick_annual_column(rows)
        assert col == 2  # 2024 at index 2

    def test_no_years_defaults_to_col_1(self):
        rows = [
            ["Label", "Value A", "Value B"],
            ["Revenue", "$100,000", "$90,000"],
        ]
        col, width = _pick_annual_column(rows)
        assert col == 1  # default

    def test_year_in_header_text(self):
        rows = [
            ["", "Fiscal Year 2023", "Fiscal Year 2024"],
            ["Revenue", "$90,000", "$100,000"],
        ]
        col, width = _pick_annual_column(rows)
        assert col == 2  # 2024

    def test_header_width_returned(self):
        rows = [
            ["", "2023", "2024"],
            ["Revenue", "$90,000", "$100,000"],
        ]
        col, width = _pick_annual_column(rows)
        assert width == 3  # 3 cells in the header row with the year

    def test_empty_rows(self):
        col, width = _pick_annual_column([])
        assert col == 1
        assert width == 1


# ===================================================================
# extract_from_html() — integration tests
# ===================================================================

class TestExtractFromHtml:
    def test_extract_revenue(self, sample_html_file):
        """Revenue should be extracted and scaled to millions."""
        value, conf = extract_from_html(sample_html_file, "revenue")
        assert value is not None
        # Net sales = $394,328 in millions → 394,328,000,000
        assert value == pytest.approx(394_328 * 1_000_000, rel=0.01)
        assert conf > 0.0

    def test_extract_net_income(self, sample_html_file):
        """Net income should be found in the table."""
        value, conf = extract_from_html(sample_html_file, "net_income")
        assert value is not None
        # Net income = $93,736 in millions → 93,736,000,000
        assert value == pytest.approx(93_736 * 1_000_000, rel=0.01)
        assert conf > 0.0

    def test_extract_eps_basic_no_scale(self, sample_html_file):
        """EPS (USD/shares) should NOT have the millions scale applied."""
        value, conf = extract_from_html(sample_html_file, "eps_basic")
        assert value is not None
        # EPS basic = $6.11, no scale
        assert value == pytest.approx(6.11, rel=0.01)
        assert conf > 0.0

    def test_nonexistent_file(self, tmp_path):
        """Non-existent file → (None, 0.0)."""
        fake_path = tmp_path / "does_not_exist.htm"
        value, conf = extract_from_html(fake_path, "revenue")
        assert value is None
        assert conf == 0.0

    def test_file_with_no_tables(self, tmp_path):
        """HTML file with no <table> elements → (None, 0.0)."""
        html_path = tmp_path / "no_tables.htm"
        html_path.write_text("<html><body><p>No tables here.</p></body></html>")
        value, conf = extract_from_html(html_path, "revenue")
        assert value is None
        assert conf == 0.0

    def test_extract_operating_income(self, sample_html_file):
        """Operating income extraction."""
        value, conf = extract_from_html(sample_html_file, "operating_income")
        assert value is not None
        assert value == pytest.approx(123_216 * 1_000_000, rel=0.01)

    def test_extract_gross_profit(self, sample_html_file):
        """Gross profit extraction."""
        value, conf = extract_from_html(sample_html_file, "gross_profit")
        assert value is not None
        assert value == pytest.approx(183_976 * 1_000_000, rel=0.01)

    def test_extract_eps_diluted_no_scale(self, sample_html_file):
        """EPS diluted should not have scale applied."""
        value, conf = extract_from_html(sample_html_file, "eps_diluted")
        assert value is not None
        assert value == pytest.approx(6.08, rel=0.01)

    def test_file_with_empty_table(self, tmp_path):
        """HTML with table that has fewer than 3 rows → (None, 0.0)."""
        html_content = """
        <html><body>
        <table>
            <tr><td>Only one row</td></tr>
        </table>
        </body></html>
        """
        html_path = tmp_path / "small_table.htm"
        html_path.write_text(html_content)
        value, conf = extract_from_html(html_path, "revenue")
        assert value is None
        assert conf == 0.0

    def test_file_with_unrelated_table(self, tmp_path):
        """HTML with a balance sheet table (no income stmt signals) → (None, 0.0)."""
        html_content = """
        <html><body>
        <table>
            <tr><th></th><th>2024</th><th>2023</th></tr>
            <tr><td>Total assets</td><td>$500,000</td><td>$450,000</td></tr>
            <tr><td>Total liabilities</td><td>$300,000</td><td>$280,000</td></tr>
            <tr><td>Equity</td><td>$200,000</td><td>$170,000</td></tr>
        </table>
        </body></html>
        """
        html_path = tmp_path / "balance_sheet.htm"
        html_path.write_text(html_content)
        value, conf = extract_from_html(html_path, "revenue")
        assert value is None
        assert conf == 0.0
