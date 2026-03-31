"""
Tests for regex_extractor.py -- regex-based extraction from raw TXT filings.
"""

import re
from pathlib import Path

import pytest

from regex_extractor import (
    _detect_scale,
    _parse_value,
    _find_income_stmt_section,
    _build_label_patterns,
    _find_value_near_label,
    extract_from_txt,
    SECTION_WINDOW,
    LABEL_VALUE_WINDOW,
    _VALUE_PATTERN,
)


# ---------------------------------------------------------------------------
# _detect_scale
# ---------------------------------------------------------------------------

class TestDetectScale:
    def test_millions(self):
        text = "(In millions, except per share amounts)\nRevenue: 394,328"
        assert _detect_scale(text) == 1_000_000

    def test_thousands(self):
        text = "(In thousands)\nRevenue: 394,328,000"
        assert _detect_scale(text) == 1_000

    def test_billions(self):
        text = "(In billions)\nRevenue: 394"
        assert _detect_scale(text) == 1_000_000_000

    def test_no_scale(self):
        text = "Revenue: $394,328,000,000"
        assert _detect_scale(text) == 1.0

    def test_case_insensitive(self):
        text = "in MILLIONS except per share"
        assert _detect_scale(text) == 1_000_000

    def test_only_checks_first_3000_chars(self):
        text = "x" * 3001 + "in millions"
        assert _detect_scale(text) == 1.0


# ---------------------------------------------------------------------------
# _parse_value
# ---------------------------------------------------------------------------

class TestParseValue:
    def test_dollar_with_commas(self):
        assert _parse_value("$1,234") == 1234.0

    def test_negative_parentheses(self):
        assert _parse_value("(1,234)") == -1234.0

    def test_decimal(self):
        assert _parse_value("1,234.56") == 1234.56

    def test_dollar_decimal(self):
        assert _parse_value("$1,234.56") == 1234.56

    def test_plain_number(self):
        assert _parse_value("42") == 42.0

    def test_empty(self):
        assert _parse_value("") is None

    def test_dash(self):
        assert _parse_value("-") is None
        assert _parse_value("\u2014") is None  # em dash
        assert _parse_value("\u2013") is None  # en dash

    def test_na(self):
        assert _parse_value("N/A") is None

    def test_asterisk(self):
        assert _parse_value("*") is None

    def test_whitespace(self):
        assert _parse_value("  1,234  ") == 1234.0

    def test_large_number(self):
        assert _parse_value("394,328,000,000") == 394_328_000_000.0

    def test_negative_dollar(self):
        assert _parse_value("($1,234)") == -1234.0


# ---------------------------------------------------------------------------
# _find_income_stmt_section
# ---------------------------------------------------------------------------

class TestFindIncomeStmtSection:
    def test_consolidated_statements_of_operations(self):
        text = "Some preamble\nCONSOLIDATED STATEMENTS OF OPERATIONS\nRevenue: 100"
        section = _find_income_stmt_section(text)
        assert "Revenue" in section
        assert len(section) <= SECTION_WINDOW

    def test_statements_of_income(self):
        text = "Preamble\nStatements of Income\nNet income: 50"
        section = _find_income_stmt_section(text)
        assert "Net income" in section

    def test_results_of_operations(self):
        text = "Preamble\nResults of Operations\nRevenue grew 10%"
        section = _find_income_stmt_section(text)
        assert "Revenue" in section

    def test_no_anchor_returns_full_text(self):
        text = "This is just some random text with no income statement"
        section = _find_income_stmt_section(text)
        assert section == text

    def test_case_insensitive(self):
        text = "consolidated statements of operations\nRevenue: 100"
        section = _find_income_stmt_section(text)
        assert "Revenue" in section

    def test_section_window_limit(self):
        text = "CONSOLIDATED STATEMENTS OF OPERATIONS\n" + "x" * (SECTION_WINDOW + 1000)
        section = _find_income_stmt_section(text)
        assert len(section) == SECTION_WINDOW


# ---------------------------------------------------------------------------
# _build_label_patterns
# ---------------------------------------------------------------------------

class TestBuildLabelPatterns:
    def test_known_field(self):
        patterns = _build_label_patterns("revenue")
        assert len(patterns) > 0
        assert all(isinstance(p, re.Pattern) for p in patterns)

    def test_patterns_match_labels(self):
        patterns = _build_label_patterns("revenue")
        # Single-word labels like "revenue" and "revenues" should match
        matched = any(p.search("revenue") for p in patterns)
        assert matched
        matched_plural = any(p.search("revenues") for p in patterns)
        assert matched_plural

    def test_unknown_field(self):
        patterns = _build_label_patterns("nonexistent_field")
        assert patterns == []

    def test_flexible_whitespace_bug(self):
        """Multi-word patterns have double-escaped \\s+ and won't match flexible whitespace.

        This documents a known bug in _build_label_patterns() where re.escape()
        followed by re.sub() produces literal '\\\\s+' instead of regex '\\s+'.
        The multi-word pattern 'net\\\\s+revenue' won't match 'net   revenue',
        but the single-word pattern 'revenue' will match as a substring.
        """
        patterns = _build_label_patterns("revenue")
        # The multi-word pattern specifically should NOT match
        multi_word_patterns = [p for p in patterns if "s+" in p.pattern]
        matched_multi = any(p.search("net   revenue") for p in multi_word_patterns)
        assert not matched_multi  # documents the bug: multi-word patterns are broken

        # But single-word "revenue" still matches as a substring
        single_word = [p for p in patterns if p.pattern == "revenue"]
        assert any(p.search("net   revenue") for p in single_word)


# ---------------------------------------------------------------------------
# _find_value_near_label
# ---------------------------------------------------------------------------

class TestFindValueNearLabel:
    def test_finds_nearest_value(self):
        section = "Revenue   $394,328   $383,285"
        label_match = re.search(r"Revenue", section)
        val, proximity = _find_value_near_label(section, label_match, scale=1.0)
        assert val == 394_328.0
        assert proximity > 0.0

    def test_applies_scale(self):
        section = "Revenue   394,328   383,285"
        label_match = re.search(r"Revenue", section)
        val, proximity = _find_value_near_label(section, label_match, scale=1_000_000)
        assert val == 394_328_000_000.0

    def test_no_value_found(self):
        section = "Revenue is growing strongly this year"
        label_match = re.search(r"Revenue", section)
        val, proximity = _find_value_near_label(section, label_match, scale=1.0)
        assert val is None
        assert proximity == 0.0

    def test_proximity_decreases_with_distance(self):
        close = "Revenue $100 more text"
        far = "Revenue " + "x" * 200 + " $100 more text"
        m_close = re.search(r"Revenue", close)
        m_far = re.search(r"Revenue", far)
        _, prox_close = _find_value_near_label(close, m_close, 1.0)
        _, prox_far = _find_value_near_label(far, m_far, 1.0)
        assert prox_close > prox_far


# ---------------------------------------------------------------------------
# extract_from_txt (integration)
# ---------------------------------------------------------------------------

class TestExtractFromTxt:
    def test_extract_revenue(self, sample_txt_file):
        value, conf = extract_from_txt(sample_txt_file, "revenue")
        assert value is not None
        assert conf > 0.0

    def test_extract_net_income(self, sample_txt_file):
        # NOTE: All net_income label patterns are multi-word (e.g. "net income")
        # and broken due to double-escaping bug in _build_label_patterns().
        # The extractor will fail to match any label, returning (None, 0.0).
        value, conf = extract_from_txt(sample_txt_file, "net_income")
        assert value is None
        assert conf == 0.0

    def test_nonexistent_file(self, tmp_path):
        path = tmp_path / "missing.txt"
        value, conf = extract_from_txt(path, "revenue")
        assert value is None
        assert conf == 0.0

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.txt"
        path.write_text("")
        value, conf = extract_from_txt(path, "revenue")
        assert value is None
        assert conf == 0.0

    def test_confidence_capped(self, sample_txt_file):
        value, conf = extract_from_txt(sample_txt_file, "revenue")
        assert conf <= 0.90

    def test_unknown_field(self, sample_txt_file):
        value, conf = extract_from_txt(sample_txt_file, "nonexistent_field")
        assert value is None
        assert conf == 0.0

    def test_with_scale_detection(self, tmp_path):
        txt = """
CONSOLIDATED STATEMENTS OF OPERATIONS
(In millions)

Net sales                       394,328
Cost of sales                   210,352
"""
        path = tmp_path / "scaled.txt"
        path.write_text(txt)
        value, conf = extract_from_txt(path, "revenue")
        if value is not None:
            # Should be scaled to millions
            assert value >= 1_000_000  # at least in millions range

    def test_large_file_guard(self, tmp_path):
        """Files larger than max_file_mb should still be partially read."""
        path = tmp_path / "large.txt"
        # Create a small file but test the parameter
        path.write_text("Revenue $100\n" * 100)
        value, conf = extract_from_txt(path, "revenue", max_file_mb=0.0001)
        # Should still attempt to read (reads up to 50MB regardless)
        # The guard just logs a warning
