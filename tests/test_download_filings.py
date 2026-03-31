"""
Tests for download_filings.py -- SGML header parsing, 10-K verification, and helpers.
"""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from download_filings import (
    parse_sgml_header,
    is_10k_submission,
    EdgarSession,
    get_10k_accessions,
    EDGAR_RATE_DELAY,
)


# ---------------------------------------------------------------------------
# parse_sgml_header
# ---------------------------------------------------------------------------

class TestParseSgmlHeader:
    def test_valid_10k_header(self):
        content = b"""<SEC-DOCUMENT>
<SEC-HEADER>
CONFORMED SUBMISSION TYPE:    10-K
FORM TYPE:                    10-K
CONFORMED PERIOD OF REPORT:   20240928
FILED AS OF DATE:             20241101
COMPANY CONFORMED NAME:       APPLE INC
CIK:                          0000320193
</SEC-HEADER>
<DOCUMENT>
body content here
"""
        header = parse_sgml_header(content)
        assert header["CONFORMED SUBMISSION TYPE"] == "10-K"
        assert header["FORM TYPE"] == "10-K"
        assert header["CONFORMED PERIOD OF REPORT"] == "20240928"
        assert header["COMPANY CONFORMED NAME"] == "APPLE INC"
        assert header["CIK"] == "0000320193"

    def test_empty_content(self):
        header = parse_sgml_header(b"")
        assert header == {}

    def test_no_header_section(self):
        content = b"Just some random text without any SEC headers"
        header = parse_sgml_header(content)
        assert header == {}

    def test_stops_at_document_tag(self):
        content = b"""<SEC-HEADER>
CONFORMED SUBMISSION TYPE:    10-K
</SEC-HEADER>
<DOCUMENT>
FAKE KEY:    should not appear
"""
        header = parse_sgml_header(content)
        assert "CONFORMED SUBMISSION TYPE" in header
        assert "FAKE KEY" not in header

    def test_multiple_fields(self):
        content = b"""<SEC-HEADER>
CONFORMED SUBMISSION TYPE:    10-K/A
FORM TYPE:                    10-K/A
FILED AS OF DATE:             20241101
</SEC-HEADER>"""
        header = parse_sgml_header(content)
        assert len(header) == 3

    def test_handles_utf8_errors(self):
        content = b"<SEC-HEADER>\nFORM TYPE:    10-K\n</SEC-HEADER>\xff\xfe"
        header = parse_sgml_header(content)
        assert header.get("FORM TYPE") == "10-K"


# ---------------------------------------------------------------------------
# is_10k_submission
# ---------------------------------------------------------------------------

class TestIs10kSubmission:
    def _make_header(self, form_type):
        return f"""<SEC-HEADER>
CONFORMED SUBMISSION TYPE:    {form_type}
</SEC-HEADER>
<DOCUMENT>
""".encode()

    def test_10k(self):
        assert is_10k_submission(self._make_header("10-K")) is True

    def test_10k_amended(self):
        assert is_10k_submission(self._make_header("10-K/A")) is True

    def test_10ksb_rejected(self):
        assert is_10k_submission(self._make_header("10-KSB")) is False

    def test_def14a_rejected(self):
        assert is_10k_submission(self._make_header("DEF 14A")) is False

    def test_10q_rejected(self):
        assert is_10k_submission(self._make_header("10-Q")) is False

    def test_20f_rejected(self):
        assert is_10k_submission(self._make_header("20-F")) is False

    def test_no_header_returns_true(self):
        """If no header found, don't reject (e.g. unit tests, mock data)."""
        assert is_10k_submission(b"no header here") is True

    def test_empty_content(self):
        assert is_10k_submission(b"") is True

    def test_form_type_fallback(self):
        """Uses FORM TYPE if CONFORMED SUBMISSION TYPE is absent."""
        content = b"""<SEC-HEADER>
FORM TYPE:    10-K
</SEC-HEADER>"""
        assert is_10k_submission(content) is True


# ---------------------------------------------------------------------------
# EdgarSession
# ---------------------------------------------------------------------------

class TestEdgarSession:
    def test_construction(self):
        session = EdgarSession()
        assert session._last_request == 0.0
        assert "User-Agent" in session.session.headers

    def test_custom_user_agent(self):
        session = EdgarSession(user_agent="test-agent test@test.com")
        assert session.session.headers["User-Agent"] == "test-agent test@test.com"

    def test_rate_limiting(self):
        """Verify that rate limiting sleeps between requests."""
        session = EdgarSession()
        import time

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch.object(session.session, "get", return_value=mock_response):
            with patch("download_filings.time.sleep") as mock_sleep:
                with patch("download_filings.time.time") as mock_time:
                    # First request: no delay needed (last_request = 0)
                    mock_time.return_value = 1000.0
                    session.get("http://example.com")

                    # Second request: should sleep
                    mock_time.return_value = 1000.05  # only 0.05s elapsed
                    session.get("http://example.com")
                    # Should have slept for ~0.07s (EDGAR_RATE_DELAY - 0.05)
                    mock_sleep.assert_called()


# ---------------------------------------------------------------------------
# get_10k_accessions
# ---------------------------------------------------------------------------

class TestGet10kAccessions:
    def test_extracts_10k_only(self):
        submissions = {
            "filings": {
                "recent": {
                    "form": ["10-K", "10-Q", "10-K", "8-K"],
                    "accessionNumber": ["acc1", "acc2", "acc3", "acc4"],
                    "filingDate": ["2024-11-01", "2024-08-01", "2023-11-01", "2024-06-01"],
                    "reportDate": ["2024-09-28", "2024-06-30", "2023-09-30", "2024-05-15"],
                }
            }
        }
        results = get_10k_accessions(submissions)
        assert len(results) == 2
        assert results[0]["accession"] == "acc1"
        assert results[1]["accession"] == "acc3"

    def test_max_filings_limit(self):
        submissions = {
            "filings": {
                "recent": {
                    "form": ["10-K", "10-K", "10-K", "10-K"],
                    "accessionNumber": ["a1", "a2", "a3", "a4"],
                    "filingDate": ["d1", "d2", "d3", "d4"],
                    "reportDate": ["r1", "r2", "r3", "r4"],
                }
            }
        }
        results = get_10k_accessions(submissions, max_filings=2)
        assert len(results) == 2

    def test_empty_submissions(self):
        results = get_10k_accessions({})
        assert results == []

    def test_empty_recent(self):
        submissions = {"filings": {"recent": {"form": [], "accessionNumber": [], "filingDate": [], "reportDate": []}}}
        results = get_10k_accessions(submissions)
        assert results == []

    def test_result_structure(self):
        submissions = {
            "filings": {
                "recent": {
                    "form": ["10-K"],
                    "accessionNumber": ["0000320193-24-000123"],
                    "filingDate": ["2024-11-01"],
                    "reportDate": ["2024-09-28"],
                }
            }
        }
        results = get_10k_accessions(submissions)
        assert len(results) == 1
        assert results[0]["accession"] == "0000320193-24-000123"
        assert results[0]["filed_date"] == "2024-11-01"
        assert results[0]["report_date"] == "2024-09-28"

    def test_skips_empty_accession(self):
        submissions = {
            "filings": {
                "recent": {
                    "form": ["10-K", "10-K"],
                    "accessionNumber": ["", "acc2"],
                    "filingDate": ["d1", "d2"],
                    "reportDate": ["r1", "r2"],
                }
            }
        }
        results = get_10k_accessions(submissions)
        assert len(results) == 1
        assert results[0]["accession"] == "acc2"
