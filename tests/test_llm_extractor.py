"""
Tests for llm_extractor.py -- Claude LLM fallback extractor (all API calls mocked).
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from types import ModuleType

import pytest

from llm_extractor import (
    _get_relevant_passage,
    _call_claude,
    extract_with_claude,
    extract_all_fields_with_claude,
    FIELDS,
    MAX_CONTEXT_CHARS,
)


# ---------------------------------------------------------------------------
# Helper: inject a fake anthropic module into sys.modules
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_anthropic():
    """Inject a fake 'anthropic' module into sys.modules and return the mock client."""
    fake_module = ModuleType("anthropic")
    mock_client = MagicMock()
    fake_module.Anthropic = MagicMock(return_value=mock_client)

    old = sys.modules.get("anthropic")
    sys.modules["anthropic"] = fake_module
    yield mock_client, fake_module
    if old is None:
        sys.modules.pop("anthropic", None)
    else:
        sys.modules["anthropic"] = old


# ---------------------------------------------------------------------------
# _get_relevant_passage
# ---------------------------------------------------------------------------

class TestGetRelevantPassage:
    def test_txt_with_anchor(self, tmp_path):
        txt = "Preamble\nCONSOLIDATED STATEMENTS OF OPERATIONS\nRevenue: $100M\nMore data here"
        path = tmp_path / "filing.txt"
        path.write_text(txt)
        passage = _get_relevant_passage(path)
        assert "Revenue" in passage
        assert len(passage) <= MAX_CONTEXT_CHARS

    def test_html_strips_tags(self, tmp_path):
        html = "<html><body><h1>CONSOLIDATED STATEMENTS OF OPERATIONS</h1><p>Revenue: $100M</p></body></html>"
        path = tmp_path / "filing.htm"
        path.write_text(html)
        passage = _get_relevant_passage(path)
        assert "<html>" not in passage
        assert "Revenue" in passage

    def test_no_anchor_returns_middle(self, tmp_path):
        txt = "A" * 1000 + "MIDDLE_CONTENT" + "B" * 1000
        path = tmp_path / "filing.txt"
        path.write_text(txt)
        passage = _get_relevant_passage(path)
        assert len(passage) > 0

    def test_nonexistent_file(self, tmp_path):
        path = tmp_path / "missing.txt"
        passage = _get_relevant_passage(path)
        assert passage == ""

    def test_max_chars_limit(self, tmp_path):
        txt = "CONSOLIDATED STATEMENTS OF OPERATIONS\n" + "x" * (MAX_CONTEXT_CHARS * 2)
        path = tmp_path / "filing.txt"
        path.write_text(txt)
        passage = _get_relevant_passage(path)
        assert len(passage) <= MAX_CONTEXT_CHARS + 200  # 200 chars before anchor

    def test_html_fallback_regex_strip(self, tmp_path):
        """If BeautifulSoup is somehow unavailable, falls back to regex strip."""
        html = "<html><body><p>CONSOLIDATED STATEMENTS OF OPERATIONS Revenue $100</p></body></html>"
        path = tmp_path / "filing.html"
        path.write_text(html)
        passage = _get_relevant_passage(path)
        assert "Revenue" in passage


# ---------------------------------------------------------------------------
# _call_claude (mocked)
# ---------------------------------------------------------------------------

class TestCallClaude:
    def test_successful_json_response(self, mock_anthropic):
        mock_client, _ = mock_anthropic
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"value": 100.0, "confidence": 0.9}')]
        mock_client.messages.create.return_value = mock_response

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            result = _call_claude("test prompt")
            assert result["value"] == 100.0
            assert result["confidence"] == 0.9

    def test_strips_markdown_fences(self, mock_anthropic):
        mock_client, _ = mock_anthropic
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='```json\n{"value": 42.0}\n```')]
        mock_client.messages.create.return_value = mock_response

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            result = _call_claude("test prompt")
            assert result["value"] == 42.0

    def test_missing_api_key(self, mock_anthropic):
        env = os.environ.copy()
        env.pop("ANTHROPIC_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
                _call_claude("test prompt")

    def test_invalid_json_response(self, mock_anthropic):
        mock_client, _ = mock_anthropic
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="not json at all")]
        mock_client.messages.create.return_value = mock_response

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with pytest.raises(json.JSONDecodeError):
                _call_claude("test prompt")

    def test_empty_response(self, mock_anthropic):
        mock_client, _ = mock_anthropic
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="")]
        mock_client.messages.create.return_value = mock_response

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with pytest.raises(json.JSONDecodeError):
                _call_claude("test prompt")


# ---------------------------------------------------------------------------
# extract_with_claude (mocked)
# ---------------------------------------------------------------------------

class TestExtractWithClaude:
    def _mock_call_claude(self, return_value):
        return patch("llm_extractor._call_claude", return_value=return_value)

    def test_successful_extraction(self, tmp_path):
        path = tmp_path / "filing.txt"
        path.write_text("CONSOLIDATED STATEMENTS OF OPERATIONS\nRevenue: $100M")

        with self._mock_call_claude({"value": 100_000_000, "confidence": 0.9}):
            value, conf = extract_with_claude(path, "revenue")
            assert value == 100_000_000.0
            assert conf == 0.85  # capped at 0.85

    def test_confidence_capped_at_085(self, tmp_path):
        path = tmp_path / "filing.txt"
        path.write_text("CONSOLIDATED STATEMENTS OF OPERATIONS\nRevenue: $100M")

        with self._mock_call_claude({"value": 100, "confidence": 0.99}):
            _, conf = extract_with_claude(path, "revenue")
            assert conf == 0.85

    def test_null_value(self, tmp_path):
        path = tmp_path / "filing.txt"
        path.write_text("CONSOLIDATED STATEMENTS OF OPERATIONS\nSome text")

        with self._mock_call_claude({"value": None, "confidence": 0.0}):
            value, conf = extract_with_claude(path, "revenue")
            assert value is None
            assert conf == 0.0

    def test_non_numeric_value(self, tmp_path):
        path = tmp_path / "filing.txt"
        path.write_text("CONSOLIDATED STATEMENTS OF OPERATIONS\nSome text")

        with self._mock_call_claude({"value": "not a number", "confidence": 0.5}):
            value, conf = extract_with_claude(path, "revenue")
            assert value is None
            assert conf == 0.0

    def test_json_decode_error(self, tmp_path):
        path = tmp_path / "filing.txt"
        path.write_text("CONSOLIDATED STATEMENTS OF OPERATIONS\nSome text")

        with patch("llm_extractor._call_claude", side_effect=json.JSONDecodeError("", "", 0)):
            value, conf = extract_with_claude(path, "revenue")
            assert value is None
            assert conf == 0.0

    def test_api_error(self, tmp_path):
        path = tmp_path / "filing.txt"
        path.write_text("CONSOLIDATED STATEMENTS OF OPERATIONS\nSome text")

        with patch("llm_extractor._call_claude", side_effect=Exception("API error")):
            value, conf = extract_with_claude(path, "revenue")
            assert value is None
            assert conf == 0.0

    def test_unknown_field(self, tmp_path):
        path = tmp_path / "filing.txt"
        path.write_text("Some text")
        value, conf = extract_with_claude(path, "nonexistent_field")
        assert value is None
        assert conf == 0.0

    def test_empty_passage(self, tmp_path):
        path = tmp_path / "filing.txt"
        path.write_text("")  # empty file
        # Should return None without calling Claude
        value, conf = extract_with_claude(path, "revenue")
        assert value is None
        assert conf == 0.0


# ---------------------------------------------------------------------------
# extract_all_fields_with_claude (mocked)
# ---------------------------------------------------------------------------

class TestExtractAllFieldsWithClaude:
    def test_empty_passage(self, tmp_path):
        path = tmp_path / "filing.txt"
        path.write_text("")
        results = extract_all_fields_with_claude(path)
        assert len(results) == len(FIELDS)
        for field_name, (val, conf) in results.items():
            assert val is None
            assert conf == 0.0

    def test_successful_batch(self, tmp_path, mock_anthropic):
        mock_client, _ = mock_anthropic
        path = tmp_path / "filing.txt"
        path.write_text("CONSOLIDATED STATEMENTS OF OPERATIONS\nRevenue: $100M\nNet income: $50M")

        batch_result = {
            field: {"value": 100.0, "confidence": 0.9}
            for field in FIELDS
        }
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps(batch_result))]
        mock_client.messages.create.return_value = mock_response

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            results = extract_all_fields_with_claude(path)
            assert len(results) == len(FIELDS)
            for field_name, (val, conf) in results.items():
                assert val == 100.0
                assert conf == 0.85  # capped

    def test_api_failure_returns_none(self, tmp_path, mock_anthropic):
        mock_client, fake_module = mock_anthropic
        path = tmp_path / "filing.txt"
        path.write_text("CONSOLIDATED STATEMENTS OF OPERATIONS\nRevenue: $100M")

        fake_module.Anthropic.side_effect = Exception("API down")

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            results = extract_all_fields_with_claude(path)
            for field_name, (val, conf) in results.items():
                assert val is None
                assert conf == 0.0
