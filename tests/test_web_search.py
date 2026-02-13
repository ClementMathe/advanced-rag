"""Tests for DuckDuckGoSearchTool web search wrapper."""

import sys
from unittest.mock import MagicMock, patch

# Register a fake ddgs module so @patch can resolve the target
# even when the real package is not installed.
if "ddgs" not in sys.modules:
    sys.modules["ddgs"] = MagicMock()

from src.web_search import DuckDuckGoSearchTool


def _mock_ddgs_instance(results):
    """Create a mock DDGS instance returning given results."""
    mock = MagicMock()
    mock.text.return_value = results
    return mock


class TestDuckDuckGoSearchTool:
    """Tests for DuckDuckGoSearchTool."""

    def test_init_default_max_results(self):
        """Default max_results should be 5."""
        tool = DuckDuckGoSearchTool()
        assert tool.max_results == 5

    def test_init_custom_max_results(self):
        """Should accept custom max_results."""
        tool = DuckDuckGoSearchTool(max_results=10)
        assert tool.max_results == 10

    @patch("ddgs.DDGS")
    def test_search_returns_formatted_docs(self, mock_ddgs_class):
        """Should return list of doc dicts with content and source."""
        mock_ddgs_class.return_value = _mock_ddgs_instance(
            [
                {
                    "title": "Result 1",
                    "body": "Body text 1",
                    "href": "https://example.com/1",
                },
                {
                    "title": "Result 2",
                    "body": "Body text 2",
                    "href": "https://example.com/2",
                },
            ]
        )

        tool = DuckDuckGoSearchTool(max_results=2)
        results = tool.search("test query")

        assert len(results) == 2
        assert results[0]["content"] == "Result 1\nBody text 1"
        assert results[0]["source"] == "https://example.com/1"
        assert results[1]["content"] == "Result 2\nBody text 2"
        assert results[1]["source"] == "https://example.com/2"

    @patch("ddgs.DDGS")
    def test_search_passes_max_results(self, mock_ddgs_class):
        """Should pass max_results to DDGS.text()."""
        mock = _mock_ddgs_instance([])
        mock_ddgs_class.return_value = mock

        tool = DuckDuckGoSearchTool(max_results=3)
        tool.search("test query")

        mock.text.assert_called_once_with("test query", max_results=3)

    @patch("ddgs.DDGS")
    def test_search_empty_results(self, mock_ddgs_class):
        """Should return empty list when no results found."""
        mock_ddgs_class.return_value = _mock_ddgs_instance([])

        tool = DuckDuckGoSearchTool()
        results = tool.search("obscure query")

        assert results == []

    @patch("ddgs.DDGS")
    def test_search_missing_href_uses_default(self, mock_ddgs_class):
        """Should use 'web_search' as default source when href missing."""
        mock_ddgs_class.return_value = _mock_ddgs_instance(
            [
                {"title": "No URL", "body": "Body text"},
            ]
        )

        tool = DuckDuckGoSearchTool()
        results = tool.search("test")

        assert results[0]["source"] == "web_search"

    @patch("ddgs.DDGS", side_effect=Exception("Network error"))
    def test_search_error_returns_empty_list(self, mock_ddgs_class):
        """Should return empty list on error (graceful degradation)."""
        tool = DuckDuckGoSearchTool()
        results = tool.search("test query")

        assert results == []

    @patch("ddgs.DDGS", side_effect=TimeoutError("Timed out"))
    def test_search_timeout_returns_empty_list(self, mock_ddgs_class):
        """Should handle timeout errors gracefully."""
        tool = DuckDuckGoSearchTool()
        results = tool.search("test query")

        assert results == []
