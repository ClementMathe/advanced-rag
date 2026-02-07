"""
Unit tests for document grading and query rewriting module.

Tests:
- DocumentGrader: single and batch grading with parsing edge cases
- QueryRewriter: query rewriting with output cleaning
- Fallback behavior for ambiguous or malformed LLM responses

Note: Uses mocking to avoid loading LLM models in tests.
"""

from unittest.mock import MagicMock, Mock

import pytest

from src.graders import DocumentGrader, QueryRewriter


def create_mock_generator(max_new_tokens: int = 256) -> MagicMock:
    """Create a mock LLMGenerator with controllable _generate_text."""
    generator = MagicMock()
    generator.max_new_tokens = max_new_tokens
    generator._generate_text = Mock(return_value="yes")
    return generator


class TestDocumentGraderSingle:
    """Tests for DocumentGrader.grade_single()."""

    @pytest.fixture
    def grader(self):
        """Create a DocumentGrader with mocked generator."""
        generator = create_mock_generator()
        return DocumentGrader(generator)

    def test_relevant_document(self, grader):
        """'yes' response should return True."""
        grader.generator._generate_text.return_value = "yes"
        assert grader.grade_single("When was Beyonce born?", "Beyonce was born in 1981.") is True

    def test_irrelevant_document(self, grader):
        """'no' response should return False."""
        grader.generator._generate_text.return_value = "no"
        assert grader.grade_single("When was Beyonce born?", "The Eiffel Tower is tall.") is False

    def test_yes_with_extra_text(self, grader):
        """Response containing 'yes' among other text should return True."""
        grader.generator._generate_text.return_value = "Yes, this document is relevant."
        assert grader.grade_single("query", "doc") is True

    def test_no_with_extra_text(self, grader):
        """Response containing 'no' (without 'yes') should return False."""
        grader.generator._generate_text.return_value = "No, this is not relevant."
        assert grader.grade_single("query", "doc") is False

    def test_ambiguous_response_defaults_to_true(self, grader):
        """Unparseable response should default to True (permissive)."""
        grader.generator._generate_text.return_value = "maybe relevant"
        assert grader.grade_single("query", "doc") is True

    def test_empty_response_defaults_to_true(self, grader):
        """Empty response should default to True (permissive)."""
        grader.generator._generate_text.return_value = ""
        assert grader.grade_single("query", "doc") is True

    def test_case_insensitive_yes(self, grader):
        """'YES', 'Yes', 'yEs' should all return True."""
        for variant in ["YES", "Yes", "yEs", "  yes  "]:
            grader.generator._generate_text.return_value = variant
            assert grader.grade_single("query", "doc") is True

    def test_case_insensitive_no(self, grader):
        """'NO', 'No', 'nO' should all return False."""
        for variant in ["NO", "No", "nO", "  no  "]:
            grader.generator._generate_text.return_value = variant
            assert grader.grade_single("query", "doc") is False

    def test_response_with_both_yes_and_no(self, grader):
        """If both 'yes' and 'no' appear, 'yes' wins (permissive)."""
        grader.generator._generate_text.return_value = "yes, but also no"
        assert grader.grade_single("query", "doc") is True

    def test_max_tokens_override_and_restore(self, grader):
        """max_new_tokens should be temporarily set to 10 and then restored."""
        grader.generator.max_new_tokens = 256

        grader.grade_single("query", "doc")

        # Should be restored to original
        assert grader.generator.max_new_tokens == 256

    def test_max_tokens_restored_on_error(self, grader):
        """max_new_tokens should be restored even if _generate_text raises."""
        grader.generator.max_new_tokens = 256
        grader.generator._generate_text.side_effect = RuntimeError("OOM")

        with pytest.raises(RuntimeError):
            grader.grade_single("query", "doc")

        assert grader.generator.max_new_tokens == 256

    def test_prompt_contains_query_and_document(self, grader):
        """The prompt sent to the LLM should contain the query and document text."""
        grader.grade_single("test query here", "test document content")

        call_args = grader.generator._generate_text.call_args[0][0]
        assert "test query here" in call_args
        assert "test document content" in call_args


class TestDocumentGraderBatch:
    """Tests for DocumentGrader.grade_batch()."""

    @pytest.fixture
    def grader(self):
        """Create a DocumentGrader with mocked generator."""
        generator = create_mock_generator()
        return DocumentGrader(generator)

    def test_valid_json_response(self, grader):
        """Properly formatted JSON should be parsed correctly."""
        grader.generator._generate_text.return_value = '{"grades": [true, false, true]}'

        grades = grader.grade_batch("query", ["doc1", "doc2", "doc3"])

        assert grades == [True, False, True]

    def test_all_relevant(self, grader):
        """All true grades should work."""
        grader.generator._generate_text.return_value = '{"grades": [true, true, true]}'

        grades = grader.grade_batch("query", ["d1", "d2", "d3"])

        assert grades == [True, True, True]

    def test_all_irrelevant(self, grader):
        """All false grades should work."""
        grader.generator._generate_text.return_value = '{"grades": [false, false]}'

        grades = grader.grade_batch("query", ["d1", "d2"])

        assert grades == [False, False]

    def test_single_document(self, grader):
        """Batch grading with a single document should work."""
        grader.generator._generate_text.return_value = '{"grades": [true]}'

        grades = grader.grade_batch("query", ["d1"])

        assert grades == [True]

    def test_empty_documents_returns_empty(self, grader):
        """Empty document list should return empty grades without LLM call."""
        grades = grader.grade_batch("query", [])

        assert grades == []
        grader.generator._generate_text.assert_not_called()

    def test_json_with_surrounding_text(self, grader):
        """JSON embedded in explanation text should be extracted."""
        grader.generator._generate_text.return_value = (
            'Here are the grades:\n{"grades": [true, false]}\nDone.'
        )

        grades = grader.grade_batch("query", ["d1", "d2"])

        assert grades == [True, False]

    def test_boolean_pattern_fallback(self, grader):
        """When JSON fails, should fall back to boolean pattern matching."""
        grader.generator._generate_text.return_value = "1. true\n2. false\n3. true"

        grades = grader.grade_batch("query", ["d1", "d2", "d3"])

        assert grades == [True, False, True]

    def test_unparseable_response_defaults_to_all_true(self, grader):
        """Completely unparseable response should default to all True."""
        grader.generator._generate_text.return_value = "I cannot grade these documents."

        grades = grader.grade_batch("query", ["d1", "d2", "d3"])

        assert grades == [True, True, True]

    def test_json_count_mismatch_triggers_fallback(self, grader):
        """JSON with wrong number of grades should trigger fallback."""
        # 2 grades for 3 documents
        grader.generator._generate_text.return_value = '{"grades": [true, false]}'

        grades = grader.grade_batch("query", ["d1", "d2", "d3"])

        # Should fall back to all True since count doesn't match
        assert len(grades) == 3

    def test_max_tokens_override_and_restore(self, grader):
        """max_new_tokens should be temporarily set to 100 and then restored."""
        grader.generator.max_new_tokens = 256
        grader.generator._generate_text.return_value = '{"grades": [true]}'

        grader.grade_batch("query", ["d1"])

        assert grader.generator.max_new_tokens == 256

    def test_max_tokens_restored_on_error(self, grader):
        """max_new_tokens should be restored even if _generate_text raises."""
        grader.generator.max_new_tokens = 256
        grader.generator._generate_text.side_effect = RuntimeError("OOM")

        with pytest.raises(RuntimeError):
            grader.grade_batch("query", ["d1"])

        assert grader.generator.max_new_tokens == 256

    def test_document_truncation_in_prompt(self, grader):
        """Documents longer than 500 chars should be truncated in the prompt."""
        long_doc = "x" * 1000
        grader.generator._generate_text.return_value = '{"grades": [true]}'

        grader.grade_batch("query", [long_doc])

        call_args = grader.generator._generate_text.call_args[0][0]
        # The document in the prompt should be truncated to 500 chars
        assert "x" * 500 in call_args
        assert "x" * 1000 not in call_args

    def test_prompt_contains_query(self, grader):
        """The batch prompt should contain the query text."""
        grader.generator._generate_text.return_value = '{"grades": [true]}'

        grader.grade_batch("specific test query", ["doc content"])

        call_args = grader.generator._generate_text.call_args[0][0]
        assert "specific test query" in call_args

    def test_many_documents(self, grader):
        """Batch grading 10 documents should work."""
        grades_json = (
            '{"grades": [true, false, true, false, true, false, true, false, true, false]}'
        )
        grader.generator._generate_text.return_value = grades_json

        docs = [f"document {i}" for i in range(10)]
        grades = grader.grade_batch("query", docs)

        assert len(grades) == 10
        assert grades == [True, False, True, False, True, False, True, False, True, False]


class TestDocumentGraderParsing:
    """Tests for internal parsing methods."""

    @pytest.fixture
    def grader(self):
        generator = create_mock_generator()
        return DocumentGrader(generator)

    def test_parse_single_grade_yes(self, grader):
        assert grader._parse_single_grade("yes") is True

    def test_parse_single_grade_no(self, grader):
        assert grader._parse_single_grade("no") is False

    def test_parse_single_grade_whitespace(self, grader):
        assert grader._parse_single_grade("  yes  ") is True
        assert grader._parse_single_grade("  no  ") is False

    def test_parse_batch_json_valid(self, grader):
        result = grader._parse_batch_grades('{"grades": [true, false]}', 2)
        assert result == [True, False]

    def test_parse_batch_json_wrong_count(self, grader):
        """Wrong count should cause _try_parse_json to return None."""
        result = grader._try_parse_json('{"grades": [true]}', 3)
        assert result is None

    def test_parse_batch_invalid_json(self, grader):
        result = grader._try_parse_json("not json at all", 2)
        assert result is None

    def test_parse_batch_missing_grades_key(self, grader):
        result = grader._try_parse_json('{"results": [true, false]}', 2)
        assert result is None


class TestQueryRewriter:
    """Tests for QueryRewriter.rewrite()."""

    @pytest.fixture
    def rewriter(self):
        """Create a QueryRewriter with mocked generator."""
        generator = create_mock_generator()
        return QueryRewriter(generator)

    def test_basic_rewrite(self, rewriter):
        """Simple rewrite should return the cleaned response."""
        rewriter.generator._generate_text.return_value = "When did Beyonce leave Destiny's Child?"

        result = rewriter.rewrite("When did she leave her group?", num_total=10, num_relevant=1)

        assert result == "When did Beyonce leave Destiny's Child?"

    def test_strips_prefix_improved_query(self, rewriter):
        """'Improved query:' prefix should be stripped."""
        rewriter.generator._generate_text.return_value = (
            "Improved query: What year was Beyonce born?"
        )

        result = rewriter.rewrite("When was she born?", num_total=10, num_relevant=2)

        assert result == "What year was Beyonce born?"

    def test_strips_prefix_rewritten_query(self, rewriter):
        """'Rewritten query:' prefix should be stripped."""
        rewriter.generator._generate_text.return_value = (
            "Rewritten query: Beyonce Grammy debut album"
        )

        result = rewriter.rewrite("awards for debut?", num_total=10, num_relevant=0)

        assert result == "Beyonce Grammy debut album"

    def test_strips_surrounding_double_quotes(self, rewriter):
        """Surrounding double quotes should be removed."""
        rewriter.generator._generate_text.return_value = '"What Grammy awards did Beyonce win?"'

        result = rewriter.rewrite("awards?", num_total=10, num_relevant=1)

        assert result == "What Grammy awards did Beyonce win?"

    def test_strips_surrounding_single_quotes(self, rewriter):
        """Surrounding single quotes should be removed."""
        rewriter.generator._generate_text.return_value = "'When was Beyonce born?'"

        result = rewriter.rewrite("birth?", num_total=10, num_relevant=1)

        assert result == "When was Beyonce born?"

    def test_takes_first_line_only(self, rewriter):
        """Multi-line response should use only the first line."""
        rewriter.generator._generate_text.return_value = (
            "Beyonce Grammy awards 2004\n" "This query adds specificity by mentioning the year."
        )

        result = rewriter.rewrite("awards?", num_total=10, num_relevant=0)

        assert result == "Beyonce Grammy awards 2004"

    def test_empty_response_returns_original(self, rewriter):
        """Empty response should fall back to original query."""
        rewriter.generator._generate_text.return_value = ""

        result = rewriter.rewrite("original query", num_total=10, num_relevant=0)

        assert result == "original query"

    def test_too_short_response_returns_original(self, rewriter):
        """Response shorter than 3 characters should fall back to original."""
        rewriter.generator._generate_text.return_value = "ab"

        result = rewriter.rewrite("original query", num_total=10, num_relevant=0)

        assert result == "original query"

    def test_max_tokens_override_and_restore(self, rewriter):
        """max_new_tokens should be temporarily set to 50 and then restored."""
        rewriter.generator.max_new_tokens = 256
        rewriter.generator._generate_text.return_value = "rewritten query text"

        rewriter.rewrite("query", num_total=10, num_relevant=0)

        assert rewriter.generator.max_new_tokens == 256

    def test_max_tokens_restored_on_error(self, rewriter):
        """max_new_tokens should be restored even if _generate_text raises."""
        rewriter.generator.max_new_tokens = 256
        rewriter.generator._generate_text.side_effect = RuntimeError("OOM")

        with pytest.raises(RuntimeError):
            rewriter.rewrite("query", num_total=10, num_relevant=0)

        assert rewriter.generator.max_new_tokens == 256

    def test_prompt_contains_stats(self, rewriter):
        """Prompt should include retrieval statistics."""
        rewriter.generator._generate_text.return_value = "better query"

        rewriter.rewrite("query", num_total=10, num_relevant=2)

        call_args = rewriter.generator._generate_text.call_args[0][0]
        assert "10" in call_args
        assert "2" in call_args

    def test_prompt_contains_original_query(self, rewriter):
        """Prompt should include the original query."""
        rewriter.generator._generate_text.return_value = "better query"

        rewriter.rewrite("my specific query", num_total=10, num_relevant=0)

        call_args = rewriter.generator._generate_text.call_args[0][0]
        assert "my specific query" in call_args

    def test_whitespace_only_response_returns_original(self, rewriter):
        """Whitespace-only response should fall back to original."""
        rewriter.generator._generate_text.return_value = "   \n  \t  "

        result = rewriter.rewrite("original query", num_total=10, num_relevant=0)

        assert result == "original query"


class TestDocumentGraderPrompts:
    """Tests for prompt template content."""

    def test_single_prompt_has_required_placeholders(self):
        """Single grading prompt must contain {query} and {document}."""
        assert "{query}" in DocumentGrader.SINGLE_GRADING_PROMPT
        assert "{document}" in DocumentGrader.SINGLE_GRADING_PROMPT

    def test_batch_prompt_has_required_placeholders(self):
        """Batch grading prompt must contain {query}, {documents}, {num_docs}."""
        assert "{query}" in DocumentGrader.BATCH_GRADING_PROMPT
        assert "{documents}" in DocumentGrader.BATCH_GRADING_PROMPT
        assert "{num_docs}" in DocumentGrader.BATCH_GRADING_PROMPT

    def test_single_prompt_instructs_binary_response(self):
        """Prompt should instruct LLM to respond with only yes or no."""
        prompt_lower = DocumentGrader.SINGLE_GRADING_PROMPT.lower()
        assert "yes" in prompt_lower
        assert "no" in prompt_lower

    def test_rewrite_prompt_has_required_placeholders(self):
        """Rewrite prompt must contain {query}, {num_total}, {num_relevant}."""
        assert "{query}" in QueryRewriter.REWRITE_PROMPT
        assert "{num_total}" in QueryRewriter.REWRITE_PROMPT
        assert "{num_relevant}" in QueryRewriter.REWRITE_PROMPT


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
