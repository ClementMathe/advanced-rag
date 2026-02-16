"""
Unit tests for error analysis taxonomy and visualizations.

Tests:
- ErrorAnalyzer: category assignment rules, normalization, scoring
- ErrorAnalysis: summary, serialization
- Visualizations: plot creation and file output

All tests use synthetic prediction dicts — no real models loaded.
"""

import json
from pathlib import Path

import pytest

from src.evaluation.error_taxonomy import (
    ErrorAnalysis,
    ErrorAnalyzer,
)

# ============================================================
# Helpers
# ============================================================


def make_pred(
    prediction="Paris",
    ground_truth="Paris",
    f1=1.0,
    exact_match=1.0,
    rouge_l=1.0,
    faithfulness=0.9,
    query="What is the capital of France?",
    contexts=None,
    **kwargs,
):
    """Create a synthetic prediction dict (flat format)."""
    result = {
        "query": query,
        "prediction": prediction,
        "ground_truth": ground_truth,
        "f1": f1,
        "exact_match": exact_match,
        "rouge_l": rouge_l,
        "faithfulness": faithfulness,
        "retrieved_contexts": contexts or ["Paris is the capital of France."],
    }
    result.update(kwargs)
    return result


def make_nested_pred(**kwargs):
    """Create a prediction dict in nested (script) format."""
    flat = make_pred(**kwargs)
    metrics = {}
    for key in ["f1", "exact_match", "rouge_l", "faithfulness"]:
        if key in flat:
            metrics[key] = flat.pop(key)
    flat["metrics"] = metrics
    return flat


# ============================================================
# TestNormalization
# ============================================================


class TestNormalization:
    def test_flat_format_unchanged(self):
        analyzer = ErrorAnalyzer()
        pred = make_pred(f1=0.5)
        result = analyzer._normalize_prediction(pred)
        assert result["f1"] == 0.5
        assert "metrics" not in result

    def test_nested_format_flattened(self):
        analyzer = ErrorAnalyzer()
        pred = make_nested_pred(f1=0.5, faithfulness=0.8)
        result = analyzer._normalize_prediction(pred)
        assert result["f1"] == 0.5
        assert result["faithfulness"] == 0.8
        assert "metrics" not in result

    def test_flat_keys_not_overwritten_by_nested(self):
        """If a key exists at top level AND in metrics, top level wins."""
        analyzer = ErrorAnalyzer()
        pred = {
            "f1": 0.9,
            "metrics": {"f1": 0.1},
            "prediction": "test",
            "ground_truth": "test",
        }
        result = analyzer._normalize_prediction(pred)
        assert result["f1"] == 0.9


# ============================================================
# TestErrorCategories
# ============================================================


class TestErrorCategories:
    @pytest.fixture
    def analyzer(self):
        return ErrorAnalyzer()

    def test_correct_exact_match(self, analyzer):
        pred = make_pred(exact_match=1.0, f1=1.0)
        cats = analyzer._categorize_single(analyzer._normalize_prediction(pred))
        assert cats == ["correct"]

    def test_correct_high_f1(self, analyzer):
        pred = make_pred(exact_match=0.0, f1=0.85)
        cats = analyzer._categorize_single(analyzer._normalize_prediction(pred))
        assert cats == ["correct"]

    def test_empty_response(self, analyzer):
        pred = make_pred(prediction="", f1=0.0, exact_match=0.0)
        cats = analyzer._categorize_single(analyzer._normalize_prediction(pred))
        assert cats == ["empty_response"]

    def test_empty_whitespace_response(self, analyzer):
        pred = make_pred(prediction="   ", f1=0.0, exact_match=0.0)
        cats = analyzer._categorize_single(analyzer._normalize_prediction(pred))
        assert cats == ["empty_response"]

    def test_retrieval_failure(self, analyzer):
        pred = make_pred(
            prediction="London",
            ground_truth="Paris",
            f1=0.0,
            exact_match=0.0,
            faithfulness=0.8,
            contexts=["London is the capital of England."],
        )
        cats = analyzer._categorize_single(analyzer._normalize_prediction(pred))
        assert "retrieval_failure" in cats

    def test_hallucination(self, analyzer):
        pred = make_pred(
            prediction="Berlin is the capital",
            ground_truth="Paris",
            f1=0.15,
            exact_match=0.0,
            faithfulness=0.2,
            contexts=["Paris is the capital of France."],
        )
        cats = analyzer._categorize_single(analyzer._normalize_prediction(pred))
        assert "hallucination" in cats

    def test_incomplete_answer(self, analyzer):
        pred = make_pred(
            prediction="Um",
            ground_truth="Paris is the capital of France",
            f1=0.1,
            exact_match=0.0,
            faithfulness=0.8,
            contexts=["Paris is the capital of France."],
        )
        cats = analyzer._categorize_single(analyzer._normalize_prediction(pred))
        assert "incomplete_answer" in cats

    def test_verbose_answer(self, analyzer):
        pred = make_pred(
            prediction="Well the capital city of the French Republic is Paris",
            ground_truth="Paris",
            f1=0.4,
            exact_match=0.0,
            rouge_l=0.5,
            faithfulness=0.9,
            contexts=["Paris is the capital of France."],
        )
        cats = analyzer._categorize_single(analyzer._normalize_prediction(pred))
        assert "verbose_answer" in cats

    def test_wrong_answer(self, analyzer):
        pred = make_pred(
            prediction="London is the capital",
            ground_truth="Paris",
            f1=0.1,
            exact_match=0.0,
            rouge_l=0.1,
            faithfulness=0.8,
            contexts=["Paris is the capital of France."],
        )
        cats = analyzer._categorize_single(analyzer._normalize_prediction(pred))
        assert "wrong_answer" in cats

    def test_low_context_relevance(self, analyzer):
        """Low faithfulness but answer IS in context (not retrieval failure)."""
        pred = make_pred(
            prediction="Maybe it is Berlin",
            ground_truth="Paris",
            f1=0.0,
            exact_match=0.0,
            faithfulness=0.2,
            contexts=["Paris is the capital of France."],
        )
        cats = analyzer._categorize_single(analyzer._normalize_prediction(pred))
        # f1=0 so hallucination won't trigger (needs f1>0),
        # but faithfulness is low and answer is in context
        assert "low_context_relevance" in cats


# ============================================================
# TestMultipleCategories
# ============================================================


class TestMultipleCategories:
    @pytest.fixture
    def analyzer(self):
        return ErrorAnalyzer()

    def test_retrieval_failure_and_wrong_answer(self, analyzer):
        """Both retrieval failure and wrong answer can co-occur."""
        pred = make_pred(
            prediction="Berlin is the capital of Germany",
            ground_truth="Paris",
            f1=0.0,
            exact_match=0.0,
            rouge_l=0.0,
            faithfulness=0.8,
            contexts=["Berlin is in Germany."],
        )
        cats = analyzer._categorize_single(analyzer._normalize_prediction(pred))
        assert "retrieval_failure" in cats
        assert "wrong_answer" in cats

    def test_hallucination_and_incomplete(self, analyzer):
        """Short hallucinated answer can be both."""
        pred = make_pred(
            prediction="No",
            ground_truth="Paris is the capital",
            f1=0.1,
            exact_match=0.0,
            faithfulness=0.1,
            contexts=["Paris is the capital of France."],
        )
        cats = analyzer._categorize_single(analyzer._normalize_prediction(pred))
        assert "hallucination" in cats
        assert "incomplete_answer" in cats

    def test_correct_has_no_other_categories(self, analyzer):
        pred = make_pred(f1=0.9, exact_match=1.0, faithfulness=0.1)
        cats = analyzer._categorize_single(analyzer._normalize_prediction(pred))
        assert cats == ["correct"]


# ============================================================
# TestAnalyze
# ============================================================


class TestAnalyze:
    @pytest.fixture
    def analyzer(self):
        return ErrorAnalyzer(max_examples=2, n_worst=3)

    @pytest.fixture
    def mixed_predictions(self):
        return [
            make_pred(f1=1.0, exact_match=1.0),  # correct
            make_pred(f1=0.9, exact_match=0.0),  # correct (high f1)
            make_pred(prediction="", f1=0.0, exact_match=0.0),  # empty
            make_pred(
                prediction="Wrong",
                ground_truth="Paris",
                f1=0.0,
                exact_match=0.0,
                faithfulness=0.8,
                contexts=["Paris is the capital."],
            ),  # wrong + incomplete
            make_pred(
                prediction="Hallucinated answer here",
                ground_truth="Paris",
                f1=0.1,
                exact_match=0.0,
                faithfulness=0.2,
                contexts=["Paris is the capital."],
            ),  # hallucination
        ]

    def test_error_distribution_counts(self, analyzer, mixed_predictions):
        analysis = analyzer.analyze(mixed_predictions)
        assert analysis.error_distribution["correct"] == 2
        assert analysis.error_distribution["empty_response"] == 1

    def test_error_rate(self, analyzer, mixed_predictions):
        analysis = analyzer.analyze(mixed_predictions)
        # 2 correct out of 5 → error rate = 3/5 = 0.6
        assert analysis.error_rate == pytest.approx(0.6, abs=0.01)

    def test_total_predictions(self, analyzer, mixed_predictions):
        analysis = analyzer.analyze(mixed_predictions)
        assert analysis.total_predictions == 5

    def test_per_query_errors_length(self, analyzer, mixed_predictions):
        analysis = analyzer.analyze(mixed_predictions)
        assert len(analysis.per_query_errors) == 5

    def test_worst_predictions_limited(self, analyzer, mixed_predictions):
        analysis = analyzer.analyze(mixed_predictions)
        assert len(analysis.worst_predictions) <= 3

    def test_worst_predictions_ordered(self, analyzer, mixed_predictions):
        analysis = analyzer.analyze(mixed_predictions)
        # Worst should have lowest composite score
        if len(analysis.worst_predictions) >= 2:
            scores = [ErrorAnalyzer._composite_badness_score(p) for p in analysis.worst_predictions]
            assert scores == sorted(scores)

    def test_examples_per_category_limited(self, analyzer, mixed_predictions):
        analysis = analyzer.analyze(mixed_predictions)
        for _cat, examples in analysis.examples_per_category.items():
            assert len(examples) <= 2  # max_examples=2

    def test_empty_predictions(self, analyzer):
        analysis = analyzer.analyze([])
        assert analysis.total_predictions == 0
        assert analysis.error_rate == 0.0
        assert analysis.per_query_errors == []


# ============================================================
# TestErrorAnalysisSummary
# ============================================================


class TestErrorAnalysisSummary:
    def test_summary_contains_counts(self):
        analysis = ErrorAnalysis(
            error_distribution={"correct": 8, "wrong_answer": 2},
            error_rate=0.2,
            total_predictions=10,
            per_query_errors=[],
            worst_predictions=[],
            examples_per_category={},
        )
        summary = analysis.summary()
        assert "10 predictions" in summary
        assert "20.0%" in summary
        assert "correct" in summary

    def test_to_dict_keys(self):
        analysis = ErrorAnalysis(
            error_distribution={},
            error_rate=0.0,
            total_predictions=0,
            per_query_errors=[],
            worst_predictions=[],
            examples_per_category={},
        )
        d = analysis.to_dict()
        assert "error_distribution" in d
        assert "error_rate" in d
        assert "total_predictions" in d


# ============================================================
# TestExportReport
# ============================================================


class TestExportReport:
    def test_export_creates_file(self, tmp_path):
        analyzer = ErrorAnalyzer()
        preds = [make_pred(f1=1.0, exact_match=1.0)]
        analysis = analyzer.analyze(preds)

        output = str(tmp_path / "report.json")
        path = analyzer.export_report(analysis, output)
        assert path.exists()

    def test_export_roundtrip(self, tmp_path):
        analyzer = ErrorAnalyzer()
        preds = [
            make_pred(f1=1.0, exact_match=1.0),
            make_pred(prediction="wrong", f1=0.0, exact_match=0.0),
        ]
        analysis = analyzer.analyze(preds)

        output = str(tmp_path / "report.json")
        analyzer.export_report(analysis, output)

        with open(output) as f:
            loaded = json.load(f)

        assert loaded["total_predictions"] == 2
        assert "error_distribution" in loaded

    def test_export_creates_parent_dirs(self, tmp_path):
        analyzer = ErrorAnalyzer()
        analysis = analyzer.analyze([make_pred()])
        output = str(tmp_path / "nested" / "dir" / "report.json")
        path = analyzer.export_report(analysis, output)
        assert path.exists()


# ============================================================
# TestCompositeBadnessScore
# ============================================================


class TestCompositeBadnessScore:
    def test_perfect_score(self):
        pred = make_pred(f1=1.0, exact_match=1.0, faithfulness=1.0)
        score = ErrorAnalyzer._composite_badness_score(pred)
        assert score == pytest.approx(1.0)

    def test_zero_score(self):
        pred = make_pred(f1=0.0, exact_match=0.0, faithfulness=0.0)
        score = ErrorAnalyzer._composite_badness_score(pred)
        assert score == pytest.approx(0.0)

    def test_missing_faithfulness_uses_default(self):
        pred = {"f1": 0.5, "exact_match": 0.0}
        score = ErrorAnalyzer._composite_badness_score(pred)
        # 0.5*0.5 + 0.3*0.5 + 0.2*0.0 = 0.25 + 0.15 = 0.40
        assert score == pytest.approx(0.40, abs=0.01)


# ============================================================
# TestVisualizations
# ============================================================


class TestVisualizations:
    def test_plot_error_distribution_creates_file(self, tmp_path):
        from src.evaluation.visualizations import plot_error_distribution

        analysis = ErrorAnalysis(
            error_distribution={
                "correct": 7,
                "wrong_answer": 2,
                "hallucination": 1,
            },
            error_rate=0.3,
            total_predictions=10,
            per_query_errors=[],
            worst_predictions=[],
            examples_per_category={},
        )
        output = str(tmp_path / "error_dist.png")
        plot_error_distribution(analysis, output)
        assert Path(output).exists()
        assert Path(output).stat().st_size > 0

    def test_plot_metric_comparison_creates_file(self, tmp_path):
        from src.evaluation.visualizations import plot_metric_comparison

        configs = {
            "linear": {"f1": {"mean": 0.5}, "rouge_l": {"mean": 0.4}},
            "agentic": {"f1": {"mean": 0.6}, "rouge_l": {"mean": 0.5}},
        }
        output = str(tmp_path / "metric_cmp.png")
        plot_metric_comparison(configs, output)
        assert Path(output).exists()
        assert Path(output).stat().st_size > 0

    def test_plot_error_distribution_empty(self, tmp_path):
        from src.evaluation.visualizations import plot_error_distribution

        analysis = ErrorAnalysis(
            error_distribution={},
            error_rate=0.0,
            total_predictions=0,
            per_query_errors=[],
            worst_predictions=[],
            examples_per_category={},
        )
        output = str(tmp_path / "empty.png")
        plot_error_distribution(analysis, output)
        assert Path(output).exists()
