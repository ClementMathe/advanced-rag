"""
Unit tests for human evaluation protocol.

Tests:
- QUALITY_RUBRIC: structure validation
- HumanAnnotation / AnnotationTask / AgreementResult: data classes
- HumanEvalProtocol: sampling, annotation storage, agreement, aggregation
- Correlation with automated metrics

All tests use synthetic data — no real models or file I/O beyond tmp_path.
"""

import json

import numpy as np
import pytest

from src.evaluation.human_eval import (
    QUALITY_RUBRIC,
    AgreementResult,
    AnnotationTask,
    HumanAnnotation,
    HumanEvalProtocol,
)

# ============================================================
# Helpers
# ============================================================


def make_annotation(
    sample_id="s0000",
    annotator_id="ann1",
    relevance=4,
    faithfulness=4,
    conciseness=4,
    notes="",
):
    """Create a synthetic HumanAnnotation."""
    return HumanAnnotation(
        sample_id=sample_id,
        annotator_id=annotator_id,
        ratings={
            "relevance": relevance,
            "faithfulness": faithfulness,
            "conciseness": conciseness,
        },
        notes=notes,
    )


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
    """Create a synthetic prediction dict."""
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


def make_varied_predictions(n=30, seed=42):
    """Create predictions spanning multiple quality levels for sampling tests."""
    rng = np.random.default_rng(seed)
    preds = []
    for i in range(n):
        f1 = float(rng.uniform(0.0, 1.0))
        faith = float(rng.uniform(0.0, 1.0))
        preds.append(
            make_pred(
                prediction=f"answer_{i}" if f1 > 0.1 else "",
                ground_truth=f"truth_{i}",
                f1=f1,
                exact_match=1.0 if f1 > 0.95 else 0.0,
                rouge_l=f1 * 0.9,
                faithfulness=faith,
                query=f"question_{i}",
                contexts=[f"context for question {i}"] if faith > 0.3 else [],
            )
        )
    return preds


# ============================================================
# TestQualityRubric
# ============================================================


class TestQualityRubric:
    def test_has_three_dimensions(self):
        assert set(QUALITY_RUBRIC.keys()) == {"relevance", "faithfulness", "conciseness"}

    def test_each_dimension_has_description(self):
        for _dim, info in QUALITY_RUBRIC.items():
            assert "description" in info
            assert isinstance(info["description"], str)

    def test_each_dimension_has_5_point_scale(self):
        for _dim, info in QUALITY_RUBRIC.items():
            scale = info["scale"]
            assert set(scale.keys()) == {1, 2, 3, 4, 5}
            for val in scale.values():
                assert isinstance(val, str)


# ============================================================
# TestHumanAnnotation
# ============================================================


class TestHumanAnnotation:
    def test_creation(self):
        ann = make_annotation()
        assert ann.sample_id == "s0000"
        assert ann.annotator_id == "ann1"
        assert ann.ratings["relevance"] == 4
        assert ann.timestamp  # auto-generated

    def test_to_dict(self):
        ann = make_annotation(notes="good answer")
        d = ann.to_dict()
        assert d["sample_id"] == "s0000"
        assert d["notes"] == "good answer"
        assert "timestamp" in d

    def test_from_dict_roundtrip(self):
        ann = make_annotation(relevance=5, faithfulness=3)
        d = ann.to_dict()
        restored = HumanAnnotation.from_dict(d)
        assert restored.ratings == ann.ratings
        assert restored.sample_id == ann.sample_id


# ============================================================
# TestAnnotationTask
# ============================================================


class TestAnnotationTask:
    def test_creation(self):
        task = AnnotationTask(
            task_id="test123",
            samples=[{"query": "q1"}],
            rubric=QUALITY_RUBRIC,
        )
        assert task.task_id == "test123"
        assert len(task.samples) == 1
        assert task.created_at  # auto-generated

    def test_to_dict_serializes_rubric(self):
        task = AnnotationTask(
            task_id="t1",
            samples=[],
            rubric=QUALITY_RUBRIC,
        )
        d = task.to_dict()
        # Rubric scale keys should be strings (JSON-safe)
        for dim_info in d["rubric"].values():
            for key in dim_info["scale"]:
                assert isinstance(key, str)

    def test_from_dict_roundtrip(self):
        task = AnnotationTask(
            task_id="t1",
            samples=[{"query": "q1", "prediction": "a1"}],
            rubric=QUALITY_RUBRIC,
        )
        d = task.to_dict()
        restored = AnnotationTask.from_dict(d)
        assert restored.task_id == task.task_id
        assert len(restored.samples) == 1
        # Scale keys should be ints after roundtrip
        assert 1 in restored.rubric["relevance"]["scale"]


# ============================================================
# TestAgreementResult
# ============================================================


class TestAgreementResult:
    def test_summary_contains_key_info(self):
        result = AgreementResult(
            dimension_kappas={"relevance": 0.7, "faithfulness": 0.5},
            average_kappa=0.6,
            n_common=20,
        )
        summary = result.summary()
        assert "20 common samples" in summary
        assert "0.600" in summary

    def test_to_dict(self):
        result = AgreementResult(
            dimension_kappas={"relevance": 0.8},
            average_kappa=0.8,
            n_common=10,
        )
        d = result.to_dict()
        assert "dimension_kappas" in d
        assert d["n_common"] == 10

    def test_auto_interpretation(self):
        result = AgreementResult(
            dimension_kappas={"relevance": 0.85, "faithfulness": 0.45},
            average_kappa=0.65,
            n_common=15,
        )
        assert result.interpretation["relevance"] == "Almost perfect agreement"
        assert result.interpretation["faithfulness"] == "Moderate agreement"


# ============================================================
# TestInterpretKappa
# ============================================================


class TestInterpretKappa:
    def test_no_agreement(self):
        assert HumanEvalProtocol.interpret_kappa(-0.1) == "No agreement"

    def test_slight(self):
        assert HumanEvalProtocol.interpret_kappa(0.1) == "Slight agreement"

    def test_fair(self):
        assert HumanEvalProtocol.interpret_kappa(0.3) == "Fair agreement"

    def test_moderate(self):
        assert HumanEvalProtocol.interpret_kappa(0.5) == "Moderate agreement"

    def test_substantial(self):
        assert HumanEvalProtocol.interpret_kappa(0.7) == "Substantial agreement"

    def test_almost_perfect(self):
        assert HumanEvalProtocol.interpret_kappa(0.9) == "Almost perfect agreement"


# ============================================================
# TestCreateAnnotationTask
# ============================================================


class TestCreateAnnotationTask:
    @pytest.fixture
    def protocol(self, tmp_path):
        return HumanEvalProtocol(output_dir=tmp_path / "human_eval")

    def test_random_sampling(self, protocol):
        preds = make_varied_predictions(30)
        task = protocol.create_annotation_task(preds, n_samples=10, strategy="random")
        assert len(task.samples) == 10
        assert task.rubric  # rubric included

    def test_worst_sampling(self, protocol):
        preds = make_varied_predictions(30)
        task = protocol.create_annotation_task(preds, n_samples=5, strategy="worst")
        assert len(task.samples) == 5

    def test_stratified_sampling(self, protocol):
        preds = make_varied_predictions(30)
        task = protocol.create_annotation_task(preds, n_samples=10, strategy="stratified")
        assert len(task.samples) == 10

    def test_seed_reproducibility_random(self, protocol):
        preds = make_varied_predictions(30)
        t1 = protocol.create_annotation_task(preds, n_samples=5, strategy="random", seed=123)
        t2 = protocol.create_annotation_task(preds, n_samples=5, strategy="random", seed=123)
        q1 = [s.get("query") for s in t1.samples]
        q2 = [s.get("query") for s in t2.samples]
        assert q1 == q2

    def test_n_samples_capped_at_predictions_length(self, protocol):
        preds = make_varied_predictions(5)
        task = protocol.create_annotation_task(preds, n_samples=100, strategy="random")
        assert len(task.samples) == 5

    def test_unknown_strategy_raises(self, protocol):
        preds = make_varied_predictions(10)
        with pytest.raises(ValueError, match="Unknown strategy"):
            protocol.create_annotation_task(preds, strategy="invalid")

    def test_empty_predictions_raises(self, protocol):
        with pytest.raises(ValueError, match="not be empty"):
            protocol.create_annotation_task([], n_samples=5)

    def test_task_saved_to_disk(self, protocol):
        preds = make_varied_predictions(10)
        task = protocol.create_annotation_task(preds, n_samples=5, strategy="random")
        task_file = protocol.output_dir / "tasks" / f"{task.task_id}.json"
        assert task_file.exists()
        with open(task_file) as f:
            data = json.load(f)
        assert data["task_id"] == task.task_id

    def test_samples_get_sample_ids(self, protocol):
        preds = make_varied_predictions(10)
        task = protocol.create_annotation_task(preds, n_samples=5, strategy="random")
        for sample in task.samples:
            assert "sample_id" in sample


# ============================================================
# TestSaveLoadAnnotations
# ============================================================


class TestSaveLoadAnnotations:
    @pytest.fixture
    def protocol(self, tmp_path):
        return HumanEvalProtocol(output_dir=tmp_path / "human_eval")

    def test_save_and_load(self, protocol):
        anns = [make_annotation(sample_id=f"s{i}") for i in range(3)]
        protocol.save_annotations(anns)
        loaded = protocol.load_annotations("ann1")
        assert len(loaded) == 3
        assert loaded[0].sample_id == "s0"

    def test_save_merges_existing(self, protocol):
        ann1 = [make_annotation(sample_id="s0", relevance=3)]
        protocol.save_annotations(ann1)
        ann2 = [make_annotation(sample_id="s0", relevance=5)]
        protocol.save_annotations(ann2)
        loaded = protocol.load_annotations("ann1")
        assert len(loaded) == 1
        assert loaded[0].ratings["relevance"] == 5  # updated

    def test_save_appends_new(self, protocol):
        protocol.save_annotations([make_annotation(sample_id="s0")])
        protocol.save_annotations([make_annotation(sample_id="s1")])
        loaded = protocol.load_annotations("ann1")
        assert len(loaded) == 2

    def test_load_all_annotators(self, protocol):
        protocol.save_annotations([make_annotation(annotator_id="ann1")])
        protocol.save_annotations([make_annotation(annotator_id="ann2")])
        loaded = protocol.load_annotations()  # all
        assert len(loaded) == 2

    def test_load_nonexistent_returns_empty(self, protocol):
        loaded = protocol.load_annotations("nobody")
        assert loaded == []

    def test_empty_annotations_raises(self, protocol):
        with pytest.raises(ValueError, match="not be empty"):
            protocol.save_annotations([])

    def test_mixed_annotators_raises(self, protocol):
        anns = [
            make_annotation(annotator_id="ann1"),
            make_annotation(annotator_id="ann2"),
        ]
        with pytest.raises(ValueError, match="same annotator_id"):
            protocol.save_annotations(anns)


# ============================================================
# TestComputeAgreement
# ============================================================


class TestComputeAgreement:
    @pytest.fixture
    def protocol(self, tmp_path):
        return HumanEvalProtocol(output_dir=tmp_path / "human_eval")

    def test_perfect_agreement(self, protocol):
        anns1 = [make_annotation(sample_id=f"s{i}", annotator_id="a1") for i in range(10)]
        anns2 = [make_annotation(sample_id=f"s{i}", annotator_id="a2") for i in range(10)]
        protocol.save_annotations(anns1)
        protocol.save_annotations(anns2)
        result = protocol.compute_agreement("a1", "a2")
        assert result.average_kappa == pytest.approx(1.0, abs=0.01)
        assert result.n_common == 10

    def test_no_agreement(self, protocol):
        # Annotator 1: all 5s, Annotator 2: cycling 1-5
        anns1 = []
        anns2 = []
        for i in range(20):
            anns1.append(
                make_annotation(
                    sample_id=f"s{i}",
                    annotator_id="a1",
                    relevance=5,
                    faithfulness=5,
                    conciseness=5,
                )
            )
            rating = (i % 5) + 1
            anns2.append(
                make_annotation(
                    sample_id=f"s{i}",
                    annotator_id="a2",
                    relevance=rating,
                    faithfulness=rating,
                    conciseness=rating,
                )
            )
        protocol.save_annotations(anns1)
        protocol.save_annotations(anns2)
        result = protocol.compute_agreement("a1", "a2")
        # Kappa should be low (near 0 or negative)
        assert result.average_kappa < 0.3

    def test_partial_overlap(self, protocol):
        # Only 5 common samples out of 10 each
        anns1 = [make_annotation(sample_id=f"s{i}", annotator_id="a1") for i in range(10)]
        anns2 = [make_annotation(sample_id=f"s{i}", annotator_id="a2") for i in range(5, 15)]
        protocol.save_annotations(anns1)
        protocol.save_annotations(anns2)
        result = protocol.compute_agreement("a1", "a2")
        assert result.n_common == 5

    def test_too_few_common_raises(self, protocol):
        anns1 = [make_annotation(sample_id="s0", annotator_id="a1")]
        anns2 = [make_annotation(sample_id="s0", annotator_id="a2")]
        protocol.save_annotations(anns1)
        protocol.save_annotations(anns2)
        with pytest.raises(ValueError, match="at least 2"):
            protocol.compute_agreement("a1", "a2")

    def test_interpretation_in_result(self, protocol):
        anns1 = [make_annotation(sample_id=f"s{i}", annotator_id="a1") for i in range(5)]
        anns2 = [make_annotation(sample_id=f"s{i}", annotator_id="a2") for i in range(5)]
        protocol.save_annotations(anns1)
        protocol.save_annotations(anns2)
        result = protocol.compute_agreement("a1", "a2")
        for dim in ["relevance", "faithfulness", "conciseness"]:
            assert dim in result.interpretation


# ============================================================
# TestAggregateAnnotations
# ============================================================


class TestAggregateAnnotations:
    @pytest.fixture
    def protocol(self, tmp_path):
        return HumanEvalProtocol(
            output_dir=tmp_path / "human_eval",
            min_annotations_per_sample=2,
        )

    def test_aggregates_two_annotators(self, protocol):
        protocol.save_annotations(
            [
                make_annotation(sample_id="s0", annotator_id="a1", relevance=4, faithfulness=2),
            ]
        )
        protocol.save_annotations(
            [
                make_annotation(sample_id="s0", annotator_id="a2", relevance=2, faithfulness=4),
            ]
        )
        agg = protocol.aggregate_annotations()
        assert "s0" in agg
        assert agg["s0"]["relevance"] == pytest.approx(3.0)
        assert agg["s0"]["faithfulness"] == pytest.approx(3.0)

    def test_skips_single_annotator(self, protocol):
        protocol.save_annotations(
            [
                make_annotation(sample_id="s0", annotator_id="a1"),
            ]
        )
        agg = protocol.aggregate_annotations()
        assert "s0" not in agg  # min_annotations_per_sample=2

    def test_empty_returns_empty(self, protocol):
        agg = protocol.aggregate_annotations()
        assert agg == {}


# ============================================================
# TestCorrelationWithAutomated
# ============================================================


class TestCorrelationWithAutomated:
    @pytest.fixture
    def protocol(self, tmp_path):
        return HumanEvalProtocol(
            output_dir=tmp_path / "human_eval",
            min_annotations_per_sample=1,
        )

    def test_positive_correlation(self, protocol):
        # High human ratings correlate with high automated metrics
        for ann_id in ["a1", "a2"]:
            anns = []
            for i in range(10):
                rating = i // 2 + 1  # 1,1,2,2,3,3,4,4,5,5
                anns.append(
                    make_annotation(
                        sample_id=f"s{i:04d}",
                        annotator_id=ann_id,
                        relevance=rating,
                        faithfulness=rating,
                        conciseness=rating,
                    )
                )
            protocol.save_annotations(anns)

        # Automated metrics that correlate with ratings
        preds = []
        for i in range(10):
            score = i / 9  # 0.0 to 1.0
            preds.append(
                make_pred(
                    sample_id=f"s{i:04d}",
                    f1=score,
                    faithfulness=score,
                    rouge_l=score,
                )
            )

        corr = protocol.correlation_with_automated(preds)
        assert "relevance" in corr
        assert corr["relevance"] > 0.5  # positive correlation

    def test_no_annotations_returns_empty(self, protocol):
        preds = [make_pred(sample_id="s0")]
        corr = protocol.correlation_with_automated(preds)
        assert corr == {}

    def test_too_few_paired_values(self, protocol):
        protocol.save_annotations(
            [
                make_annotation(sample_id="s0000", annotator_id="a1"),
            ]
        )
        # Only 1 matching prediction — need at least 3
        preds = [make_pred(sample_id="s0000")]
        corr = protocol.correlation_with_automated(preds)
        assert corr == {}  # not enough data
