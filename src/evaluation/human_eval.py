"""
Human evaluation protocol for RAG system outputs.

Provides:
- Quality rubric with 3 dimensions (relevance, faithfulness, conciseness)
- Stratified sampling for annotation task creation
- Annotation storage and loading (JSON-based)
- Inter-annotator agreement via Cohen's Kappa (quadratic weights)
- Annotation aggregation and correlation with automated metrics

Usage::

    protocol = HumanEvalProtocol()
    task = protocol.create_annotation_task(predictions, n_samples=50)
    # ... annotators complete task ...
    agreement = protocol.compute_agreement("ann1", "ann2")
    print(agreement.summary())
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger
from scipy import stats
from sklearn.metrics import cohen_kappa_score

from src.evaluation.error_taxonomy import ErrorAnalyzer

# ---------------------------------------------------------------------------
# Quality rubric
# ---------------------------------------------------------------------------

QUALITY_RUBRIC: Dict[str, Dict[str, Any]] = {
    "relevance": {
        "description": "Does the answer address the query?",
        "scale": {
            1: "Completely irrelevant — answer does not address the query at all",
            2: "Mostly irrelevant — tangentially related but misses the point",
            3: "Partially relevant — addresses query but missing key information",
            4: "Mostly relevant — addresses query with minor gaps",
            5: "Fully relevant — directly and completely addresses the query",
        },
    },
    "faithfulness": {
        "description": "Is the answer supported by the retrieved contexts?",
        "scale": {
            1: "Hallucinated — no support in contexts",
            2: "Mostly unsupported — minor overlap with contexts",
            3: "Partially supported — some claims supported, some not",
            4: "Mostly supported — main claims grounded in contexts",
            5: "Fully supported — every claim traceable to contexts",
        },
    },
    "conciseness": {
        "description": "Is the answer appropriately concise?",
        "scale": {
            1: "Extremely verbose — mostly filler or repetition",
            2: "Verbose — significant unnecessary content",
            3: "Acceptable — some unnecessary content but mostly focused",
            4: "Concise — minimal unnecessary content",
            5: "Perfectly concise — every word contributes to the answer",
        },
    },
}

# Map human eval dimensions to automated metric keys
_DIMENSION_TO_METRIC: Dict[str, str] = {
    "relevance": "f1",
    "faithfulness": "faithfulness",
    "conciseness": "rouge_l",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class HumanAnnotation:
    """A single human annotation for one prediction."""

    sample_id: str
    annotator_id: str
    ratings: Dict[str, int]  # dimension -> 1-5 rating
    notes: str = ""
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "annotator_id": self.annotator_id,
            "ratings": self.ratings,
            "notes": self.notes,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> HumanAnnotation:
        return cls(
            sample_id=data["sample_id"],
            annotator_id=data["annotator_id"],
            ratings=data["ratings"],
            notes=data.get("notes", ""),
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class AnnotationTask:
    """A batch of samples to annotate with instructions."""

    task_id: str
    samples: List[Dict[str, Any]]
    rubric: Dict[str, Any]
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "samples": self.samples,
            "rubric": {
                dim: {
                    "description": info["description"],
                    "scale": {str(k): v for k, v in info["scale"].items()},
                }
                for dim, info in self.rubric.items()
            },
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AnnotationTask:
        rubric = {}
        for dim, info in data["rubric"].items():
            rubric[dim] = {
                "description": info["description"],
                "scale": {int(k): v for k, v in info["scale"].items()},
            }
        return cls(
            task_id=data["task_id"],
            samples=data["samples"],
            rubric=rubric,
            created_at=data.get("created_at", ""),
        )


@dataclass
class AgreementResult:
    """Inter-annotator agreement results."""

    dimension_kappas: Dict[str, float]
    average_kappa: float
    n_common: int
    interpretation: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.interpretation:
            self.interpretation = {
                dim: HumanEvalProtocol.interpret_kappa(k)
                for dim, k in self.dimension_kappas.items()
            }

    def summary(self) -> str:
        lines = [
            f"Inter-Annotator Agreement ({self.n_common} common samples)",
            f"Average Kappa: {self.average_kappa:.3f} "
            f"({HumanEvalProtocol.interpret_kappa(self.average_kappa)})",
            "",
        ]
        for dim, kappa in self.dimension_kappas.items():
            interp = self.interpretation.get(dim, "")
            lines.append(f"  {dim:<20s} kappa={kappa:.3f}  ({interp})")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension_kappas": self.dimension_kappas,
            "average_kappa": self.average_kappa,
            "n_common": self.n_common,
            "interpretation": self.interpretation,
        }


# ---------------------------------------------------------------------------
# HumanEvalProtocol
# ---------------------------------------------------------------------------


class HumanEvalProtocol:
    """
    Human evaluation protocol for RAG outputs.

    Manages sample selection (stratified/random/worst), annotation storage,
    inter-annotator agreement, and annotation aggregation.

    Args:
        dimensions: Rating dimensions. Defaults to QUALITY_RUBRIC keys.
        min_annotations_per_sample: Minimum annotators per sample for
            aggregation.
        output_dir: Directory for storing tasks and annotations.
    """

    def __init__(
        self,
        dimensions: Optional[List[str]] = None,
        min_annotations_per_sample: int = 2,
        output_dir: str | Path = "outputs/human_eval",
    ) -> None:
        self.dimensions = dimensions or list(QUALITY_RUBRIC.keys())
        self.min_annotations_per_sample = min_annotations_per_sample
        self.output_dir = Path(output_dir)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def create_annotation_task(
        self,
        predictions: List[Dict[str, Any]],
        n_samples: int = 50,
        strategy: str = "stratified",
        seed: int = 42,
    ) -> AnnotationTask:
        """
        Select samples for annotation and create a task.

        Args:
            predictions: Full list of per-query prediction dicts.
            n_samples: Number of samples to select.
            strategy: "stratified" (proportional per error category),
                "worst" (lowest composite score), or "random".
            seed: Random seed for reproducibility.

        Returns:
            AnnotationTask with selected samples and rubric.

        Raises:
            ValueError: If strategy is unknown or predictions empty.
        """
        if not predictions:
            raise ValueError("Predictions list must not be empty")

        n_samples = min(n_samples, len(predictions))

        if strategy == "stratified":
            selected = self._stratified_sample(predictions, n_samples, seed)
        elif strategy == "worst":
            selected = self._worst_sample(predictions, n_samples)
        elif strategy == "random":
            selected = self._random_sample(predictions, n_samples, seed)
        else:
            raise ValueError(
                f"Unknown strategy '{strategy}'. " f"Choose from: stratified, worst, random"
            )

        # Assign sample IDs
        for i, sample in enumerate(selected):
            if "sample_id" not in sample:
                sample["sample_id"] = f"s{i:04d}"

        task = AnnotationTask(
            task_id=str(uuid.uuid4())[:8],
            samples=selected,
            rubric={dim: QUALITY_RUBRIC[dim] for dim in self.dimensions if dim in QUALITY_RUBRIC},
        )

        # Save task
        tasks_dir = self.output_dir / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        task_path = tasks_dir / f"{task.task_id}.json"
        with open(task_path, "w") as f:
            json.dump(task.to_dict(), f, indent=2)

        logger.info(
            f"Created annotation task {task.task_id}: "
            f"{len(selected)} samples ({strategy} sampling)"
        )
        return task

    def _stratified_sample(
        self,
        predictions: List[Dict[str, Any]],
        n_samples: int,
        seed: int,
    ) -> List[Dict[str, Any]]:
        """Sample proportionally from error categories."""
        analyzer = ErrorAnalyzer()
        analysis = analyzer.analyze(predictions)

        # Group predictions by primary error category
        buckets: Dict[str, List[int]] = {}
        for i, errors in enumerate(analysis.per_query_errors):
            categories = errors.get("categories", [])
            primary = categories[0] if categories else "correct"
            buckets.setdefault(primary, []).append(i)

        # Compute proportional allocation
        rng = np.random.default_rng(seed)
        selected_indices: List[int] = []
        total = len(predictions)

        for _cat, indices in buckets.items():
            proportion = len(indices) / total
            n_from_cat = max(1, round(proportion * n_samples))
            n_from_cat = min(n_from_cat, len(indices))
            chosen = rng.choice(indices, size=n_from_cat, replace=False)
            selected_indices.extend(chosen.tolist())

        # Trim or pad to exact n_samples
        if len(selected_indices) > n_samples:
            rng.shuffle(selected_indices)
            selected_indices = selected_indices[:n_samples]
        elif len(selected_indices) < n_samples:
            remaining = set(range(total)) - set(selected_indices)
            if remaining:
                extra = rng.choice(
                    list(remaining),
                    size=min(n_samples - len(selected_indices), len(remaining)),
                    replace=False,
                )
                selected_indices.extend(extra.tolist())

        return [predictions[i] for i in selected_indices]

    def _worst_sample(
        self,
        predictions: List[Dict[str, Any]],
        n_samples: int,
    ) -> List[Dict[str, Any]]:
        """Select worst predictions by composite badness score."""
        scored = []
        for pred in predictions:
            f1 = pred.get("f1", 0.0)
            faith = pred.get("faithfulness", 0.0)
            em = pred.get("exact_match", 0.0)
            score = 0.5 * f1 + 0.3 * faith + 0.2 * em
            scored.append((score, pred))
        scored.sort(key=lambda x: x[0])
        return [pred for _, pred in scored[:n_samples]]

    def _random_sample(
        self,
        predictions: List[Dict[str, Any]],
        n_samples: int,
        seed: int,
    ) -> List[Dict[str, Any]]:
        """Uniform random selection."""
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(predictions), size=n_samples, replace=False)
        return [predictions[i] for i in indices]

    # ------------------------------------------------------------------
    # Annotation management
    # ------------------------------------------------------------------

    def save_annotations(
        self,
        annotations: List[HumanAnnotation],
    ) -> Path:
        """
        Save annotations to JSON file, grouped by annotator.

        Args:
            annotations: List of annotations (must share same annotator_id).

        Returns:
            Path to the saved annotation file.

        Raises:
            ValueError: If annotations list is empty or has mixed annotators.
        """
        if not annotations:
            raise ValueError("Annotations list must not be empty")

        annotator_ids = {a.annotator_id for a in annotations}
        if len(annotator_ids) > 1:
            raise ValueError(
                f"All annotations must share the same annotator_id, " f"got: {annotator_ids}"
            )

        annotator_id = annotations[0].annotator_id
        annot_dir = self.output_dir / "annotations"
        annot_dir.mkdir(parents=True, exist_ok=True)

        file_path = annot_dir / f"{annotator_id}.json"

        # Load existing annotations if file exists
        existing: List[Dict[str, Any]] = []
        if file_path.exists():
            with open(file_path) as f:
                existing = json.load(f)

        # Merge: replace if same sample_id, append if new
        existing_ids = {e["sample_id"]: i for i, e in enumerate(existing)}
        for ann in annotations:
            d = ann.to_dict()
            if ann.sample_id in existing_ids:
                existing[existing_ids[ann.sample_id]] = d
            else:
                existing.append(d)

        with open(file_path, "w") as f:
            json.dump(existing, f, indent=2)

        logger.info(
            f"Saved {len(annotations)} annotations for annotator "
            f"'{annotator_id}' to {file_path}"
        )
        return file_path

    def load_annotations(
        self,
        annotator_id: Optional[str] = None,
    ) -> List[HumanAnnotation]:
        """
        Load annotations from disk.

        Args:
            annotator_id: If provided, load only this annotator's data.
                If None, load all annotators.

        Returns:
            List of HumanAnnotation objects.
        """
        annot_dir = self.output_dir / "annotations"
        if not annot_dir.exists():
            return []

        files: List[Path] = []
        if annotator_id is not None:
            p = annot_dir / f"{annotator_id}.json"
            if p.exists():
                files.append(p)
        else:
            files = sorted(annot_dir.glob("*.json"))

        annotations: List[HumanAnnotation] = []
        for fp in files:
            with open(fp) as f:
                data = json.load(f)
            for item in data:
                annotations.append(HumanAnnotation.from_dict(item))

        return annotations

    # ------------------------------------------------------------------
    # Inter-annotator agreement
    # ------------------------------------------------------------------

    def compute_agreement(
        self,
        annotator1_id: str,
        annotator2_id: str,
    ) -> AgreementResult:
        """
        Compute Cohen's Kappa between two annotators on common samples.

        Uses quadratic weights (appropriate for ordinal 1-5 scale).

        Args:
            annotator1_id: First annotator identifier.
            annotator2_id: Second annotator identifier.

        Returns:
            AgreementResult with per-dimension kappas.

        Raises:
            ValueError: If fewer than 2 common annotations found.
        """
        ann1 = self.load_annotations(annotator1_id)
        ann2 = self.load_annotations(annotator2_id)

        # Index by sample_id
        map1 = {a.sample_id: a for a in ann1}
        map2 = {a.sample_id: a for a in ann2}
        common_ids = sorted(set(map1.keys()) & set(map2.keys()))

        if len(common_ids) < 2:
            raise ValueError(f"Need at least 2 common annotations, " f"found {len(common_ids)}")

        dimension_kappas: Dict[str, float] = {}
        for dim in self.dimensions:
            ratings1 = [map1[sid].ratings.get(dim, 3) for sid in common_ids]
            ratings2 = [map2[sid].ratings.get(dim, 3) for sid in common_ids]

            kappa = cohen_kappa_score(
                ratings1, ratings2, weights="quadratic", labels=[1, 2, 3, 4, 5]
            )
            # cohen_kappa_score returns NaN when all ratings are identical
            # (zero variance). In that case, perfect agreement = 1.0.
            if np.isnan(kappa) and ratings1 == ratings2:
                kappa = 1.0
            dimension_kappas[dim] = float(kappa)

        avg = float(np.mean(list(dimension_kappas.values())))

        result = AgreementResult(
            dimension_kappas=dimension_kappas,
            average_kappa=avg,
            n_common=len(common_ids),
        )

        logger.info(
            f"Agreement between '{annotator1_id}' and '{annotator2_id}': "
            f"avg_kappa={avg:.3f} ({self.interpret_kappa(avg)}), "
            f"n={len(common_ids)}"
        )
        return result

    @staticmethod
    def interpret_kappa(kappa: float) -> str:
        """Landis-Koch interpretation of kappa value."""
        if kappa < 0:
            return "No agreement"
        elif kappa < 0.20:
            return "Slight agreement"
        elif kappa < 0.40:
            return "Fair agreement"
        elif kappa < 0.60:
            return "Moderate agreement"
        elif kappa < 0.80:
            return "Substantial agreement"
        else:
            return "Almost perfect agreement"

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate_annotations(self) -> Dict[str, Dict[str, float]]:
        """
        Aggregate all annotations into per-sample mean ratings.

        Returns:
            Dict mapping sample_id to {dimension: mean_rating, ...}.
            Only includes samples with at least min_annotations_per_sample
            annotations.
        """
        all_annotations = self.load_annotations()
        if not all_annotations:
            return {}

        # Group by sample_id
        by_sample: Dict[str, List[HumanAnnotation]] = {}
        for ann in all_annotations:
            by_sample.setdefault(ann.sample_id, []).append(ann)

        aggregated: Dict[str, Dict[str, float]] = {}
        for sample_id, anns in by_sample.items():
            if len(anns) < self.min_annotations_per_sample:
                continue

            dim_ratings: Dict[str, List[int]] = {d: [] for d in self.dimensions}
            for ann in anns:
                for dim in self.dimensions:
                    if dim in ann.ratings:
                        dim_ratings[dim].append(ann.ratings[dim])

            aggregated[sample_id] = {
                dim: float(np.mean(vals)) if vals else 0.0 for dim, vals in dim_ratings.items()
            }

        return aggregated

    def correlation_with_automated(
        self,
        predictions: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Compute Spearman correlation between human ratings and automated metrics.

        Maps dimensions to automated metrics:
        - relevance <-> f1
        - faithfulness <-> faithfulness
        - conciseness <-> rouge_l

        Args:
            predictions: Per-query prediction dicts with sample_id and
                automated metric values.

        Returns:
            Dict mapping dimension to Spearman rho. Only includes
            dimensions with at least 3 paired values.
        """
        aggregated = self.aggregate_annotations()
        if not aggregated:
            return {}

        # Index predictions by sample_id
        pred_map = {p.get("sample_id", f"s{i:04d}"): p for i, p in enumerate(predictions)}

        correlations: Dict[str, float] = {}
        for dim, metric_key in _DIMENSION_TO_METRIC.items():
            if dim not in self.dimensions:
                continue

            human_vals: List[float] = []
            auto_vals: List[float] = []

            for sample_id, ratings in aggregated.items():
                pred = pred_map.get(sample_id)
                if pred is None:
                    continue
                auto_val = pred.get(metric_key)
                if auto_val is None:
                    continue
                human_vals.append(ratings.get(dim, 0.0))
                auto_vals.append(float(auto_val))

            if len(human_vals) < 3:
                continue

            rho, _ = stats.spearmanr(human_vals, auto_vals)
            correlations[dim] = float(rho)

        return correlations
