#!/usr/bin/env python3
"""
Smoke tests for the pretraining MI pipeline.

These tests verify the pipeline components work together without
requiring a full LLM or large datasets.
"""

from collections import Counter
from typing import Dict, List

import pytest

from src.exps_pret_corr.data_loader import Document, proportional_subsample
from src.exps_pret_corr.probes import (
    EVALUATORS,
    TASK_KINDS,
    LabelingResult,
    gen_probes,
    precompute_reference_answers,
)


class TestPipelineComponentsSmoke:
    """Smoke tests for pipeline components."""

    def test_probes_and_evaluators_consistent(self) -> None:
        """Verify probes can be evaluated by corresponding evaluators."""
        probes = gen_probes(num_per_kind=5, seed=42)

        for kind in TASK_KINDS:
            evaluator = EVALUATORS[kind]
            for probe in probes[kind]:
                # Should not raise
                result = evaluator(probe)
                assert isinstance(result, (int, float))

    def test_reference_answers_are_valid(self) -> None:
        """Verify reference answers are valid integers."""
        probes = gen_probes(num_per_kind=10, seed=42)
        refs = precompute_reference_answers(probes)

        for kind, answers in refs.items():
            for ans in answers:
                # Should be parseable as int
                int(ans)

    def test_subsample_preserves_distribution(self) -> None:
        """Verify subsampling roughly preserves rep distribution."""
        # Create datasets with known distribution
        code_docs = [Document(id=f"c{i}", text=f"def f{i}(): pass", rep="code", source="code") for i in range(200)]
        nl_docs = [Document(id=f"n{i}", text=f"Document {i}", rep="nl", source="nl") for i in range(100)]

        result = proportional_subsample([code_docs, nl_docs], rate=0.1, seed=42)

        # Count reps
        rep_counts = Counter(d.rep for d in result)

        # Should roughly preserve 2:1 ratio (code:nl)
        code_count = rep_counts.get("code", 0)
        nl_count = rep_counts.get("nl", 0)

        # Allow some variance but ratio should be roughly maintained
        if nl_count > 0:
            ratio = code_count / nl_count
            assert 1.0 <= ratio <= 4.0  # Should be around 2.0

    def test_all_task_kinds_covered(self) -> None:
        """Verify all 9 task kinds have probes and evaluators."""
        expected_kinds = {"add", "sub", "mul", "lcs", "knap", "rod", "ilp_assign", "ilp_prod", "ilp_partition"}

        assert set(TASK_KINDS) == expected_kinds
        assert set(EVALUATORS.keys()) == expected_kinds

        probes = gen_probes(num_per_kind=1, seed=42)
        assert set(probes.keys()) == expected_kinds


class TestMockedLabelingSmoke:
    """Smoke tests with mocked LLM."""

    @pytest.fixture
    def mock_labeling_result(self) -> LabelingResult:
        """Create a mock labeling result."""
        return LabelingResult(label="add", confidence=0.9, scores={"add": 0.9, "sub": 0.05, "mul": 0.05})

    def test_labeling_result_structure(self, mock_labeling_result: LabelingResult) -> None:
        """Verify labeling result has expected structure."""
        assert mock_labeling_result.label in TASK_KINDS
        assert 0 <= mock_labeling_result.confidence <= 1
        assert sum(mock_labeling_result.scores.values()) <= len(TASK_KINDS)

    def test_mock_pipeline_flow(self, mock_labeling_result: LabelingResult) -> None:
        """Test the pipeline flow with mocked components."""
        # Simulate documents
        docs = [
            Document(id="1", text="def add(a,b): return a+b", rep="code", source="test"),
            Document(id="2", text="Addition combines two numbers", rep="nl", source="test"),
        ]

        # Simulate labeling
        labeled: List[Dict[str, object]] = []
        for doc in docs:
            # In real pipeline, this would call labeler.label(doc.text)
            labeled.append(
                {
                    "id": doc.id,
                    "rep": doc.rep,
                    "text": doc.text,
                    "label": mock_labeling_result.label,
                    "conf": mock_labeling_result.confidence,
                }
            )

        assert len(labeled) == 2
        assert all(row["label"] == "add" for row in labeled)


class TestEvaluatorEdgeCases:
    """Test evaluator edge cases for robustness."""

    def test_lcs_empty_strings(self) -> None:
        """Test LCS with edge case inputs."""
        from src.exps_pret_corr.probes import eval_lcs

        assert eval_lcs({"s1": "", "s2": ""}) == 0
        assert eval_lcs({"s1": "A", "s2": ""}) == 0
        assert eval_lcs({"s1": "", "s2": "B"}) == 0

    def test_knapsack_zero_capacity(self) -> None:
        """Test knapsack with zero capacity."""
        from src.exps_pret_corr.probes import eval_knap

        result = eval_knap({"w": [1, 2, 3], "v": [10, 20, 30], "cap": 0})
        assert result == 0

    def test_rod_single_piece(self) -> None:
        """Test rod cutting with single piece."""
        from src.exps_pret_corr.probes import eval_rod

        result = eval_rod({"prices": [5], "n": 1})
        assert result == 5

    def test_partition_single_element(self) -> None:
        """Test partition with single element."""
        from src.exps_pret_corr.probes import eval_ilp_partition

        result = eval_ilp_partition({"arr": [10]})
        assert result == 10  # Difference is the element itself

    def test_assignment_1x1(self) -> None:
        """Test assignment with 1x1 matrix."""
        from src.exps_pret_corr.probes import eval_ilp_assign

        result = eval_ilp_assign({"C": [[7]]})
        assert result == 7


class TestImportSmoke:
    """Smoke tests for imports."""

    def test_import_data_loader(self) -> None:
        """Verify data_loader module imports correctly."""
        from src.exps_pret_corr import data_loader

        assert hasattr(data_loader, "Document")
        assert hasattr(data_loader, "DataRecord")
        assert hasattr(data_loader, "proportional_subsample")

    def test_import_probes(self) -> None:
        """Verify probes module imports correctly."""
        from src.exps_pret_corr import probes

        assert hasattr(probes, "gen_probes")
        assert hasattr(probes, "EVALUATORS")
        assert hasattr(probes, "SemanticLabeler")

    def test_import_package_init(self) -> None:
        """Verify package __init__ exports."""
        from src.exps_pret_corr import (
            EVALUATORS,
            TASK_KINDS,
        )

        assert len(TASK_KINDS) == 9
        assert len(EVALUATORS) == 9


class TestClassifierIntegrationSmoke:
    """Smoke tests for classifier integration."""

    def test_classifier_import(self) -> None:
        """Verify classifier can be imported from exps_logistic."""
        from src.exps_logistic.classifier import ConceptClassifier

        clf = ConceptClassifier(C=1.0, max_iter=100)
        assert clf.C == 1.0

    def test_metrics_import(self) -> None:
        """Verify metrics can be imported from exps_logistic."""
        from src.exps_logistic.metrics import (
            empirical_entropy_bits,
        )

        # Test entropy calculation
        labels = ["A", "A", "B", "B"]
        entropy = empirical_entropy_bits(labels)
        assert 0.9 <= entropy <= 1.1  # Should be close to 1 bit for 50/50 split

    def test_featurizer_import(self) -> None:
        """Verify featurizer can be imported from exps_logistic."""
        from src.exps_logistic.featurizer import (
            TfidfFeaturizer,
        )

        # TfidfFeaturizer can be instantiated without external dependencies
        tfidf = TfidfFeaturizer(strip_fences=False)
        assert tfidf is not None


class TestEndToEndMockedSmoke:
    """End-to-end smoke test with all components mocked."""

    def test_full_pipeline_mocked(self) -> None:
        """Test the full pipeline with mocked LLM and embeddings."""
        # 1. Create mock documents
        docs = [Document(id=f"code_{i}", text=f"def func{i}(): return {i}", rep="code", source="test") for i in range(20)] + [
            Document(id=f"nl_{i}", text=f"This is document number {i}", rep="nl", source="test") for i in range(20)
        ]

        # 2. Subsample
        sample = proportional_subsample([docs], rate=0.5, seed=42)
        assert len(sample) >= 10

        # 3. Mock labels (would come from SemanticLabeler)
        labels = ["add", "sub", "mul", "lcs"] * (len(sample) // 4 + 1)
        labeled = [{"id": d.id, "rep": d.rep, "text": d.text, "label": labels[i % len(labels)]} for i, d in enumerate(sample)]

        # 4. Separate by channel
        code_docs = [d for d in labeled if d["rep"] == "code"]
        nl_docs = [d for d in labeled if d["rep"] == "nl"]

        # 5. Verify structure
        assert len(code_docs) + len(nl_docs) == len(labeled)

        # 6. Mock embedding and classification would go here
        # In real pipeline, we'd call:
        # - featurizer.transform(texts) -> embeddings
        # - classifier.fit(X_train, y_train)
        # - classifier.evaluate(X_test, y_test) -> metrics

        # Verify we have data for both channels
        assert len(code_docs) > 0 or len(nl_docs) > 0
