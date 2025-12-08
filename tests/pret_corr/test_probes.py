#!/usr/bin/env python3
"""Unit tests for probes module."""

from src.exps_pret_corr.probes import (
    EVALUATORS,
    TASK_KINDS,
    LabelingResult,
    SemanticLabeler,
    create_apply_prompt,
    eval_add,
    eval_ilp_assign,
    eval_ilp_partition,
    eval_ilp_prod,
    eval_knap,
    eval_lcs,
    eval_mul,
    eval_rod,
    eval_sub,
    gen_probes,
    normalize_answer,
    precompute_reference_answers,
    serialize_probe,
)


class TestProbeGeneration:
    """Tests for probe generation."""

    def test_gen_probes_returns_all_kinds(self) -> None:
        """Verify all task kinds are present."""
        probes = gen_probes(num_per_kind=5, seed=42)

        for kind in TASK_KINDS:
            assert kind in probes
            assert len(probes[kind]) == 5

    def test_gen_probes_deterministic(self) -> None:
        """Verify probe generation is deterministic with same seed."""
        probes1 = gen_probes(num_per_kind=3, seed=123)
        probes2 = gen_probes(num_per_kind=3, seed=123)

        for kind in TASK_KINDS:
            assert probes1[kind] == probes2[kind]

    def test_gen_probes_different_seeds(self) -> None:
        """Verify different seeds produce different probes."""
        probes1 = gen_probes(num_per_kind=3, seed=1)
        probes2 = gen_probes(num_per_kind=3, seed=2)

        # At least one kind should have different probes
        any_different = any(probes1[k] != probes2[k] for k in TASK_KINDS)
        assert any_different

    def test_arithmetic_probe_structure(self) -> None:
        """Verify arithmetic probes have correct structure."""
        probes = gen_probes(num_per_kind=1, seed=42)

        for kind in ["add", "sub", "mul"]:
            p = probes[kind][0]
            assert "a" in p and "b" in p
            assert isinstance(p["a"], int) and isinstance(p["b"], int)
            assert 100 <= p["a"] <= 999  # 3-digit numbers

    def test_sub_probe_order(self) -> None:
        """Verify subtraction probes have a >= b."""
        probes = gen_probes(num_per_kind=10, seed=42)

        for p in probes["sub"]:
            assert p["a"] >= p["b"]

    def test_lcs_probe_structure(self) -> None:
        """Verify LCS probes have string fields."""
        probes = gen_probes(num_per_kind=1, seed=42)
        p = probes["lcs"][0]
        assert "s1" in p and "s2" in p
        assert isinstance(p["s1"], str) and isinstance(p["s2"], str)
        assert 6 <= len(p["s1"]) <= 10

    def test_knap_probe_structure(self) -> None:
        """Verify knapsack probes have correct structure."""
        probes = gen_probes(num_per_kind=1, seed=42)
        p = probes["knap"][0]
        assert "w" in p and "v" in p and "cap" in p
        assert len(p["w"]) == len(p["v"])
        assert p["cap"] > 0


class TestEvaluators:
    """Tests for ground-truth evaluators."""

    def test_eval_add(self) -> None:
        assert eval_add({"a": 100, "b": 200}) == 300
        assert eval_add({"a": 999, "b": 1}) == 1000

    def test_eval_sub(self) -> None:
        assert eval_sub({"a": 500, "b": 200}) == 300
        assert eval_sub({"a": 100, "b": 100}) == 0

    def test_eval_mul(self) -> None:
        assert eval_mul({"a": 100, "b": 10}) == 1000
        assert eval_mul({"a": 123, "b": 456}) == 56088

    def test_eval_lcs(self) -> None:
        # Same strings
        assert eval_lcs({"s1": "ABC", "s2": "ABC"}) == 3
        # No common
        assert eval_lcs({"s1": "ABC", "s2": "DEF"}) == 0
        # Classic example
        assert eval_lcs({"s1": "ABCBDAB", "s2": "BDCAB"}) == 4

    def test_eval_knap(self) -> None:
        # Simple case: all items fit
        result = eval_knap({"w": [1, 2, 3], "v": [10, 20, 30], "cap": 6})
        assert result == 60

        # Can't fit all items
        result = eval_knap({"w": [2, 3, 4], "v": [3, 4, 5], "cap": 5})
        assert result == 7  # items 0 and 1

    def test_eval_rod(self) -> None:
        # Length 4 with prices [1, 5, 8, 9]
        # Optimal: cut into 2+2 = 5+5 = 10
        result = eval_rod({"prices": [1, 5, 8, 9], "n": 4})
        assert result == 10

    def test_eval_ilp_assign(self) -> None:
        # 2x2 assignment
        C = [[1, 2], [3, 1]]
        result = eval_ilp_assign({"C": C})
        assert result == 2  # (0,0) + (1,1) = 1+1 = 2

    def test_eval_ilp_prod(self) -> None:
        # Simple case: a=[2, 3], b=[10, 9], p=[5, 4]
        # x0 = 10//2 = 5, x1 = 9//3 = 3
        # profit = 5*5 + 4*3 = 25 + 12 = 37
        result = eval_ilp_prod({"a": [2, 3], "b": [10, 9], "p": [5, 4]})
        assert result == 37

    def test_eval_ilp_partition(self) -> None:
        # [1, 2, 3, 4] -> sum=10, optimal: {1,4} vs {2,3} -> diff=0
        result = eval_ilp_partition({"arr": [1, 2, 3, 4]})
        assert result == 0

        # [1, 2, 3, 4, 5] -> sum=15, optimal: {1,2,5}=8 vs {3,4}=7 -> diff=1
        result = eval_ilp_partition({"arr": [1, 2, 3, 4, 5]})
        assert result == 1

    def test_all_evaluators_registered(self) -> None:
        """Verify all task kinds have evaluators."""
        for kind in TASK_KINDS:
            assert kind in EVALUATORS


class TestReferenceAnswers:
    """Tests for reference answer computation."""

    def test_precompute_reference_answers(self) -> None:
        probes = gen_probes(num_per_kind=3, seed=42)
        refs = precompute_reference_answers(probes)

        for kind in TASK_KINDS:
            assert kind in refs
            assert len(refs[kind]) == 3
            # All answers should be string representations of integers
            for ans in refs[kind]:
                assert isinstance(ans, str)
                int(ans)  # Should not raise


class TestPromptCreation:
    """Tests for prompt creation utilities."""

    def test_serialize_probe_simple(self) -> None:
        result = serialize_probe({"a": 100, "b": 200})
        assert "a=100" in result
        assert "b=200" in result

    def test_serialize_probe_list(self) -> None:
        result = serialize_probe({"w": [1, 2, 3]})
        assert "w=[1,2,3]" in result

    def test_serialize_probe_2d_list(self) -> None:
        result = serialize_probe({"C": [[1, 2], [3, 4]]})
        assert "C=" in result
        assert "[1,2]" in result

    def test_create_apply_prompt(self) -> None:
        prompt = create_apply_prompt("Add two numbers", {"a": 1, "b": 2})

        assert "Apply the following procedure" in prompt
        assert "Add two numbers" in prompt
        assert "a=1" in prompt
        assert "ANSWER:" in prompt


class TestNormalizeAnswer:
    """Tests for answer normalization."""

    def test_simple_number(self) -> None:
        assert normalize_answer("42") == "42"
        assert normalize_answer("  42  ") == "42"

    def test_with_prefix(self) -> None:
        assert normalize_answer("Answer: 42") == "42"
        assert normalize_answer("Final: 100") == "100"
        assert normalize_answer("Result = 55") == "55"

    def test_multiline(self) -> None:
        assert normalize_answer("Some text\nMore text\n42") == "42"

    def test_negative(self) -> None:
        assert normalize_answer("-5") == "-5"

    def test_float(self) -> None:
        assert normalize_answer("3.14") == "3.14"

    def test_no_number(self) -> None:
        assert normalize_answer("no number here") == "⊥"

    def test_scientific_notation(self) -> None:
        assert normalize_answer("1e6") == "1e6"


class TestLabelingResult:
    """Tests for LabelingResult dataclass."""

    def test_creation(self) -> None:
        result = LabelingResult(label="add", confidence=0.8, scores={"add": 0.8, "sub": 0.1, "mul": 0.1})

        assert result.label == "add"
        assert result.confidence == 0.8
        assert len(result.scores) == 3

    def test_none_label(self) -> None:
        result = LabelingResult(label=None, confidence=0.3, scores={})
        assert result.label is None


class TestSemanticLabeler:
    """Tests for SemanticLabeler class."""

    def test_init(self) -> None:
        """Verify labeler can be initialized without loading model."""
        labeler = SemanticLabeler(
            model_name="gpt2",  # Small model for testing
            probes_per_kind=2,
            conf_threshold=0.5,
        )
        # Model should not be loaded yet (lazy loading)
        assert labeler._tok is None
        assert labeler._model is None

    def test_probes_generated_on_access(self) -> None:
        """Verify probes are generated when accessed."""
        labeler = SemanticLabeler(probes_per_kind=3, seed=42)
        # Force probe generation by accessing internal state
        ensure_loaded = labeler._ensure_loaded.__wrapped__  # type: ignore[attr-defined]
        ensure_loaded(labeler)  # Bypass lazy loading partially

        # Probes should now be set
        assert labeler._probes is not None
        assert len(labeler._probes["add"]) == 3
