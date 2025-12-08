#!/usr/bin/env python3
"""Unit tests for data_loader module."""

from pathlib import Path
from typing import List

import pytest

from src.exps_pret_corr.data_loader import (
    DataRecord,
    Document,
    get_default_data_records,
    looks_like_code,
    proportional_subsample,
)


class TestLooksLikeCode:
    """Tests for code detection heuristic."""

    def test_python_def(self) -> None:
        assert looks_like_code("def foo():\n    return 1")

    def test_python_class(self) -> None:
        assert looks_like_code("class MyClass:\n    pass")

    def test_curly_braces(self) -> None:
        assert looks_like_code("function foo() { return 1; }")

    def test_return_keyword(self) -> None:
        assert looks_like_code("return x + y")

    def test_for_loop(self) -> None:
        assert looks_like_code("for i in range(10):")

    def test_while_loop(self) -> None:
        assert looks_like_code("while True:")

    def test_plain_text(self) -> None:
        assert not looks_like_code("This is just plain text about programming.")

    def test_empty_string(self) -> None:
        assert not looks_like_code("")

    def test_semicolon(self) -> None:
        assert looks_like_code("int x = 5;")


class TestDataRecord:
    """Tests for DataRecord dataclass."""

    def test_creation(self) -> None:
        record = DataRecord(
            name="test",
            path=Path("/tmp/test"),
            file_type="parquet",
            column_name="text",
            rep="code",
        )

        assert record.name == "test"
        assert record.file_type == "parquet"
        assert record.rep == "code"

    def test_all_fields(self) -> None:
        record = DataRecord(
            name="my_dataset",
            path=Path("/data/my_dataset"),
            file_type="json",
            column_name="content",
            rep="nl",
        )

        assert record.name == "my_dataset"
        assert record.path == Path("/data/my_dataset")
        assert record.file_type == "json"
        assert record.column_name == "content"
        assert record.rep == "nl"


class TestDocument:
    """Tests for Document dataclass."""

    def test_creation(self) -> None:
        doc = Document(
            id="doc_1",
            text="Hello world",
            rep="nl",
            source="test_dataset",
        )

        assert doc.id == "doc_1"
        assert doc.text == "Hello world"
        assert doc.rep == "nl"
        assert doc.source == "test_dataset"

    def test_code_document(self) -> None:
        doc = Document(
            id="code_123",
            text="def add(a, b): return a + b",
            rep="code",
            source="starcoder",
        )

        assert doc.rep == "code"
        assert "def " in doc.text


class TestDefaultDataRecords:
    """Tests for default data record configuration."""

    def test_returns_list(self) -> None:
        records = get_default_data_records()
        assert isinstance(records, list)
        assert len(records) >= 2

    def test_contains_starcoder(self) -> None:
        records = get_default_data_records()
        names = [r.name for r in records]
        assert "starcoder" in names

    def test_contains_dclm(self) -> None:
        records = get_default_data_records()
        names = [r.name for r in records]
        assert "dclm" in names

    def test_starcoder_is_code(self) -> None:
        records = get_default_data_records()
        starcoder = next(r for r in records if r.name == "starcoder")
        assert starcoder.rep == "code"

    def test_dclm_is_nl(self) -> None:
        records = get_default_data_records()
        dclm = next(r for r in records if r.name == "dclm")
        assert dclm.rep == "nl"

    def test_paths_are_absolute_or_relative(self) -> None:
        records = get_default_data_records()
        for r in records:
            assert isinstance(r.path, Path)


class TestProportionalSubsample:
    """Tests for proportional subsampling."""

    @pytest.fixture
    def sample_datasets(self) -> List[List[Document]]:
        """Create sample datasets for testing."""
        # Dataset 1: 100 code samples
        ds1 = [Document(id=f"code_{i}", text=f"def func{i}(): pass", rep="code", source="ds1") for i in range(100)]
        # Dataset 2: 200 NL samples
        ds2 = [Document(id=f"nl_{i}", text=f"This is document {i}", rep="nl", source="ds2") for i in range(200)]
        return [ds1, ds2]

    def test_subsample_rate(self, sample_datasets: List[List[Document]]) -> None:
        """Verify approximate subsample rate."""
        result = proportional_subsample(sample_datasets, rate=0.1, seed=42)

        # Should get roughly 10% of 300 = 30 samples
        assert 20 <= len(result) <= 40

    def test_deterministic(self, sample_datasets: List[List[Document]]) -> None:
        """Verify subsampling is deterministic with same seed."""
        result1 = proportional_subsample(sample_datasets, rate=0.1, seed=42)
        result2 = proportional_subsample(sample_datasets, rate=0.1, seed=42)

        assert len(result1) == len(result2)
        ids1 = set(d.id for d in result1)
        ids2 = set(d.id for d in result2)
        assert ids1 == ids2

    def test_different_seeds(self, sample_datasets: List[List[Document]]) -> None:
        """Verify different seeds produce different samples."""
        result1 = proportional_subsample(sample_datasets, rate=0.1, seed=1)
        result2 = proportional_subsample(sample_datasets, rate=0.1, seed=2)

        ids1 = set(d.id for d in result1)
        ids2 = set(d.id for d in result2)
        # Should have some overlap but not be identical
        assert ids1 != ids2

    def test_preserves_rep(self, sample_datasets: List[List[Document]]) -> None:
        """Verify both rep types are preserved."""
        result = proportional_subsample(sample_datasets, rate=0.1, seed=42)

        reps = set(d.rep for d in result)
        assert "code" in reps
        assert "nl" in reps

    def test_empty_input(self) -> None:
        """Handle empty input gracefully."""
        result = proportional_subsample([], rate=0.1, seed=42)
        assert result == []

    def test_empty_dataset(self) -> None:
        """Handle empty datasets in input."""
        ds1 = [Document(id="1", text="hello", rep="nl", source="ds1")]
        result = proportional_subsample([ds1, []], rate=0.5, seed=42)
        assert len(result) >= 1

    def test_high_rate(self, sample_datasets: List[List[Document]]) -> None:
        """Test with high subsample rate."""
        result = proportional_subsample(sample_datasets, rate=0.5, seed=42)

        # Should get roughly 50% of 300 = 150 samples
        assert 100 <= len(result) <= 200

    def test_minimum_one_per_dataset(self, sample_datasets: List[List[Document]]) -> None:
        """Verify at least one sample from each non-empty dataset."""
        result = proportional_subsample(sample_datasets, rate=0.001, seed=42)

        sources = set(d.source for d in result)
        assert len(sources) == 2  # Both datasets represented

    def test_single_dataset(self) -> None:
        """Test with a single dataset."""
        ds = [Document(id=str(i), text=f"text {i}", rep="nl", source="single") for i in range(50)]
        result = proportional_subsample([ds], rate=0.2, seed=42)

        assert 5 <= len(result) <= 15  # ~10 samples
        assert all(d.source == "single" for d in result)
