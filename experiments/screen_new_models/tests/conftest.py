"""Pytest configuration for screen_new_models tests."""

import sys
from pathlib import Path

import pytest

# Ensure repo root is in path
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")


@pytest.fixture
def repo_root() -> Path:
    """Get the repository root path."""
    return REPO_ROOT
