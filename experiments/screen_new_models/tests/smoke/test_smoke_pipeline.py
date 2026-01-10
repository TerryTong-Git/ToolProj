#!/usr/bin/env python3
"""
Smoke tests for the model screening pipeline.

These tests verify basic functionality:
1. Can we import required modules?
2. Can we access the LLM API (OpenRouter)?
3. Can we generate a simple dataset?
4. Can we parse results from a JSONL file?
5. Can the MI analysis pipeline load data?

Run with: uv run pytest experiments/screen_new_models/tests/smoke/ -v
"""

import json
import os
import sys
from pathlib import Path

import pytest


def find_repo_root() -> Path:
    """Find the repository root by looking for pyproject.toml."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists() and (parent / "src").exists():
            return parent
    raise RuntimeError("Could not find repository root")


# Ensure repo root is in path
REPO_ROOT = find_repo_root()
sys.path.insert(0, str(REPO_ROOT))

# Also check the main ToolProj directory for .env (worktree shares src but not .env)
MAIN_REPO = Path("/Users/terrytong/Documents/CCG/ToolProj")


class TestImports:
    """Test that all required modules can be imported."""

    def test_import_performance_main(self) -> None:
        """Can import the main performance experiment module."""
        from src.exps_performance import main
        assert hasattr(main, "run")
        assert hasattr(main, "Args")

    def test_import_dataset(self) -> None:
        """Can import the dataset module."""
        from src.exps_performance import dataset
        assert hasattr(dataset, "make_dataset")

    def test_import_arms(self) -> None:
        """Can import the arms module."""
        from src.exps_performance.arms import Arm1, Arm2, Arm3, Arm4
        assert Arm1 is not None
        assert Arm2 is not None
        assert Arm3 is not None
        assert Arm4 is not None

    def test_import_llm(self) -> None:
        """Can import the LLM module."""
        from src.exps_performance import llm
        assert hasattr(llm, "llm")

    def test_import_logistic_main(self) -> None:
        """Can import the logistic MI analysis module."""
        from src.exps_logistic import main
        assert hasattr(main, "run")

    def test_import_logistic_data_utils(self) -> None:
        """Can import the logistic data utilities."""
        from src.exps_logistic import data_utils
        assert hasattr(data_utils, "load_data")
        assert hasattr(data_utils, "filter_by_rep")
        assert hasattr(data_utils, "prepare_labels")


class TestEnvironment:
    """Test that required environment variables and files exist."""

    def test_env_file_exists(self) -> None:
        """Check that .env file exists (in main repo or worktree)."""
        env_path = REPO_ROOT / ".env"
        main_env_path = MAIN_REPO / ".env"
        assert env_path.exists() or main_env_path.exists(), (
            f"Missing .env file at {env_path} or {main_env_path}"
        )

    def test_openrouter_api_key(self) -> None:
        """Check that OPENROUTER_API_KEY is set."""
        # Load from .env if not already in environment
        # Check both worktree and main repo locations
        for env_path in [REPO_ROOT / ".env", MAIN_REPO / ".env"]:
            if env_path.exists():
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        # Handle "export VAR=val" or "VAR=val" formats
                        if "OPENROUTER_API_KEY=" in line:
                            # Remove "export " prefix if present
                            if line.startswith("export "):
                                line = line[7:]
                            key = line.split("=", 1)[1].strip('"\'')
                            os.environ.setdefault("OPENROUTER_API_KEY", key)
                            break

        api_key = os.environ.get("OPENROUTER_API_KEY")
        assert api_key is not None, "OPENROUTER_API_KEY not set"
        assert len(api_key) > 10, "OPENROUTER_API_KEY appears invalid"

    def test_screening_script_exists(self) -> None:
        """Check that the screening script exists."""
        # Check both worktree and main repo
        script_path = REPO_ROOT / "src" / "exps_performance" / "scripts" / "screen_new_models.sh"
        main_script_path = MAIN_REPO / "src" / "exps_performance" / "scripts" / "screen_new_models.sh"
        assert script_path.exists() or main_script_path.exists(), (
            f"Missing screening script at {script_path} or {main_script_path}"
        )


class TestDataGeneration:
    """Test that we can generate simple test datasets."""

    def test_make_simple_dataset(self) -> None:
        """Can generate a small dataset with basic kinds."""
        from src.exps_performance.dataset import make_dataset

        kinds = ["add", "sub"]
        n_samples = 2
        digits_list = [2]

        data = make_dataset(kinds, n_samples, digits_list)
        assert len(data) > 0, "Dataset should not be empty"

        # Verify each item has required attributes
        for item in data:
            assert hasattr(item, "kind")
            assert hasattr(item, "digits")
            assert hasattr(item, "record")
            assert item.kind in kinds
            assert item.digits in digits_list

    def test_make_screening_dataset(self) -> None:
        """Can generate a dataset matching screening script config."""
        from src.exps_performance.dataset import make_dataset

        # Matching screen_new_models.sh config
        kinds = ["add", "sub", "mul", "lcs", "knap", "rod"]
        n_samples = 5
        digits_list = [2, 4, 6]

        data = make_dataset(kinds, n_samples, digits_list)
        assert len(data) > 0, "Dataset should not be empty"

        # Count by kind
        kind_counts: dict[str, int] = {}
        for item in data:
            kind_counts[item.kind] = kind_counts.get(item.kind, 0) + 1

        # Should have samples for each kind
        for kind in kinds:
            assert kind in kind_counts, f"Missing kind: {kind}"


class TestResultParsing:
    """Test that we can parse results from JSONL files."""

    @pytest.fixture
    def sample_result_line(self) -> str:
        """Generate a sample result line matching the expected format."""
        return json.dumps({
            "request_id": "test_request_123",
            "unique_tag": "test_tag_123",
            "index_in_kind": 1,
            "model": "test/model",
            "seed": 0,
            "exp_id": "run_20260110",
            "digit": 4,
            "kind": "add",
            "question": "What is 12 + 34?",
            "answer": "46",
            "nl_question": "Solve: 12 + 34",
            "nl_answer": "46",
            "nl_reasoning": "12 + 34 = 46",
            "nl_correct": True,
            "nl_parse_err": False,
            "nl_err_msg": "",
            "code_question": "Solve: 12 + 34",
            "code_answer": "46",
            "code_correct": True,
            "code_parse_err": False,
            "code_gen_err": "",
            "code_err_msg": "",
            "sim_code": "def solution(): return 12 + 34",
            "sim_question": "Solve: 12 + 34",
            "sim_reasoning": "Adding 12 and 34",
            "sim_answer": "46",
            "sim_correct": True,
            "sim_parse_err": False,
            "sim_err_msg": "ok",
            "controlsim_question": "",
            "controlsim_reasoning": "",
            "controlsim_answer": "",
            "controlsim_correct": False,
            "controlsim_parse_err": False,
            "controlsim_err_msg": "",
        })

    def test_parse_single_result(self, sample_result_line: str) -> None:
        """Can parse a single result line."""
        result = json.loads(sample_result_line)

        # Verify key fields
        assert result["kind"] == "add"
        assert result["digit"] == 4
        assert result["nl_correct"] is True
        assert result["code_correct"] is True
        assert result["sim_correct"] is True
        assert result["model"] == "test/model"
        assert result["seed"] == 0

    def test_parse_result_into_dataframe(self, sample_result_line: str, tmp_path: Path) -> None:
        """Can parse results into a pandas DataFrame."""
        import pandas as pd

        # Write sample results to temp file
        results_file = tmp_path / "res.jsonl"
        with open(results_file, "w") as f:
            f.write(sample_result_line + "\n")
            f.write(sample_result_line.replace('"add"', '"sub"') + "\n")

        # Read as DataFrame
        df = pd.read_json(results_file, lines=True)

        assert len(df) == 2
        assert "kind" in df.columns
        assert "digit" in df.columns
        assert "nl_correct" in df.columns
        assert "code_correct" in df.columns


class TestMIAnalysisDataLoading:
    """Test MI analysis can load and process data."""

    def test_load_existing_results(self) -> None:
        """Can load data from existing results directory if available."""
        import pandas as pd

        # Check both worktree and main repo
        results_dir = REPO_ROOT / "src" / "exps_performance" / "results"
        main_results_dir = MAIN_REPO / "src" / "exps_performance" / "results"

        target_dir = None
        for d in [results_dir, main_results_dir]:
            if d.exists() and list(d.iterdir()):
                target_dir = d
                break

        if target_dir is None:
            pytest.skip("No existing results to test with")

        # Load raw JSONL files directly (without the full load_data which requires rationales)
        all_files = list(target_dir.rglob("res.jsonl"))
        if not all_files:
            pytest.skip("No res.jsonl files found in results directory")

        # Read first file to verify structure
        df = pd.read_json(all_files[0], lines=True)
        assert len(df) > 0, "Should load some data from existing results"

        # Verify required columns exist for performance experiment output
        required_cols = ["kind", "digit", "model"]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"


class TestAPIConnectivity:
    """Test API connectivity (these may be slow or require API keys)."""

    @pytest.mark.slow
    def test_openrouter_connectivity(self) -> None:
        """Test that we can connect to OpenRouter API."""
        import httpx

        # Load API key from both locations
        api_key = None
        for env_path in [REPO_ROOT / ".env", MAIN_REPO / ".env"]:
            if env_path.exists():
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if "OPENROUTER_API_KEY=" in line:
                            # Handle "export VAR=val" format
                            if line.startswith("export "):
                                line = line[7:]
                            api_key = line.split("=", 1)[1].strip('"\'')
                            break
                if api_key:
                    break

        if not api_key:
            api_key = os.environ.get("OPENROUTER_API_KEY")

        if not api_key:
            pytest.skip("OPENROUTER_API_KEY not available")

        # Simple API check - get models list
        response = httpx.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0,
        )

        assert response.status_code == 200, f"API returned {response.status_code}"
        data = response.json()
        assert "data" in data, "Response should contain 'data' field"
        assert len(data["data"]) > 0, "Should have at least one model"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
