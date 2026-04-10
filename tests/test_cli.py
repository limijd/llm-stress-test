import subprocess
import sys
import pytest
from pathlib import Path
from llm_stress_test import _yaml as yaml
from llm_stress_test.cli import main


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    """通过 subprocess 调用 CLI，避免依赖 click.testing.CliRunner。"""
    return subprocess.run(
        [sys.executable, "-m", "llm_stress_test", *args],
        capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(Path(__file__).parent.parent / "src")},
    )


class TestCLI:
    def test_help(self):
        result = _run_cli("--help")
        assert result.returncode == 0
        assert "run" in result.stdout
        assert "validate" in result.stdout
        assert "report" in result.stdout

    def test_validate_valid_config(self, tmp_path):
        config = {
            "target": {"name": "test", "api_url": "http://localhost/v1/chat/completions", "api_key": "sk-test", "model": "test"},
            "engine": "native",
            "test": {"concurrency": [1], "requests_per_level": [10], "dataset": "openqa"},
            "pass_criteria": [{"metric": "success_rate", "operator": ">=", "threshold": 1.0}],
            "degradation": {"enabled": False, "start_concurrency": 1, "step": 1, "min_concurrency": 1},
            "output": {"dir": "./results", "formats": ["json"], "charts": False},
        }
        config_path = tmp_path / "test.yaml"
        config_path.write_text(yaml.dump(config, sort_keys=False))
        result = _run_cli("validate", "--config", str(config_path))
        assert result.returncode == 0

    def test_validate_invalid_config(self, tmp_path):
        config_path = tmp_path / "bad.yaml"
        config_path.write_text("target: {}")
        result = _run_cli("validate", "--config", str(config_path))
        assert result.returncode != 0

    def test_run_missing_config(self):
        result = _run_cli("run")
        assert result.returncode != 0
