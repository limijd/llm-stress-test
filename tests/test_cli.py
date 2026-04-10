import pytest
import yaml
from click.testing import CliRunner
from llm_stress_test.cli import main

@pytest.fixture
def runner():
    return CliRunner()

class TestCLI:
    def test_help(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.output
        assert "validate" in result.output
        assert "report" in result.output

    def test_validate_valid_config(self, runner, tmp_path):
        config = {
            "target": {"name": "test", "api_url": "http://localhost/v1/chat/completions", "api_key": "sk-test", "model": "test"},
            "engine": "native",
            "test": {"concurrency": [1], "requests_per_level": [10], "dataset": "openqa"},
            "pass_criteria": [{"metric": "success_rate", "operator": ">=", "threshold": 1.0}],
            "degradation": {"enabled": False, "start_concurrency": 1, "step": 1, "min_concurrency": 1},
            "output": {"dir": "./results", "formats": ["json"], "charts": False},
        }
        config_path = tmp_path / "test.yaml"
        config_path.write_text(yaml.dump(config))
        result = runner.invoke(main, ["validate", "--config", str(config_path)])
        assert result.exit_code == 0

    def test_validate_invalid_config(self, runner, tmp_path):
        config_path = tmp_path / "bad.yaml"
        config_path.write_text("target: {}")
        result = runner.invoke(main, ["validate", "--config", str(config_path)])
        assert result.exit_code != 0

    def test_run_missing_config(self, runner):
        result = runner.invoke(main, ["run"])
        assert result.exit_code != 0
