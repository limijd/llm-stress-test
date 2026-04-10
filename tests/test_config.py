import os
import pytest
from pathlib import Path
from llm_stress_test import _yaml as yaml
from llm_stress_test.config import (
    load_config, validate_config, merge_cli_overrides,
    expand_env_vars, sanitize_for_export, ConfigError,
)

MINIMAL_CONFIG = {
    "target": {
        "name": "test",
        "api_url": "http://localhost:8000/v1/chat/completions",
        "api_key": "sk-test",
        "model": "test-model",
    },
    "engine": "native",
    "request": {"stream": True, "extra_args": {}},
    "test": {
        "concurrency": [1, 5],
        "requests_per_level": [10, 50],
        "dataset": "openqa",
    },
    "pass_criteria": [
        {"metric": "success_rate", "operator": ">=", "threshold": 1.0},
    ],
    "degradation": {
        "enabled": False, "start_concurrency": 5,
        "step": 1, "min_concurrency": 1,
    },
    "output": {"dir": "./results", "formats": ["json"], "charts": False},
}

def write_yaml(data: dict, path: Path):
    path.write_text(yaml.dump(data, allow_unicode=True, sort_keys=False))

class TestLoadConfig:
    def test_load_valid_yaml(self, tmp_path):
        p = tmp_path / "config.yaml"
        write_yaml(MINIMAL_CONFIG, p)
        cfg = load_config(str(p))
        assert cfg["target"]["name"] == "test"

    def test_load_nonexistent_file(self):
        with pytest.raises(ConfigError, match="配置文件不存在"):
            load_config("/nonexistent/path.yaml")

    def test_load_invalid_yaml(self, tmp_path):
        p = tmp_path / "bad.yaml"
        # 顶层为列表而非字典，应报错
        p.write_text("- not a dict")
        with pytest.raises(ConfigError, match="顶层必须是字典"):
            load_config(str(p))

class TestValidateConfig:
    def test_valid_config_passes(self):
        validate_config(MINIMAL_CONFIG)

    def test_missing_target(self):
        cfg = {k: v for k, v in MINIMAL_CONFIG.items() if k != "target"}
        with pytest.raises(ConfigError, match="target"):
            validate_config(cfg)

    def test_mismatched_array_lengths(self):
        cfg = {**MINIMAL_CONFIG, "test": {**MINIMAL_CONFIG["test"], "requests_per_level": [10]}}
        with pytest.raises(ConfigError, match="长度必须一致"):
            validate_config(cfg)

    def test_invalid_engine(self):
        cfg = {**MINIMAL_CONFIG, "engine": "unknown"}
        with pytest.raises(ConfigError, match="engine"):
            validate_config(cfg)

    def test_invalid_operator(self):
        cfg = {**MINIMAL_CONFIG, "pass_criteria": [
            {"metric": "success_rate", "operator": "!=", "threshold": 1.0},
        ]}
        with pytest.raises(ConfigError, match="操作符"):
            validate_config(cfg)

class TestExpandEnvVars:
    def test_expand_env_var(self, monkeypatch):
        monkeypatch.setenv("TEST_KEY", "my-secret-key")
        cfg = {"target": {"api_key": "${TEST_KEY}"}}
        result = expand_env_vars(cfg)
        assert result["target"]["api_key"] == "my-secret-key"

    def test_literal_value_unchanged(self):
        cfg = {"target": {"api_key": "sk-literal"}}
        result = expand_env_vars(cfg)
        assert result["target"]["api_key"] == "sk-literal"

    def test_missing_env_var_raises(self):
        cfg = {"target": {"api_key": "${NONEXISTENT_VAR_12345}"}}
        with pytest.raises(ConfigError, match="环境变量.*未设置"):
            expand_env_vars(cfg)

class TestMergeCLIOverrides:
    def test_override_engine(self):
        cfg = {**MINIMAL_CONFIG}
        overrides = {"engine": "evalscope"}
        result = merge_cli_overrides(cfg, overrides)
        assert result["engine"] == "evalscope"

    def test_override_concurrency(self):
        cfg = {**MINIMAL_CONFIG}
        overrides = {"test.concurrency": [1, 10]}
        result = merge_cli_overrides(cfg, overrides)
        assert result["test"]["concurrency"] == [1, 10]

class TestSanitizeForExport:
    def test_redact_literal_key(self):
        cfg = {**MINIMAL_CONFIG}
        result = sanitize_for_export(cfg)
        assert result["target"]["api_key"] == "***REDACTED***"

    def test_preserve_env_var_placeholder(self):
        cfg = {**MINIMAL_CONFIG, "target": {**MINIMAL_CONFIG["target"], "api_key": "${LLM_API_KEY}"}}
        result = sanitize_for_export(cfg)
        assert result["target"]["api_key"] == "${LLM_API_KEY}"
