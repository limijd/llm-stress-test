import json
import pytest
from unittest.mock import patch, MagicMock
from llm_stress_test.engine.evalscope import EvalScopeEngine
from llm_stress_test.models import EngineConfig

def _make_config():
    return EngineConfig(
        api_url="http://test-api/v1/chat/completions", api_key="test-key",
        model="test-model", concurrency=5, num_requests=50,
        dataset="openqa", stream=True,
        extra_args={"chat_template_kwargs": {"thinking": True}},
    )

class TestEvalScopeEngine:
    def test_check_available_installed(self):
        engine = EvalScopeEngine()
        with patch("shutil.which", return_value="/usr/bin/evalscope"):
            ok, msg = engine.check_available()
            assert ok is True

    def test_check_available_not_installed(self):
        engine = EvalScopeEngine()
        with patch("shutil.which", return_value=None):
            with patch("subprocess.run", side_effect=FileNotFoundError):
                ok, msg = engine.check_available()
                assert ok is False
                assert "evalscope" in msg

    def test_build_command(self):
        engine = EvalScopeEngine()
        config = _make_config()
        cmd = engine._build_command(config)
        assert "evalscope" in cmd[0]
        assert "perf" in cmd
        assert "--parallel" in cmd
        assert "5" in cmd
        assert "--number" in cmd
        assert "50" in cmd
        assert "--url" in cmd
        assert "--model" in cmd
        assert "--stream" in cmd

    def test_run_process_crash(self):
        engine = EvalScopeEngine()
        config = _make_config()
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.stdout = ""
        mock_process.stderr = "Segmentation fault"
        with patch("subprocess.run", return_value=mock_process):
            result = engine.run(config)
        assert all(not r.success for r in result.requests)
