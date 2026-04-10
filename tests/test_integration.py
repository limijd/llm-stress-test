"""端到端集成测试：使用 mock HTTP 验证完整流程"""
import json
import pytest
import yaml
from pathlib import Path
from click.testing import CliRunner
from llm_stress_test.cli import main

def _make_sse_response():
    chunks = []
    for token in ["Hello", " ", "world"]:
        chunk = {"choices": [{"delta": {"content": token}, "index": 0, "finish_reason": None}]}
        chunks.append(f"data: {json.dumps(chunk)}\n\n")
    final = {"choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 10, "completion_tokens": 3}}
    chunks.append(f"data: {json.dumps(final)}\n\n")
    chunks.append("data: [DONE]\n\n")
    return "".join(chunks)

class TestIntegration:
    def test_full_run_with_native_engine(self, tmp_path):
        config = {
            "target": {"name": "integration-test",
                       "api_url": "http://test-api/v1/chat/completions",
                       "api_key": "sk-test", "model": "test-model"},
            "engine": "native",
            "request": {"stream": True, "extra_args": {}},
            "test": {"concurrency": [1], "requests_per_level": [2], "dataset": "openqa"},
            "pass_criteria": [{"metric": "success_rate", "operator": ">=", "threshold": 1.0}],
            "degradation": {"enabled": False, "start_concurrency": 1, "step": 1, "min_concurrency": 1},
            "output": {"dir": str(tmp_path / "results"), "formats": ["json", "csv"], "charts": False},
        }
        config_path = tmp_path / "test.yaml"
        config_path.write_text(yaml.dump(config))

        from aioresponses import aioresponses
        sse_body = _make_sse_response()

        with aioresponses() as m:
            m.post("http://test-api/v1/chat/completions", body=sse_body,
                   content_type="text/event-stream", repeat=True)
            runner = CliRunner()
            result = runner.invoke(main, ["run", "--config", str(config_path)])

        # Verify success
        assert result.exit_code == 0, f"CLI failed: {result.output}\n{result.exception}"
        assert "报告已生成" in result.output or "results" in result.output

        # Verify output files
        results_dirs = list((tmp_path / "results").iterdir())
        assert len(results_dirs) == 1
        result_dir = results_dirs[0]
        assert (result_dir / "summary.json").exists()
        assert (result_dir / "summary.csv").exists()
        # CLI 保存的配置快照文件名为 config_snapshot.yaml
        assert (result_dir / "config_snapshot.yaml").exists()

        # Verify secret redaction
        saved_config = yaml.safe_load((result_dir / "config_snapshot.yaml").read_text())
        assert saved_config["target"]["api_key"] == "***REDACTED***"
