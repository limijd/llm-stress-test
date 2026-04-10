"""端到端集成测试：使用 mock HTTP server 验证完整流程"""
import json
import subprocess
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import pytest
from llm_stress_test import _yaml as yaml


def _make_sse_body():
    chunks = []
    for token in ["Hello", " ", "world"]:
        chunk = {"choices": [{"delta": {"content": token}, "index": 0, "finish_reason": None}]}
        chunks.append(f"data: {json.dumps(chunk)}\n\n")
    final = {"choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 10, "completion_tokens": 3}}
    chunks.append(f"data: {json.dumps(final)}\n\n")
    chunks.append("data: [DONE]\n\n")
    return "".join(chunks).encode("utf-8")


class _SSEHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        body = _make_sse_body()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


class TestIntegration:
    def test_full_run_with_native_engine(self, tmp_path):
        # 启动 mock server
        server = HTTPServer(("127.0.0.1", 0), _SSEHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever)
        thread.daemon = True
        thread.start()

        try:
            config = {
                "target": {"name": "integration-test",
                           "api_url": f"http://127.0.0.1:{port}/v1/chat/completions",
                           "api_key": "sk-test", "model": "test-model"},
                "engine": "native",
                "request": {"stream": True, "extra_args": {}},
                "test": {"concurrency": [1], "requests_per_level": [2], "dataset": "openqa"},
                "pass_criteria": [{"metric": "success_rate", "operator": ">=", "threshold": 1.0}],
                "degradation": {"enabled": False, "start_concurrency": 1, "step": 1, "min_concurrency": 1},
                "output": {"dir": str(tmp_path / "results"), "formats": ["json", "csv"], "charts": False},
            }
            config_path = tmp_path / "test.yaml"
            config_path.write_text(yaml.dump(config, sort_keys=False))

            # 通过 subprocess 运行 CLI
            result = subprocess.run(
                [sys.executable, "-m", "llm_stress_test", "run", "--config", str(config_path)],
                capture_output=True, text=True,
                env={**__import__("os").environ, "PYTHONPATH": str(Path(__file__).parent.parent / "src")},
                timeout=60,
            )

            # 验证成功
            assert result.returncode == 0, f"CLI failed: {result.stdout}\n{result.stderr}"
            assert "报告已生成" in result.stdout or "results" in result.stdout

            # 验证输出文件
            results_dirs = list((tmp_path / "results").iterdir())
            assert len(results_dirs) == 1
            result_dir = results_dirs[0]
            assert (result_dir / "summary.json").exists()
            assert (result_dir / "summary.csv").exists()
            assert (result_dir / "config_snapshot.yaml").exists()

            # 验证 API key 脱敏
            saved_config = yaml.safe_load((result_dir / "config_snapshot.yaml").read_text())
            assert saved_config["target"]["api_key"] == "***REDACTED***"
        finally:
            server.shutdown()
