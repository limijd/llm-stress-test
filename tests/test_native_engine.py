"""NativeEngine 测试 — 使用 stdlib mock HTTP server 替代 aioresponses"""
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

import pytest
from llm_stress_test.engine.native import NativeEngine
from llm_stress_test.models import EngineConfig


def _make_sse_body(tokens: list[str], include_usage: bool = True) -> bytes:
    """生成 SSE 格式的响应体。"""
    chunks = []
    for token in tokens:
        chunk = {"choices": [{"delta": {"content": token}, "index": 0, "finish_reason": None}]}
        chunks.append(f"data: {json.dumps(chunk)}\n\n")
    final = {"choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]}
    if include_usage:
        final["usage"] = {"prompt_tokens": 10, "completion_tokens": len(tokens)}
    chunks.append(f"data: {json.dumps(final)}\n\n")
    chunks.append("data: [DONE]\n\n")
    return "".join(chunks).encode("utf-8")


def _make_json_body(output_tokens: int = 5) -> bytes:
    """生成非流式 JSON 响应体。"""
    body = {
        "choices": [{"message": {"content": "Hello world"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": output_tokens},
    }
    return json.dumps(body).encode("utf-8")


class _MockHandler(BaseHTTPRequestHandler):
    """可配置的 mock HTTP handler。"""
    # 类变量，由测试设置
    response_body: bytes = b""
    content_type: str = "application/json"
    status_code: int = 200
    simulate_error: bool = False

    def do_POST(self):
        # 读取请求体
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)

        if self.simulate_error:
            self.send_error(500, "Internal Server Error")
            return

        self.send_response(self.status_code)
        self.send_header("Content-Type", self.content_type)
        self.send_header("Content-Length", str(len(self.response_body)))
        self.end_headers()
        self.wfile.write(self.response_body)

    def log_message(self, format, *args):
        pass  # 静默日志


@pytest.fixture
def mock_server():
    """启动一个本地 mock HTTP server，返回 (server, port, handler_class)。"""
    # 创建一个新的 handler 类，避免测试间状态污染
    class Handler(_MockHandler):
        response_body = b""
        content_type = "application/json"
        status_code = 200
        simulate_error = False

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    yield server, port, Handler
    server.shutdown()


def _make_config(port: int, concurrency=1, num_requests=2, stream=True):
    return EngineConfig(
        api_url=f"http://127.0.0.1:{port}/v1/chat/completions",
        api_key="test-key", model="test-model",
        concurrency=concurrency, num_requests=num_requests,
        dataset="openqa", stream=stream, extra_args={},
    )


class TestNativeEngine:
    def test_check_available(self):
        engine = NativeEngine()
        ok, msg = engine.check_available()
        assert ok is True

    def test_streaming_request(self, mock_server):
        server, port, handler = mock_server
        handler.response_body = _make_sse_body(["Hello", " world", "!"])
        handler.content_type = "text/event-stream"

        engine = NativeEngine()
        config = _make_config(port, concurrency=1, num_requests=2)
        result = engine.run(config)

        assert len(result.requests) == 2
        assert all(r.success for r in result.requests)
        assert all(r.output_tokens > 0 for r in result.requests)
        assert result.concurrency == 1

    def test_non_streaming_request(self, mock_server):
        server, port, handler = mock_server
        handler.response_body = _make_json_body(output_tokens=5)
        handler.content_type = "application/json"

        engine = NativeEngine()
        config = _make_config(port, concurrency=1, num_requests=1, stream=False)
        result = engine.run(config)

        assert len(result.requests) == 1
        assert result.requests[0].success is True
        assert result.requests[0].output_tokens == 5

    def test_connection_error(self):
        """连接到一个不存在的端口，应返回错误而非抛异常。"""
        engine = NativeEngine()
        config = EngineConfig(
            api_url="http://127.0.0.1:1/v1/chat/completions",
            api_key="test-key", model="test-model",
            concurrency=1, num_requests=1,
            dataset="openqa", stream=True, extra_args={},
        )
        result = engine.run(config)
        assert len(result.requests) == 1
        assert result.requests[0].success is False
        assert result.requests[0].error is not None
