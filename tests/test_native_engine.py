import json
import pytest
from aioresponses import aioresponses
from llm_stress_test.engine.native import NativeEngine
from llm_stress_test.models import EngineConfig


def _make_sse_response(tokens: list[str], include_usage: bool = True) -> str:
    chunks = []
    for token in tokens:
        chunk = {"choices": [{"delta": {"content": token}, "index": 0, "finish_reason": None}]}
        chunks.append(f"data: {json.dumps(chunk)}\n\n")
    final = {"choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]}
    if include_usage:
        final["usage"] = {"prompt_tokens": 10, "completion_tokens": len(tokens)}
    chunks.append(f"data: {json.dumps(final)}\n\n")
    chunks.append("data: [DONE]\n\n")
    return "".join(chunks)


def _make_config(concurrency=1, num_requests=2, stream=True):
    return EngineConfig(
        api_url="http://test-api/v1/chat/completions",
        api_key="test-key", model="test-model",
        concurrency=concurrency, num_requests=num_requests,
        dataset="openqa", stream=stream, extra_args={},
    )


class TestNativeEngine:
    def test_check_available(self):
        engine = NativeEngine()
        ok, msg = engine.check_available()
        assert ok is True

    @pytest.mark.asyncio
    async def test_streaming_request(self):
        engine = NativeEngine()
        config = _make_config(concurrency=1, num_requests=2)
        sse_body = _make_sse_response(["Hello", " world", "!"])
        with aioresponses() as m:
            m.post("http://test-api/v1/chat/completions", body=sse_body,
                   content_type="text/event-stream", repeat=True)
            result = engine.run(config)
        assert len(result.requests) == 2
        assert all(r.success for r in result.requests)
        assert all(r.output_tokens > 0 for r in result.requests)
        assert result.concurrency == 1

    @pytest.mark.asyncio
    async def test_non_streaming_request(self):
        engine = NativeEngine()
        config = _make_config(concurrency=1, num_requests=1, stream=False)
        response_body = {
            "choices": [{"message": {"content": "Hello world"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        with aioresponses() as m:
            m.post("http://test-api/v1/chat/completions", payload=response_body, repeat=True)
            result = engine.run(config)
        assert len(result.requests) == 1
        assert result.requests[0].success is True
        assert result.requests[0].output_tokens == 5

    @pytest.mark.asyncio
    async def test_connection_error(self):
        engine = NativeEngine()
        config = _make_config(concurrency=1, num_requests=1)
        with aioresponses() as m:
            m.post("http://test-api/v1/chat/completions",
                   exception=ConnectionError("refused"), repeat=True)
            result = engine.run(config)
        assert len(result.requests) == 1
        assert result.requests[0].success is False
        assert result.requests[0].error is not None
