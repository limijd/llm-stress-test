"""基于 asyncio + urllib.request 的原生引擎 — 零外部依赖"""
from __future__ import annotations

import asyncio
import json
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from ..models import EngineConfig, LevelResult, RequestMetric
from .base import BaseEngine


class NativeEngine(BaseEngine):
    """直接调用 OpenAI 兼容 API 的原生压测引擎。"""

    def __init__(self) -> None:
        # 进度回调：on_progress(completed, total, concurrency, req_metric)
        self.on_progress: callable | None = None
        # 复用 SSL 上下文避免重复握手开销
        self._ssl_ctx = ssl.create_default_context()

    def check_available(self) -> tuple[bool, str]:
        """原生引擎仅依赖标准库，始终可用。"""
        return True, "原生引擎（stdlib）可用"

    def run(self, config: EngineConfig) -> LevelResult:
        """同步入口，内部用 asyncio.run 驱动异步逻辑。"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self._run_async(config))
                return future.result()
        else:
            return asyncio.run(self._run_async(config))

    async def _run_async(self, config: EngineConfig) -> LevelResult:
        """加载 prompt，并发派发所有请求，收集指标。"""
        prompts = self._load_prompts(config.dataset, config.num_requests)
        sem = asyncio.Semaphore(config.concurrency)
        completed = 0

        async def _tracked_send(i: int) -> RequestMetric:
            nonlocal completed
            async with sem:
                result = await asyncio.to_thread(
                    self._send_request_sync, config, prompts[i % len(prompts)]
                )
            completed += 1
            if self.on_progress:
                self.on_progress(completed, config.num_requests, config.concurrency, result)
            return result

        start = time.monotonic()
        tasks = [_tracked_send(i) for i in range(config.num_requests)]
        metrics: list[RequestMetric] = await asyncio.gather(*tasks)
        duration = time.monotonic() - start

        return LevelResult(
            concurrency=config.concurrency,
            num_requests=config.num_requests,
            requests=list(metrics),
            duration=duration,
        )

    def _send_request_sync(
        self,
        config: EngineConfig,
        prompt: dict,
    ) -> RequestMetric:
        """在线程中同步发送单条请求。"""
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }
        body: dict[str, Any] = {
            "model": config.model,
            "messages": [prompt],
            "stream": config.stream,
            **config.extra_args,
        }

        t_start = time.monotonic()
        try:
            req = urllib.request.Request(
                config.api_url,
                data=json.dumps(body).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            # 根据协议选择是否使用 SSL
            parsed = urllib.parse.urlparse(config.api_url)
            ctx = self._ssl_ctx if parsed.scheme == "https" else None
            resp = urllib.request.urlopen(req, context=ctx, timeout=300)

            if config.stream:
                return self._handle_stream(resp, t_start)
            else:
                return self._handle_non_stream(resp, t_start)
        except Exception as exc:
            total = time.monotonic() - t_start
            return RequestMetric(
                success=False,
                ttft=0.0,
                total_latency=total,
                output_tokens=0,
                input_tokens=0,
                tpot=0.0,
                error=str(exc),
            )

    def _handle_stream(self, resp, t_start: float) -> RequestMetric:
        """逐行解析 SSE，统计 TTFT、token 数。"""
        first_token_time: float | None = None
        output_tokens = 0
        input_tokens = 0

        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload == "[DONE]":
                break

            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue

            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                if delta.get("content") and first_token_time is None:
                    first_token_time = time.monotonic()
            usage = chunk.get("usage")
            if usage:
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)

        resp.close()
        t_end = time.monotonic()
        total_latency = t_end - t_start
        ttft = (first_token_time - t_start) if first_token_time else total_latency
        tpot = (total_latency - ttft) / output_tokens if output_tokens > 1 else 0.0

        return RequestMetric(
            success=True,
            ttft=ttft,
            total_latency=total_latency,
            output_tokens=output_tokens,
            input_tokens=input_tokens,
            tpot=tpot,
            error=None,
        )

    def _handle_non_stream(self, resp, t_start: float) -> RequestMetric:
        """解析非流式 JSON 响应。"""
        data = json.loads(resp.read().decode("utf-8"))
        resp.close()
        t_end = time.monotonic()
        total_latency = t_end - t_start

        usage = data.get("usage", {})
        output_tokens = usage.get("completion_tokens", 0)
        input_tokens = usage.get("prompt_tokens", 0)

        return RequestMetric(
            success=True,
            ttft=total_latency,
            total_latency=total_latency,
            output_tokens=output_tokens,
            input_tokens=input_tokens,
            tpot=total_latency / output_tokens if output_tokens > 1 else 0.0,
            error=None,
        )

    def _load_prompts(self, dataset: str, num_requests: int) -> list[dict]:
        from ..dataset import load_dataset
        return load_dataset(dataset)
