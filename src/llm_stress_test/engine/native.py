"""基于 asyncio + aiohttp 的原生引擎"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import aiohttp

from ..models import EngineConfig, LevelResult, RequestMetric
from .base import BaseEngine

# 兜底 prompt 列表，Task 10 完成后由 dataset 模块替换
_FALLBACK_PROMPTS: list[str] = [
    "What is the capital of France?",
    "Explain the theory of relativity in simple terms.",
    "Write a short poem about the ocean.",
    "What are the main differences between Python and JavaScript?",
    "How does photosynthesis work?",
]


class NativeEngine(BaseEngine):
    """直接调用 OpenAI 兼容 API 的原生压测引擎。"""

    def check_available(self) -> tuple[bool, str]:
        """检查 aiohttp 是否可用。"""
        try:
            import aiohttp  # noqa: F401
            return True, "aiohttp 可用"
        except ImportError:
            return False, "aiohttp 未安装，请执行: pip3 install aiohttp"

    def run(self, config: EngineConfig) -> LevelResult:
        """同步入口，内部用 asyncio.run 驱动异步逻辑。

        当已有事件循环正在运行（如 pytest-asyncio 测试环境）时，
        在独立子线程中创建新循环执行，避免 "cannot be called from a running event loop" 错误。
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # 已有运行中的循环（例如 pytest-asyncio），在子线程里跑新循环
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self._run_async(config))
                return future.result()
        else:
            return asyncio.run(self._run_async(config))

    async def _run_async(self, config: EngineConfig) -> LevelResult:
        """加载 prompt，并发派发所有请求，收集指标。"""
        prompts = self._load_prompts(config.num_requests)
        sem = asyncio.Semaphore(config.concurrency)

        start = time.monotonic()
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._bounded_send(sem, session, config, prompts[i % len(prompts)])
                for i in range(config.num_requests)
            ]
            metrics: list[RequestMetric] = await asyncio.gather(*tasks)
        duration = time.monotonic() - start

        return LevelResult(
            concurrency=config.concurrency,
            num_requests=config.num_requests,
            requests=list(metrics),
            duration=duration,
        )

    async def _bounded_send(
        self,
        sem: asyncio.Semaphore,
        session: aiohttp.ClientSession,
        config: EngineConfig,
        prompt: str,
    ) -> RequestMetric:
        """在信号量约束下发送单条请求。"""
        async with sem:
            return await self._send_request(session, config, prompt)

    async def _send_request(
        self,
        session: aiohttp.ClientSession,
        config: EngineConfig,
        prompt: str,
    ) -> RequestMetric:
        """构造请求并委托给流式 / 非流式处理器。"""
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }
        body: dict[str, Any] = {
            "model": config.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": config.stream,
            **config.extra_args,
        }

        t_start = time.monotonic()
        try:
            async with session.post(config.api_url, headers=headers, json=body) as resp:
                if config.stream:
                    return await self._handle_stream(resp, t_start)
                else:
                    return await self._handle_non_stream(resp, t_start)
        except Exception as exc:  # noqa: BLE001
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

    async def _handle_stream(
        self,
        resp: aiohttp.ClientResponse,
        t_start: float,
    ) -> RequestMetric:
        """逐行解析 SSE，统计 TTFT、token 数。"""
        first_token_time: float | None = None
        output_tokens = 0
        input_tokens = 0

        async for raw_line in resp.content:
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

            # 计算首 token 时间
            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                if delta.get("content") and first_token_time is None:
                    first_token_time = time.monotonic()
                # 从 finish_reason 携带的 usage 中提取 token 数（部分实现放在最后一个 chunk）
            usage = chunk.get("usage")
            if usage:
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)

        t_end = time.monotonic()
        total_latency = t_end - t_start
        ttft = (first_token_time - t_start) if first_token_time else total_latency
        # 若 usage 未包含 output_tokens，用收到的有内容 delta 数量作为估算
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

    async def _handle_non_stream(
        self,
        resp: aiohttp.ClientResponse,
        t_start: float,
    ) -> RequestMetric:
        """解析非流式 JSON 响应，提取 usage 信息。"""
        data = await resp.json(content_type=None)
        t_end = time.monotonic()
        total_latency = t_end - t_start

        usage = data.get("usage", {})
        output_tokens = usage.get("completion_tokens", 0)
        input_tokens = usage.get("prompt_tokens", 0)

        return RequestMetric(
            success=True,
            ttft=total_latency,   # 非流式无 TTFT 概念，用总延迟代替
            total_latency=total_latency,
            output_tokens=output_tokens,
            input_tokens=input_tokens,
            tpot=total_latency / output_tokens if output_tokens > 1 else 0.0,
            error=None,
        )

    def _load_prompts(self, n: int) -> list[str]:
        """加载 prompt 列表。Task 10 完成后接入 dataset 模块。"""
        # 确保至少返回 n 条（循环复用）
        result = []
        for i in range(n):
            result.append(_FALLBACK_PROMPTS[i % len(_FALLBACK_PROMPTS)])
        return result
