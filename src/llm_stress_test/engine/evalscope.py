"""EvalScope 引擎：通过 subprocess 包装 evalscope perf CLI"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from typing import Any

from ..models import EngineConfig, LevelResult, RequestMetric
from .base import BaseEngine

# evalscope perf 超时（1 小时）
_TIMEOUT_SECONDS = 3600


class EvalScopeEngine(BaseEngine):
    """调用 `evalscope perf` CLI 完成压测的引擎。"""

    def check_available(self) -> tuple[bool, str]:
        """检查 evalscope 是否可执行。

        先用 shutil.which 查找，找不到再尝试 `python3 -m evalscope --version`。
        """
        if shutil.which("evalscope") is not None:
            return True, "evalscope 可用"

        # 兜底：尝试以模块方式运行
        try:
            result = subprocess.run(
                ["python3", "-m", "evalscope", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return True, "evalscope 可通过 python3 -m evalscope 使用"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return False, "evalscope 未安装，请执行: pip3 install evalscope"

    def _build_command(self, config: EngineConfig) -> list[str]:
        """根据 EngineConfig 构造 evalscope perf 的 CLI 参数列表。"""
        cmd = [
            "evalscope",
            "perf",
            "--url", config.api_url,
            "--parallel", str(config.concurrency),
            "--number", str(config.num_requests),
            "--api", "openai",
            "--model", config.model,
            "--dataset", config.dataset,
        ]

        if config.stream:
            cmd.append("--stream")

        if config.extra_args:
            cmd += ["--extra-args", json.dumps(config.extra_args, ensure_ascii=False)]

        return cmd

    def run(self, config: EngineConfig) -> LevelResult:
        """调用 evalscope perf，解析输出并返回 LevelResult。"""
        cmd = self._build_command(config)
        t_start = time.monotonic()

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=_TIMEOUT_SECONDS,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            duration = time.monotonic() - t_start
            return self._make_error_result(config, str(exc), duration)

        duration = time.monotonic() - t_start

        if proc.returncode != 0:
            error_msg = proc.stderr or f"evalscope exited with code {proc.returncode}"
            return self._make_error_result(config, error_msg, duration)

        return self._parse_output(config, proc.stdout, proc.stderr, duration)

    def _parse_output(
        self,
        config: EngineConfig,
        stdout: str,
        stderr: str,
        duration: float,
    ) -> LevelResult:
        """尝试解析 evalscope 的输出，转换为 LevelResult。

        evalscope 的输出格式不固定，因此采用多级宽松解析：
        1. 尝试将 stdout 整体作为 JSON 解析；
        2. 逐行搜索 JSON 对象；
        3. 全部失败时返回错误结果。
        """
        # 第 1 级：整体 JSON
        data = _try_parse_json(stdout)
        if data is not None:
            return self._make_level_result(config, data, duration)

        # 第 2 级：逐行搜索 JSON
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            data = _try_parse_json(line)
            if data is not None:
                return self._make_level_result(config, data, duration)

        # 第 3 级：解析失败，返回错误结果
        error_msg = stderr or "evalscope 输出无法解析"
        return self._make_error_result(config, error_msg, duration)

    def _make_level_result(
        self,
        config: EngineConfig,
        data: dict[str, Any],
        duration: float,
    ) -> LevelResult:
        """将 evalscope 输出的 JSON 字段映射到 RequestMetric 列表。

        evalscope perf 聚合结果中可能含有的字段（尽力而为映射）：
          - total_requests / num_requests
          - success_requests / failed_requests
          - avg_ttft / time_to_first_token
          - avg_latency / avg_total_time
          - throughput_tokens / avg_output_tokens
          - avg_input_tokens
        """
        total = int(data.get("total_requests", data.get("num_requests", config.num_requests)))
        success_count = int(data.get("success_requests", total))
        failed_count = int(data.get("failed_requests", total - success_count))

        avg_ttft: float = float(
            data.get("avg_ttft", data.get("time_to_first_token", 0.0)) or 0.0
        )
        avg_latency: float = float(
            data.get("avg_latency", data.get("avg_total_time", duration / max(total, 1))) or 0.0
        )
        avg_output_tokens: int = int(
            data.get("avg_output_tokens", data.get("throughput_tokens", 0)) or 0
        )
        avg_input_tokens: int = int(data.get("avg_input_tokens", 0) or 0)
        avg_tpot: float = (
            (avg_latency - avg_ttft) / avg_output_tokens
            if avg_output_tokens > 1
            else 0.0
        )

        metrics: list[RequestMetric] = []
        for i in range(total):
            success = i < success_count
            metrics.append(
                RequestMetric(
                    success=success,
                    ttft=avg_ttft if success else 0.0,
                    total_latency=avg_latency if success else 0.0,
                    output_tokens=avg_output_tokens if success else 0,
                    input_tokens=avg_input_tokens if success else 0,
                    tpot=avg_tpot if success else 0.0,
                    error=None if success else "request failed",
                )
            )

        return LevelResult(
            concurrency=config.concurrency,
            num_requests=total,
            requests=metrics,
            duration=duration,
        )

    def _make_error_result(
        self,
        config: EngineConfig,
        error_msg: str,
        duration: float = 0.0,
    ) -> LevelResult:
        """构造全部失败的 LevelResult。"""
        failed_metric = RequestMetric(
            success=False,
            ttft=0.0,
            total_latency=0.0,
            output_tokens=0,
            input_tokens=0,
            tpot=0.0,
            error=error_msg,
        )
        return LevelResult(
            concurrency=config.concurrency,
            num_requests=config.num_requests,
            requests=[failed_metric] * config.num_requests,
            duration=duration,
        )


def _try_parse_json(text: str) -> dict[str, Any] | None:
    """安全地将字符串解析为 JSON dict，失败返回 None。"""
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
    return None
