"""编排器：多级并发调度 + 自动降级探测"""
from __future__ import annotations

from dataclasses import dataclass, field

from .engine.base import BaseEngine
from .metrics import aggregate, judge
from .models import (
    AggregatedMetrics,
    Criterion,
    EngineConfig,
    LevelResult,
    PassResult,
    SystemicError,
)


class SystemicAbort(Exception):
    """系统性故障，终止测试"""

    def __init__(self, error: SystemicError) -> None:
        super().__init__(error.message)
        self.error = error


@dataclass
class LevelReport:
    concurrency: int
    num_requests: int
    level_result: LevelResult
    aggregated: AggregatedMetrics
    pass_result: PassResult


@dataclass
class TestRunResult:
    level_reports: list[LevelReport]
    level_results: list[LevelResult]
    target_passed: bool = False
    max_passing_concurrency: int | None = None
    degradation_skipped: bool = False
    aborted: bool = False
    abort_error: SystemicError | None = None


class Orchestrator:
    def __init__(self, engine: BaseEngine, criteria: list[Criterion]) -> None:
        self._engine = engine
        self._criteria = criteria

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def run_test(
        self,
        concurrency: list[int],
        requests_per_level: list[int],
        config_template: EngineConfig,
        degradation_enabled: bool = True,
        degradation_step: int = 10,
        degradation_min: int = 10,
    ) -> TestRunResult:
        """执行完整测试流程。

        Phase 1: 从低到高依次跑所有并发档位，无论是否通过都继续。
        Phase 2: 仅以最后（最高）一个档位的结果判定目标是否通过。
        Phase 3: 若目标失败且启用了降级，从目标并发向下探测，找到第一个通过的并发值。
        """
        level_reports: list[LevelReport] = []
        level_results: list[LevelResult] = []

        # 以并发值为键缓存已测结果，避免重复执行
        tested_cache: dict[int, LevelReport] = {}

        # Phase 1 —— 依次执行所有档位
        try:
            for c, n in zip(concurrency, requests_per_level):
                report = self._run_level(config_template, c, n)
                tested_cache[c] = report
                level_reports.append(report)
                level_results.append(report.level_result)
        except SystemicAbort as exc:
            return TestRunResult(
                level_reports=level_reports,
                level_results=level_results,
                aborted=True,
                abort_error=exc.error,
            )

        # Phase 2 —— 判定目标（最高并发档位）
        target_report = level_reports[-1]
        target_passed = target_report.pass_result.passed
        target_concurrency = concurrency[-1]
        target_requests = requests_per_level[-1]

        if target_passed:
            return TestRunResult(
                level_reports=level_reports,
                level_results=level_results,
                target_passed=True,
                max_passing_concurrency=target_concurrency,
            )

        # Phase 3 —— 降级探测
        if not degradation_enabled:
            return TestRunResult(
                level_reports=level_reports,
                level_results=level_results,
                target_passed=False,
                max_passing_concurrency=None,
                degradation_skipped=True,
            )

        # 计算每请求的 base_ratio，用于推算探测档位的请求数
        base_ratio: float = target_requests / target_concurrency if target_concurrency > 0 else 1.0

        max_passing: int | None = None
        probe = target_concurrency - degradation_step

        try:
            while probe >= degradation_min:
                if probe in tested_cache:
                    report = tested_cache[probe]
                else:
                    probe_requests = max(1, round(base_ratio * probe))
                    report = self._run_level(config_template, probe, probe_requests)
                    tested_cache[probe] = report
                    level_reports.append(report)
                    level_results.append(report.level_result)

                if report.pass_result.passed:
                    max_passing = probe
                    break

                probe -= degradation_step
        except SystemicAbort as exc:
            return TestRunResult(
                level_reports=level_reports,
                level_results=level_results,
                target_passed=False,
                aborted=True,
                abort_error=exc.error,
            )

        return TestRunResult(
            level_reports=level_reports,
            level_results=level_results,
            target_passed=False,
            max_passing_concurrency=max_passing,
        )

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _run_level(
        self,
        template: EngineConfig,
        concurrency: int,
        num_requests: int,
    ) -> LevelReport:
        """执行单个并发档位，返回聚合报告。"""
        config = EngineConfig(
            api_url=template.api_url,
            api_key=template.api_key,
            model=template.model,
            concurrency=concurrency,
            num_requests=num_requests,
            dataset=template.dataset,
            stream=template.stream,
            extra_args=template.extra_args,
        )
        result = self._engine.run(config)
        self._check_systemic_errors(result)
        agg = aggregate(result)
        pass_result = judge(agg, self._criteria)
        return LevelReport(
            concurrency=concurrency,
            num_requests=num_requests,
            level_result=result,
            aggregated=agg,
            pass_result=pass_result,
        )

    def _check_systemic_errors(self, result: LevelResult) -> None:
        """检查系统性错误，发现时抛出 SystemicAbort。

        规则：
        - Auth 错误（401/403）→ 立即中止
        - 网络错误（dns/connect/refused）在本轮中 >= 3 → 中止
        - 连续 5xx 服务端错误 >= 10 → 中止
        """
        network_error_keywords = ("dns", "connect", "refused", "timeout", "connection")
        network_errors = 0
        consecutive_5xx = 0
        max_consecutive_5xx = 0

        for req in result.requests:
            if req.success:
                consecutive_5xx = 0
                continue

            # 检查 Auth 错误
            if req.error and any(
                str(code) in req.error for code in ("401", "403")
            ):
                raise SystemicAbort(
                    SystemicError(
                        error_type="auth_error",
                        message=f"认证失败：{req.error}",
                        status_code=int(next(c for c in ("401", "403") if c in req.error)),
                    )
                )

            # 统计网络错误
            if req.error and any(kw in req.error.lower() for kw in network_error_keywords):
                network_errors += 1

            # 统计连续 5xx
            if req.error and any(
                str(code) in req.error for code in ("500", "501", "502", "503", "504")
            ):
                consecutive_5xx += 1
                max_consecutive_5xx = max(max_consecutive_5xx, consecutive_5xx)
            else:
                consecutive_5xx = 0

        if network_errors >= 3:
            raise SystemicAbort(
                SystemicError(
                    error_type="network_error",
                    message=f"本轮发现 {network_errors} 个网络错误，疑似网络不可达",
                )
            )

        if max_consecutive_5xx >= 10:
            raise SystemicAbort(
                SystemicError(
                    error_type="server_error",
                    message=f"检测到 {max_consecutive_5xx} 个连续 5xx 错误，服务端不可用",
                )
            )
