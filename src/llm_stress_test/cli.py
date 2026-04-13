"""命令行入口：run / validate / report 三个子命令 — 纯 argparse，零外部依赖"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time as _time
from pathlib import Path

from . import _yaml as yaml
from .config import (
    ConfigError,
    expand_env_vars,
    load_config,
    merge_cli_overrides,
    sanitize_for_export,
    validate_config,
)
from .engine import get_engine
from .models import Criterion, EngineConfig
from .orchestrator import Orchestrator, SystemicAbort
from .report.exporter import create_result_dir, export_csv, export_json


# ── ANSI 彩色输出 ───────────────────────────────────────

_USE_COLOR: bool | None = None

_COLORS = {
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "cyan": "\033[36m",
    "dim": "\033[2m",
    "bold": "\033[1m",
    "reset": "\033[0m",
}


def _color_enabled() -> bool:
    global _USE_COLOR
    if _USE_COLOR is None:
        _USE_COLOR = sys.stdout.isatty()
    return _USE_COLOR


def _s(text: str, fg: str | None = None, bold: bool = False) -> str:
    """给文本加 ANSI 颜色。"""
    if not _color_enabled():
        return text
    prefix = ""
    if bold:
        prefix += _COLORS["bold"]
    if fg and fg in _COLORS:
        prefix += _COLORS[fg]
    if prefix:
        return f"{prefix}{text}{_COLORS['reset']}"
    return text


def _echo(msg: str = "", err: bool = False, nl: bool = True) -> None:
    stream = sys.stderr if err else sys.stdout
    print(msg, end="\n" if nl else "", file=stream, flush=True)


# ── 格式化工具 ─────────────────────────────────────────

def _bar(ratio: float, width: int = 20) -> str:
    """生成一个文本进度条 ████░░░░"""
    filled = int(ratio * width)
    empty = width - filled
    if _color_enabled():
        return _s("█" * filled, fg="green") + _s("░" * empty, fg="dim")
    return "█" * filled + "░" * empty


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m{s:.0f}s"


def _divider(char: str = "─", width: int = 72) -> str:
    return char * width


# ── 子命令实现 ──────────────────────────────────────────


def cmd_run(args: argparse.Namespace) -> None:
    """执行压力测试"""

    # ---- 1. 加载配置 ----
    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        _echo(_s(f"  ERROR  配置加载失败：{e}", fg="red"), err=True)
        sys.exit(1)

    overrides: dict = {}
    if args.engine:
        overrides["engine"] = args.engine
    if args.api_url:
        overrides["target.api_url"] = args.api_url
    if args.model:
        overrides["target.model"] = args.model
    if args.dataset:
        overrides["test.dataset"] = args.dataset
    if args.concurrency:
        try:
            levels = [int(c.strip()) for c in args.concurrency.split(",")]
            overrides["test.concurrency"] = levels
        except ValueError:
            _echo(_s("  ERROR  --concurrency 格式错误，需要逗号分隔的整数", fg="red"), err=True)
            sys.exit(1)

    if overrides:
        cfg = merge_cli_overrides(cfg, overrides)

    try:
        validate_config(cfg)
    except ConfigError as e:
        _echo(_s(f"  ERROR  配置校验失败：{e}", fg="red"), err=True)
        sys.exit(1)

    try:
        cfg = expand_env_vars(cfg)
    except ConfigError as e:
        _echo(_s(f"  ERROR  环境变量展开失败：{e}", fg="red"), err=True)
        sys.exit(1)

    # ---- 2. 初始化引擎 ----
    engine_name: str = cfg["engine"]
    try:
        engine = get_engine(engine_name)
    except ValueError as e:
        _echo(_s(str(e), fg="red"), err=True)
        sys.exit(1)

    available, reason = engine.check_available()
    if not available:
        _echo(_s(f"  ERROR  引擎 {engine_name} 不可用：{reason}", fg="red"), err=True)
        sys.exit(1)

    # ---- 3. 构建 Criterion / EngineConfig ----
    criteria = [
        Criterion(metric=c["metric"], operator=c["operator"], threshold=c["threshold"])
        for c in cfg.get("pass_criteria", [])
    ]

    target = cfg["target"]
    test_cfg = cfg["test"]
    request_cfg = cfg.get("request", {})
    config_template = EngineConfig(
        api_url=target["api_url"],
        api_key=target["api_key"],
        model=target["model"],
        concurrency=0,
        num_requests=0,
        dataset=test_cfg.get("dataset", "openqa"),
        stream=request_cfg.get("stream", True),
        extra_args=request_cfg.get("extra_args", {}),
    )

    concurrency_levels = test_cfg["concurrency"]
    requests_levels = test_cfg["requests_per_level"]
    total_requests = sum(requests_levels)

    # ---- 4. 打印 Banner ----
    _echo()
    _echo(_s("╔══════════════════════════════════════════════════════════════════════╗", fg="cyan"))
    _echo(_s("║                    LLM Stress Test                                  ║", fg="cyan"))
    _echo(_s("╚══════════════════════════════════════════════════════════════════════╝", fg="cyan"))
    _echo()
    _echo(_s("  Test Configuration", bold=True))
    _echo(_divider())
    _echo(f"  Target     {_s(target.get('name', ''), bold=True)}")
    _echo(f"  API URL    {target['api_url']}")
    _echo(f"  Model      {target['model']}")
    _echo(f"  Engine     {engine_name} ({reason})")
    _echo(f"  Stream     {request_cfg.get('stream', True)}")
    _echo(f"  Dataset    {test_cfg.get('dataset', 'openqa')}")
    _echo(f"  Levels     {' -> '.join(f'c{c}×{n}' for c, n in zip(concurrency_levels, requests_levels))}")
    _echo(f"  Total      {total_requests} requests across {len(concurrency_levels)} levels")
    _echo()
    _echo(_s("  Pass Criteria", bold=True))
    _echo(_divider())
    for c in criteria:
        _echo(f"  {c.metric} {c.operator} {c.threshold}")
    _echo()

    # ---- 5. 执行测试 ----
    _echo(_s("  Phase 1: Ramp-up Test", bold=True))
    _echo(_divider("═"))

    _level_start_time = [0.0]
    _level_ok = [0]
    _level_fail = [0]
    all_reports: list = []  # 用于最终汇总

    def _on_progress(completed: int, total: int, concurrency: int, req_metric) -> None:
        elapsed = _time.monotonic() - _level_start_time[0]
        if req_metric.success:
            _level_ok[0] += 1
            status = _s("OK", fg="green")
            detail = (
                f"ttft={req_metric.ttft:.2f}s "
                f"latency={req_metric.total_latency:.1f}s "
                f"tokens={req_metric.output_tokens}"
            )
        else:
            _level_fail[0] += 1
            status = _s("FAIL", fg="red")
            err_msg = (req_metric.error or "unknown")[:50]
            detail = err_msg

        pct = completed / total
        bar = _bar(pct)
        _echo(
            f"\r  {bar} {completed:>3}/{total}  "
            f"{_fmt_duration(elapsed):>6}  "
            f"{status} {detail}",
            nl=False,
        )

    def _on_level_start(c, n, idx, total, is_degradation):
        _level_start_time[0] = _time.monotonic()
        _level_ok[0] = 0
        _level_fail[0] = 0

        if is_degradation:
            _echo()
            _echo(f"  {_s('DEGRADE', fg='yellow')}  concurrency={c}  requests={n}")
        else:
            _echo(f"  {_s(f'[{idx}/{total}]', fg='cyan', bold=True)}  concurrency={_s(str(c), bold=True)}  requests={n}")

    def _on_level_done(report, idx, total):
        all_reports.append(report)
        elapsed = _time.monotonic() - _level_start_time[0]
        agg = report.aggregated
        passed = report.pass_result.passed
        badge = _s(" PASS ", fg="green", bold=True) if passed else _s(" FAIL ", fg="red", bold=True)

        # 换行清除进度条残留
        _echo()
        _echo(f"  {_divider('·', 68)}")
        _echo(f"  Result: {badge}  Duration: {_fmt_duration(elapsed)}  "
              f"OK: {_level_ok[0]}  Failed: {_level_fail[0]}")
        _echo()

        # 指标表格
        col_w = 14
        headers = ["Success%", "Tok/s", "TTFT(s)", "TPOT(s)", "P50(s)", "P99(s)", "Avg(s)"]
        values = [
            f"{agg.success_rate * 100:.1f}%",
            f"{agg.gen_toks_per_sec:.1f}",
            f"{agg.avg_ttft:.3f}",
            f"{agg.avg_tpot:.3f}",
            f"{agg.p50_latency:.2f}",
            f"{agg.p99_latency:.2f}",
            f"{agg.avg_latency:.2f}",
        ]
        _echo("  " + "".join(h.center(col_w) for h in headers))
        _echo("  " + "".join("─" * col_w for _ in headers))
        _echo("  " + "".join(v.center(col_w) for v in values))
        _echo()

        # 不通过的判据明细
        if not passed:
            for d in report.pass_result.details:
                if not d.passed:
                    _echo(f"  {_s('✗', fg='red')} {d.metric}: "
                          f"{_s(f'{d.actual:.3f}', fg='red')} {d.operator} {d.threshold}")
            _echo()

    if hasattr(engine, "on_progress"):
        engine.on_progress = _on_progress

    degradation = cfg.get("degradation", {})
    orchestrator = Orchestrator(engine=engine, criteria=criteria)
    orchestrator.on_level_start = _on_level_start
    orchestrator.on_level_done = _on_level_done

    test_start = _time.monotonic()
    try:
        result = orchestrator.run_test(
            concurrency=concurrency_levels,
            requests_per_level=requests_levels,
            config_template=config_template,
            degradation_enabled=degradation.get("enabled", True),
            degradation_step=degradation.get("step", 10),
            degradation_min=degradation.get("min_concurrency", 10),
        )
    except SystemicAbort as exc:
        _echo(_s(f"\n  ABORT  {exc}", fg="red", bold=True), err=True)
        sys.exit(2)

    test_duration = _time.monotonic() - test_start

    # ---- 6. 汇总表格 ----
    _echo(_s("  Summary", bold=True))
    _echo(_divider("═"))

    # 表头
    col_widths = [6, 6, 10, 10, 10, 10, 10, 10, 8]
    headers = ["Conc", "Reqs", "Success%", "Tok/s", "TTFT(s)", "TPOT(s)", "P50(s)", "P99(s)", "Result"]
    header_line = "  " + "".join(h.center(w) for h, w in zip(headers, col_widths))
    _echo(header_line)
    _echo("  " + "".join("─" * w for w in col_widths))

    prev_tps = None
    for report in result.level_reports:
        agg = report.aggregated
        passed = report.pass_result.passed
        badge = _s("PASS", fg="green") if passed else _s("FAIL", fg="red")

        # 吞吐变化
        tps = agg.gen_toks_per_sec
        if prev_tps and prev_tps > 0:
            delta_pct = (tps - prev_tps) / prev_tps * 100
            if delta_pct >= 5:
                tps_str = f"{tps:.1f}{_s(f'+{delta_pct:.0f}%', fg='green')}"
            elif delta_pct <= -5:
                tps_str = f"{tps:.1f}{_s(f'{delta_pct:.0f}%', fg='red')}"
            else:
                tps_str = f"{tps:.1f}"
        else:
            tps_str = f"{tps:.1f}"
        prev_tps = tps

        vals = [
            str(report.concurrency).center(col_widths[0]),
            str(report.num_requests).center(col_widths[1]),
            f"{agg.success_rate * 100:.1f}%".center(col_widths[2]),
            tps_str.center(col_widths[3]),
            f"{agg.avg_ttft:.3f}".center(col_widths[4]),
            f"{agg.avg_tpot:.3f}".center(col_widths[5]),
            f"{agg.p50_latency:.2f}".center(col_widths[6]),
            f"{agg.p99_latency:.2f}".center(col_widths[7]),
            badge.center(col_widths[8]),
        ]
        _echo("  " + "".join(vals))

    _echo("  " + "".join("─" * w for w in col_widths))
    _echo()

    # ---- 7. 结论 ----
    _echo(_s("  Conclusion", bold=True))
    _echo(_divider("═"))

    if result.aborted:
        _echo(f"  {_s('ABORTED', fg='red', bold=True)}  {result.abort_error.message}")
    elif result.target_passed:
        _echo(f"  {_s('PASSED', fg='green', bold=True)}  "
              f"Target concurrency {_s(str(result.max_passing_concurrency), bold=True)} meets all criteria.")
    else:
        target_c = concurrency_levels[-1]
        _echo(f"  {_s('FAILED', fg='red', bold=True)}  "
              f"Target concurrency {target_c} did not meet all criteria.")
        if result.max_passing_concurrency:
            _echo(f"  {_s('Max stable concurrency:', bold=True)} "
                  f"{_s(str(result.max_passing_concurrency), fg='yellow', bold=True)}")
        else:
            _echo(f"  {_s('No passing concurrency found.', fg='red')}")

    # 性能趋势分析
    reports = result.level_reports
    if len(reports) >= 2:
        _echo()
        _echo(_s("  Performance Trend", bold=True))
        _echo(_divider())

        first = reports[0]
        last = reports[-1]

        # TTFT 趋势
        ttft_ratio = last.aggregated.avg_ttft / first.aggregated.avg_ttft if first.aggregated.avg_ttft > 0 else 0
        if ttft_ratio > 2:
            _echo(f"  {_s('!', fg='yellow')} TTFT increased {_s(f'{ttft_ratio:.1f}x', fg='red')} "
                  f"from c{first.concurrency} to c{last.concurrency} "
                  f"({first.aggregated.avg_ttft:.2f}s -> {last.aggregated.avg_ttft:.2f}s)")
        elif ttft_ratio > 1.2:
            _echo(f"  TTFT increased {ttft_ratio:.1f}x from c{first.concurrency} to c{last.concurrency}")

        # 吞吐趋势
        tps_first = first.aggregated.gen_toks_per_sec
        tps_last = last.aggregated.gen_toks_per_sec
        if tps_first > 0:
            tps_ratio = tps_last / tps_first
            if tps_ratio < 0.7:
                _echo(f"  {_s('!', fg='yellow')} Throughput dropped to {_s(f'{tps_ratio:.0%}', fg='red')} "
                      f"({tps_first:.1f} -> {tps_last:.1f} tok/s)")
            elif tps_ratio > 1.3:
                _echo(f"  Throughput scaled {_s(f'{tps_ratio:.1f}x', fg='green')} "
                      f"({tps_first:.1f} -> {tps_last:.1f} tok/s)")
            else:
                _echo(f"  Throughput relatively stable ({tps_first:.1f} -> {tps_last:.1f} tok/s)")

        # 瓶颈判断
        failing_reports = [r for r in reports if not r.pass_result.passed]
        if failing_reports:
            failing_metrics: dict[str, int] = {}
            for r in failing_reports:
                for d in r.pass_result.details:
                    if not d.passed:
                        failing_metrics[d.metric] = failing_metrics.get(d.metric, 0) + 1
            if failing_metrics:
                bottleneck = max(failing_metrics, key=failing_metrics.get)
                _echo(f"  Bottleneck: {_s(bottleneck, fg='yellow', bold=True)} "
                      f"(failed in {failing_metrics[bottleneck]}/{len(reports)} levels)")

    _echo()
    _echo(f"  Total duration: {_fmt_duration(test_duration)}")
    _echo(f"  Total requests: {sum(r.num_requests for r in reports)}")
    _echo()

    # ---- 8. 保存结果 ----
    output_cfg = cfg.get("output", {})
    base_dir = output_cfg.get("dir", "./results")
    result_dir = create_result_dir(
        base_dir=base_dir,
        model_name=target.get("model", "unknown"),
        engine_name=engine_name,
    )

    config_snapshot = sanitize_for_export(cfg)
    (Path(result_dir) / "config_snapshot.yaml").write_text(
        yaml.dump(config_snapshot, allow_unicode=True), encoding="utf-8"
    )

    formats = output_cfg.get("formats", ["json", "csv", "html"])
    generate_charts_flag = output_cfg.get("charts", True)

    saved = []
    if "json" in formats:
        export_json(result.level_reports, result_dir)
        saved.append("JSON")

    if "csv" in formats:
        export_csv(result.level_reports, result_dir)
        saved.append("CSV")

    if generate_charts_flag and result.level_reports:
        try:
            from .report.chart import generate_charts
            criteria_thresholds = {c["metric"]: c["threshold"] for c in cfg.get("pass_criteria", [])}
            generate_charts(result.level_reports, result_dir, pass_criteria=criteria_thresholds)
            saved.append("Charts")
        except Exception as exc:
            _echo(_s(f"  WARN  图表生成失败: {exc}", fg="yellow"), err=True)

    if "html" in formats and result.level_reports:
        try:
            from .report.html import generate_html_report
            generate_html_report(
                reports=result.level_reports,
                result_dir=result_dir,
                config_snapshot=config_snapshot,
                target_passed=result.target_passed,
                max_passing_concurrency=result.max_passing_concurrency,
            )
            saved.append("HTML")
        except Exception as exc:
            _echo(_s(f"  WARN  HTML 报告生成失败: {exc}", fg="yellow"), err=True)

    _echo(_s("  Output", bold=True))
    _echo(_divider())
    _echo(f"  Directory  {result_dir}/")
    _echo(f"  Formats    {', '.join(saved)}")
    _echo()


def cmd_validate(args: argparse.Namespace) -> None:
    """校验配置文件格式"""
    try:
        cfg = load_config(args.config)
        validate_config(cfg)
    except ConfigError as e:
        _echo(_s(f"  ERROR  配置校验失败：{e}", fg="red"), err=True)
        sys.exit(1)
    _echo(_s("  OK  配置校验通过", fg="green"))


def cmd_report(args: argparse.Namespace) -> None:
    """从已有结果目录生成（或补生成）报告"""
    summary_file = Path(args.result_dir) / "summary.json"
    if not summary_file.exists():
        _echo(_s(f"  ERROR  未找到 summary.json：{summary_file}", fg="red"), err=True)
        sys.exit(1)

    try:
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        _echo(_s(f"  ERROR  summary.json 解析失败：{e}", fg="red"), err=True)
        sys.exit(1)

    _echo(f"  结果目录: {args.result_dir}")
    _echo(f"  共 {len(summary)} 个并发档位")
    _echo()
    for item in summary:
        badge = _s("PASS", fg="green") if item.get("passed") else _s("FAIL", fg="red")
        m = item.get("metrics", {})
        _echo(
            f"  c{item['concurrency']:>3}  "
            f"Success: {m.get('success_rate', 0) * 100:.1f}%  "
            f"Tok/s: {m.get('gen_toks_per_sec', 0):.1f}  "
            f"TTFT: {m.get('avg_ttft', 0):.3f}s  "
            f"[{badge}]"
        )

    fmt_list = [f.strip() for f in args.formats.split(",")]
    _echo(f"\n  导出格式: {', '.join(fmt_list)}")


# ── 入口 ────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="llm-stress-test",
        description="LLM 推理服务压力测试工具",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="日志详细程度（-v 为 INFO，-vv 为 DEBUG）",
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # ---- run ----
    p_run = subparsers.add_parser("run", help="执行压力测试")
    p_run.add_argument("--config", required=True, help="YAML 配置文件路径")
    p_run.add_argument("--engine", default=None, help="覆盖配置中的引擎名称")
    p_run.add_argument("--concurrency", default=None, help="覆盖并发数列表，逗号分隔")
    p_run.add_argument("--api-url", default=None, help="覆盖目标 API URL")
    p_run.add_argument("--model", default=None, help="覆盖模型名称")
    p_run.add_argument("--dataset", default=None, help="覆盖数据集名称")

    # ---- validate ----
    p_validate = subparsers.add_parser("validate", help="校验配置文件格式")
    p_validate.add_argument("--config", required=True, help="YAML 配置文件路径")

    # ---- report ----
    p_report = subparsers.add_parser("report", help="从已有结果生成报告")
    p_report.add_argument("--result-dir", required=True, help="测试结果目录路径")
    p_report.add_argument("--formats", default="html,csv", help="导出格式，逗号分隔")

    args = parser.parse_args()

    level = logging.WARNING
    if args.verbose == 1:
        level = logging.INFO
    elif args.verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s  %(levelname)s  %(name)s  %(message)s")

    if args.command == "run":
        cmd_run(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "report":
        cmd_report(args)
    else:
        parser.print_help()
        sys.exit(1)
