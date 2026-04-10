"""命令行入口：run / validate / report 三个子命令 — 纯 argparse，零外部依赖"""
from __future__ import annotations

import argparse
import json
import logging
import sys
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

_COLORS = {
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "cyan": "\033[36m",
    "bold": "\033[1m",
    "reset": "\033[0m",
}


def _style(text: str, fg: str | None = None, bold: bool = False) -> str:
    """给文本加 ANSI 颜色。终端不支持时退化为纯文本。"""
    if not sys.stderr.isatty() and not sys.stdout.isatty():
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
    """输出消息。"""
    stream = sys.stderr if err else sys.stdout
    end = "\n" if nl else ""
    print(msg, end=end, file=stream, flush=True)


# ── 子命令实现 ──────────────────────────────────────────


def cmd_run(args: argparse.Namespace) -> None:
    """执行压力测试"""
    # ---- 1. 加载、合并 CLI 覆盖、校验、展开环境变量 ----
    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        _echo(_style(f"配置加载失败：{e}", fg="red"), err=True)
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
            _echo(_style("--concurrency 格式错误，需要逗号分隔的整数", fg="red"), err=True)
            sys.exit(1)

    if overrides:
        cfg = merge_cli_overrides(cfg, overrides)

    try:
        validate_config(cfg)
    except ConfigError as e:
        _echo(_style(f"配置校验失败：{e}", fg="red"), err=True)
        sys.exit(1)

    try:
        cfg = expand_env_vars(cfg)
    except ConfigError as e:
        _echo(_style(f"环境变量展开失败：{e}", fg="red"), err=True)
        sys.exit(1)

    # ---- 2. 初始化引擎 ----
    engine_name: str = cfg["engine"]
    try:
        engine = get_engine(engine_name)
    except ValueError as e:
        _echo(_style(str(e), fg="red"), err=True)
        sys.exit(1)

    available, reason = engine.check_available()
    if not available:
        _echo(_style(f"引擎 {engine_name} 不可用：{reason}", fg="red"), err=True)
        sys.exit(1)

    # ---- 3. 构建 Criterion 列表 ----
    criteria = [
        Criterion(
            metric=c["metric"],
            operator=c["operator"],
            threshold=c["threshold"],
        )
        for c in cfg.get("pass_criteria", [])
    ]

    # ---- 4. 构建 EngineConfig 模板 ----
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

    # ---- 5. 执行测试（带实时进度输出） ----
    import time as _time
    _level_start_time = [0.0]

    def _on_progress(completed: int, total: int, concurrency: int, req_metric) -> None:
        elapsed = _time.monotonic() - _level_start_time[0]
        status = "ok" if req_metric.success else "FAIL"
        toks = req_metric.output_tokens
        _echo(
            f"\r  [{completed}/{total}] "
            f"elapsed={elapsed:.1f}s  "
            f"last: {status} {toks}toks {req_metric.total_latency:.1f}s",
            nl=False,
        )

    def _on_level_start(c, n, idx, total, is_degradation):
        _level_start_time[0] = _time.monotonic()
        if is_degradation:
            _echo(f"\n[降级] 并发={c}, 请求数={n}")
        else:
            _echo(f"\n[{idx}/{total}] 并发={c}, 请求数={n}")

    def _on_level_done(report, idx, total):
        agg = report.aggregated
        passed = report.pass_result.passed
        badge = _style("PASS", fg="green") if passed else _style("FAIL", fg="red")
        _echo(
            f"\n  => Success Rate: {agg.success_rate * 100:.1f}%"
            f"  Gen toks/s: {agg.gen_toks_per_sec:.1f}"
            f"  Avg TTFT: {agg.avg_ttft:.1f}s"
            f"  [{badge}]"
        )
        if not passed:
            for d in report.pass_result.details:
                if not d.passed:
                    _echo(f"     {d.metric}: {d.actual:.3f} {d.operator} {d.threshold}")

    if hasattr(engine, "on_progress"):
        engine.on_progress = _on_progress

    degradation = cfg.get("degradation", {})
    orchestrator = Orchestrator(engine=engine, criteria=criteria)
    orchestrator.on_level_start = _on_level_start
    orchestrator.on_level_done = _on_level_done

    _echo(_style(f"\n[Engine: {engine_name}] {reason}", bold=True))
    try:
        result = orchestrator.run_test(
            concurrency=test_cfg["concurrency"],
            requests_per_level=test_cfg["requests_per_level"],
            config_template=config_template,
            degradation_enabled=degradation.get("enabled", True),
            degradation_step=degradation.get("step", 10),
            degradation_min=degradation.get("min_concurrency", 10),
        )
    except SystemicAbort as exc:
        _echo(_style(f"系统性中止：{exc}", fg="red"), err=True)
        sys.exit(2)

    # ---- 6. 终端输出结论 ----
    _echo("\n" + "=" * 60)
    if result.target_passed:
        _echo(_style(f"结论: 最大通过并发数 = {result.max_passing_concurrency}", fg="green"))
    elif result.aborted:
        _echo(_style(f"测试中止：{result.abort_error.message}", fg="red"))
    else:
        if result.max_passing_concurrency:
            _echo(_style(f"结论: 最大通过并发数 = {result.max_passing_concurrency}", fg="yellow"))
            _echo("建议: 要求赛事主方增加 GPU Server 数，或调整参赛队伍数量")
        else:
            _echo(_style("结论: 未找到通过的并发档位", fg="red"))
            _echo("建议: 要求赛事主方增加 GPU Server 数，或调整参赛队伍数量")

    # ---- 7. 创建结果目录，保存配置快照 ----
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

    # ---- 8. 按 output.formats 导出 ----
    formats = output_cfg.get("formats", ["json", "csv", "html"])
    generate_charts_flag = output_cfg.get("charts", True)

    if "json" in formats:
        export_json(result.level_reports, result_dir)

    if "csv" in formats:
        export_csv(result.level_reports, result_dir)

    if generate_charts_flag and result.level_reports:
        try:
            from .report.chart import generate_charts
            criteria_thresholds = {c["metric"]: c["threshold"] for c in cfg.get("pass_criteria", [])}
            generate_charts(result.level_reports, result_dir, pass_criteria=criteria_thresholds)
        except Exception as exc:
            _echo(_style(f"图表生成失败（非致命）：{exc}", fg="yellow"), err=True)

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
        except Exception as exc:
            _echo(_style(f"HTML 报告生成失败（非致命）：{exc}", fg="yellow"), err=True)

    _echo(_style(f"\n报告已生成: {result_dir}/", fg="cyan"))


def cmd_validate(args: argparse.Namespace) -> None:
    """校验配置文件格式"""
    try:
        cfg = load_config(args.config)
        validate_config(cfg)
    except ConfigError as e:
        _echo(_style(f"配置校验失败：{e}", fg="red"), err=True)
        sys.exit(1)

    _echo(_style("配置校验通过", fg="green"))


def cmd_report(args: argparse.Namespace) -> None:
    """从已有结果目录生成（或补生成）报告"""
    summary_file = Path(args.result_dir) / "summary.json"
    if not summary_file.exists():
        _echo(_style(f"未找到 summary.json：{summary_file}", fg="red"), err=True)
        sys.exit(1)

    try:
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        _echo(_style(f"summary.json 解析失败：{e}", fg="red"), err=True)
        sys.exit(1)

    _echo(f"结果目录: {args.result_dir}")
    _echo(f"共 {len(summary)} 个并发档位")
    for item in summary:
        badge = _style("PASS", fg="green") if item.get("passed") else _style("FAIL", fg="red")
        m = item.get("metrics", {})
        _echo(
            f"  并发={item['concurrency']}"
            f"  Success Rate: {m.get('success_rate', 0) * 100:.1f}%"
            f"  Gen toks/s: {m.get('gen_toks_per_sec', 0):.1f}"
            f"  [{badge}]"
        )

    fmt_list = [f.strip() for f in args.formats.split(",")]
    _echo(f"导出格式: {', '.join(fmt_list)}")


# ── 入口 ────────────────────────────────────────────────


def main() -> None:
    """构建 argparse 解析器并分发子命令。"""
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

    # 设置日志级别
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
