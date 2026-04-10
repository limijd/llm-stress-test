"""命令行入口：run / validate / report 三个子命令"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
import yaml

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

# ------------------------------------------------------------------ #
# 顶层命令组
# ------------------------------------------------------------------ #

@click.group()
@click.option("-v", "--verbose", count=True, help="日志详细程度（-v 为 INFO，-vv 为 DEBUG）")
def main(verbose: int) -> None:
    """LLM 推理服务压力测试工具"""
    level = logging.WARNING
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s  %(levelname)s  %(name)s  %(message)s")


# ------------------------------------------------------------------ #
# run 子命令
# ------------------------------------------------------------------ #

@main.command()
@click.option("--config", "config_path", required=True, help="YAML 配置文件路径")
@click.option("--engine", "engine_override", default=None, help="覆盖配置中的引擎名称")
@click.option("--concurrency", default=None, help="覆盖并发数列表，逗号分隔，例如 1,5,10")
@click.option("--api-url", default=None, help="覆盖目标 API URL")
@click.option("--model", default=None, help="覆盖模型名称")
@click.option("--dataset", default=None, help="覆盖数据集名称")
def run(
    config_path: str,
    engine_override: str | None,
    concurrency: str | None,
    api_url: str | None,
    model: str | None,
    dataset: str | None,
) -> None:
    """执行压力测试"""
    # ---- 1. 加载、合并 CLI 覆盖、校验、展开环境变量 ----
    try:
        cfg = load_config(config_path)
    except ConfigError as e:
        click.echo(click.style(f"配置加载失败：{e}", fg="red"), err=True)
        sys.exit(1)

    # 构造 CLI 覆盖项
    overrides: dict = {}
    if engine_override:
        overrides["engine"] = engine_override
    if api_url:
        overrides["target.api_url"] = api_url
    if model:
        overrides["target.model"] = model
    if dataset:
        overrides["test.dataset"] = dataset
    if concurrency:
        try:
            levels = [int(c.strip()) for c in concurrency.split(",")]
            overrides["test.concurrency"] = levels
        except ValueError:
            click.echo(click.style("--concurrency 格式错误，需要逗号分隔的整数", fg="red"), err=True)
            sys.exit(1)

    if overrides:
        cfg = merge_cli_overrides(cfg, overrides)

    try:
        validate_config(cfg)
    except ConfigError as e:
        click.echo(click.style(f"配置校验失败：{e}", fg="red"), err=True)
        sys.exit(1)

    try:
        cfg = expand_env_vars(cfg)
    except ConfigError as e:
        click.echo(click.style(f"环境变量展开失败：{e}", fg="red"), err=True)
        sys.exit(1)

    # ---- 2. 初始化引擎 ----
    engine_name: str = cfg["engine"]
    try:
        engine = get_engine(engine_name)
    except ValueError as e:
        click.echo(click.style(str(e), fg="red"), err=True)
        sys.exit(1)

    available, reason = engine.check_available()
    if not available:
        click.echo(click.style(f"引擎 {engine_name} 不可用：{reason}", fg="red"), err=True)
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
        concurrency=0,        # 由 Orchestrator 按档位覆盖
        num_requests=0,       # 同上
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
        click.echo(
            f"\r  [{completed}/{total}] "
            f"elapsed={elapsed:.1f}s  "
            f"last: {status} {toks}toks {req_metric.total_latency:.1f}s",
            nl=False,
        )

    def _on_level_start(c, n, idx, total, is_degradation):
        _level_start_time[0] = _time.monotonic()
        if is_degradation:
            click.echo(f"\n[降级] 并发={c}, 请求数={n}")
        else:
            click.echo(f"\n[{idx}/{total}] 并发={c}, 请求数={n}")

    def _on_level_done(report, idx, total):
        agg = report.aggregated
        passed = report.pass_result.passed
        badge = click.style("PASS", fg="green") if passed else click.style("FAIL", fg="red")
        click.echo(
            f"\n  => Success Rate: {agg.success_rate * 100:.1f}%"
            f"  Gen toks/s: {agg.gen_toks_per_sec:.1f}"
            f"  Avg TTFT: {agg.avg_ttft:.1f}s"
            f"  [{badge}]"
        )
        if not passed:
            for d in report.pass_result.details:
                if not d.passed:
                    click.echo(f"     {d.metric}: {d.actual:.3f} {d.operator} {d.threshold}")

    # 设置回调
    if hasattr(engine, 'on_progress'):
        engine.on_progress = _on_progress

    degradation = cfg.get("degradation", {})
    orchestrator = Orchestrator(engine=engine, criteria=criteria)
    orchestrator.on_level_start = _on_level_start
    orchestrator.on_level_done = _on_level_done

    click.echo(click.style(f"\n[Engine: {engine_name}] {reason}", bold=True))
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
        click.echo(click.style(f"系统性中止：{exc}", fg="red"), err=True)
        sys.exit(2)

    # ---- 6. 终端输出结论 ----
    click.echo("\n" + "=" * 60)
    if result.target_passed:
        click.echo(click.style(f"结论: 最大通过并发数 = {result.max_passing_concurrency}", fg="green"))
    elif result.aborted:
        click.echo(click.style(f"测试中止：{result.abort_error.message}", fg="red"))
    else:
        if result.max_passing_concurrency:
            click.echo(click.style(f"结论: 最大通过并发数 = {result.max_passing_concurrency}", fg="yellow"))
            click.echo("建议: 要求赛事主方增加 GPU Server 数，或调整参赛队伍数量")
        else:
            click.echo(click.style("结论: 未找到通过的并发档位", fg="red"))
            click.echo("建议: 要求赛事主方增加 GPU Server 数，或调整参赛队伍数量")

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
            # 构建 pass_criteria 阈值字典供图表绘制阈值线
            criteria_thresholds = {c["metric"]: c["threshold"] for c in cfg.get("pass_criteria", [])}
            generate_charts(result.level_reports, result_dir, pass_criteria=criteria_thresholds)
        except Exception as exc:  # 图表生成失败不应阻断主流程
            click.echo(click.style(f"图表生成失败（非致命）：{exc}", fg="yellow"), err=True)

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
            click.echo(click.style(f"HTML 报告生成失败（非致命）：{exc}", fg="yellow"), err=True)

    click.echo(click.style(f"\n报告已生成: {result_dir}/", fg="cyan"))


# ------------------------------------------------------------------ #
# validate 子命令
# ------------------------------------------------------------------ #

@main.command()
@click.option("--config", "config_path", required=True, help="YAML 配置文件路径")
def validate(config_path: str) -> None:
    """校验配置文件格式"""
    try:
        cfg = load_config(config_path)
        validate_config(cfg)
    except ConfigError as e:
        click.echo(click.style(f"配置校验失败：{e}", fg="red"), err=True)
        sys.exit(1)

    click.echo(click.style("配置校验通过", fg="green"))


# ------------------------------------------------------------------ #
# report 子命令
# ------------------------------------------------------------------ #

@main.command()
@click.option("--result-dir", required=True, help="测试结果目录路径")
@click.option("--formats", default="html,csv", show_default=True, help="导出格式，逗号分隔")
def report(result_dir: str, formats: str) -> None:
    """从已有结果目录生成（或补生成）报告"""
    summary_file = Path(result_dir) / "summary.json"
    if not summary_file.exists():
        click.echo(click.style(f"未找到 summary.json：{summary_file}", fg="red"), err=True)
        sys.exit(1)

    try:
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        click.echo(click.style(f"summary.json 解析失败：{e}", fg="red"), err=True)
        sys.exit(1)

    click.echo(f"结果目录: {result_dir}")
    click.echo(f"共 {len(summary)} 个并发档位")
    for item in summary:
        badge = click.style("PASS", fg="green") if item.get("passed") else click.style("FAIL", fg="red")
        m = item.get("metrics", {})
        click.echo(
            f"  并发={item['concurrency']}"
            f"  Success Rate: {m.get('success_rate', 0) * 100:.1f}%"
            f"  Gen toks/s: {m.get('gen_toks_per_sec', 0):.1f}"
            f"  [{badge}]"
        )

    fmt_list = [f.strip() for f in formats.split(",")]
    click.echo(f"导出格式: {', '.join(fmt_list)}")
