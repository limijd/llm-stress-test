"""图表生成：使用 matplotlib（Agg 无头模式）生成 4 张 PNG 图表"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # 无头模式，必须在 import pyplot 之前设置

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 颜色常量
COLOR_PRIMARY = "#4A90D9"
COLOR_SECONDARY = "#F5A623"
COLOR_THRESHOLD = "red"


def _add_value_labels(ax, bars, fmt="{:.1f}"):
    """在每个柱形顶部添加数值标签"""
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def generate_charts(reports, result_dir, pass_criteria=None):
    """生成 4 张性能图表到 result_dir/charts/ 目录。

    Args:
        reports: list[LevelReport]，各并发档位的测试报告
        result_dir: 结果目录路径（字符串），其下应存在 charts/ 子目录
        pass_criteria: 可选，dict 格式的通过标准，用于在图表上绘制阈值线
    """
    charts_dir = Path(result_dir) / "charts"
    charts_dir.mkdir(exist_ok=True)

    # 提取数据
    concurrencies = [r.concurrency for r in reports]
    x_labels = [str(c) for c in concurrencies]
    x = np.arange(len(concurrencies))

    throughputs = [r.aggregated.gen_toks_per_sec for r in reports]
    ttfts = [r.aggregated.avg_ttft for r in reports]
    p50s = [r.aggregated.p50_latency for r in reports]
    p99s = [r.aggregated.p99_latency for r in reports]
    success_rates = [r.aggregated.success_rate * 100 for r in reports]

    bar_width = 0.5

    # ------------------------------------------------------------------ #
    # 1. throughput.png — 吞吐量（tokens/s）
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, throughputs, width=bar_width, color=COLOR_PRIMARY)
    _add_value_labels(ax, bars, fmt="{:.1f}")
    ax.set_title("Throughput (tokens/s)")
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Tokens/s")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)

    # 绘制阈值线（若提供）
    if pass_criteria and "gen_toks_per_sec" in pass_criteria:
        threshold = pass_criteria["gen_toks_per_sec"]
        ax.axhline(y=threshold, color=COLOR_THRESHOLD, linestyle="--", label=f"Threshold: {threshold}")
        ax.legend()

    fig.tight_layout()
    fig.savefig(charts_dir / "throughput.png", dpi=100)
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # 2. ttft.png — 首 token 延迟（TTFT）
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, ttfts, width=bar_width, color=COLOR_PRIMARY)
    _add_value_labels(ax, bars, fmt="{:.3f}")
    ax.set_title("Average TTFT (s)")
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Seconds")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)

    if pass_criteria and "avg_ttft" in pass_criteria:
        threshold = pass_criteria["avg_ttft"]
        ax.axhline(y=threshold, color=COLOR_THRESHOLD, linestyle="--", label=f"Threshold: {threshold}")
        ax.legend()

    fig.tight_layout()
    fig.savefig(charts_dir / "ttft.png", dpi=100)
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # 3. latency_p50_p99.png — P50 / P99 延迟分组柱状图
    # ------------------------------------------------------------------ #
    group_width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bars_p50 = ax.bar(x - group_width / 2, p50s, width=group_width, color=COLOR_PRIMARY, label="P50")
    bars_p99 = ax.bar(x + group_width / 2, p99s, width=group_width, color=COLOR_SECONDARY, label="P99")
    _add_value_labels(ax, bars_p50, fmt="{:.2f}")
    _add_value_labels(ax, bars_p99, fmt="{:.2f}")
    ax.set_title("Latency P50 / P99 (s)")
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Seconds")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()

    fig.tight_layout()
    fig.savefig(charts_dir / "latency_p50_p99.png", dpi=100)
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # 4. success_rate.png — 请求成功率
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, success_rates, width=bar_width, color=COLOR_PRIMARY)
    _add_value_labels(ax, bars, fmt="{:.1f}")
    ax.set_title("Success Rate (%)")
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Success Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 110)
    # 固定 100% 阈值线
    ax.axhline(y=100, color=COLOR_THRESHOLD, linestyle="--", label="100%")
    ax.legend()

    fig.tight_layout()
    fig.savefig(charts_dir / "success_rate.png", dpi=100)
    plt.close(fig)
