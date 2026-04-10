"""下载 LongAlpaca-12k 数据集并转换为 JSONL 格式"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

_URL = "https://huggingface.co/datasets/Yukang/LongAlpaca-12k/resolve/main/LongAlpaca-12k.json"
_CACHE_DIR = Path.home() / ".cache" / "llm-stress-test" / "datasets"
_CACHE_FILE = _CACHE_DIR / "LongAlpaca-12k.json"

# 输出目录：datasets/ 与本脚本同目录
_OUTPUT_FILE = Path(__file__).parent / "longalpaca.jsonl"


def download() -> None:
    """下载并转换 LongAlpaca-12k 数据集。"""
    # 确保缓存目录存在
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # 若缓存中已有原始文件，跳过下载
    if not _CACHE_FILE.exists():
        print(f"正在从 HuggingFace 下载 LongAlpaca-12k ...")
        urllib.request.urlretrieve(_URL, _CACHE_FILE)
        print(f"下载完成，已缓存至: {_CACHE_FILE}")
    else:
        print(f"命中缓存: {_CACHE_FILE}")

    # 读取 JSON 并转换为 JSONL
    print(f"正在转换为 JSONL 格式 ...")
    with _CACHE_FILE.open("r", encoding="utf-8") as f:
        records = json.load(f)

    with _OUTPUT_FILE.open("w", encoding="utf-8") as out:
        for record in records:
            # LongAlpaca 格式: {"instruction": "...", "input": "...", "output": "..."}
            instruction = record.get("instruction", "")
            input_text = record.get("input", "")
            content = f"{instruction}\n{input_text}".strip() if input_text else instruction
            line = {"messages": [{"role": "user", "content": content}]}
            out.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"转换完成，共 {len(records)} 条记录，输出至: {_OUTPUT_FILE}")


if __name__ == "__main__":
    download()
