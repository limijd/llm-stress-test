"""数据集加载与管理"""
from __future__ import annotations
import json
from pathlib import Path

_PACKAGE_DIR = Path(__file__).parent
_PROJECT_ROOT = _PACKAGE_DIR.parent.parent
_DATASETS_DIR = _PROJECT_ROOT / "datasets"

_BUILTIN_DATASETS = {
    "openqa": _DATASETS_DIR / "openqa.jsonl",
    "longalpaca": _DATASETS_DIR / "longalpaca.jsonl",
}

_FALLBACK_PROMPTS = [
    {"role": "user", "content": "请简要介绍量子计算的基本原理。"},
    {"role": "user", "content": "什么是深度学习？请用简单的语言解释。"},
    {"role": "user", "content": "请解释 Python 中的异步编程模型。"},
    {"role": "user", "content": "描述 TCP 三次握手的过程。"},
    {"role": "user", "content": "什么是 MapReduce？它解决了什么问题？"},
]

class DatasetError(Exception):
    """数据集加载错误"""

def load_dataset(name_or_path: str) -> list[dict]:
    if name_or_path in _BUILTIN_DATASETS:
        path = _BUILTIN_DATASETS[name_or_path]
        if not path.exists():
            return _FALLBACK_PROMPTS.copy()
        return _load_jsonl(path)
    path = Path(name_or_path)
    if not path.exists():
        raise DatasetError(f"数据集不存在: {name_or_path}")
    return _load_jsonl(path)

def _load_jsonl(path: Path) -> list[dict]:
    prompts = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                raise DatasetError(f"数据集格式错误 (行 {line_num}): {e}") from e
            if "messages" in data:
                messages = data["messages"]
                if messages:
                    prompts.append(messages[0] if isinstance(messages[0], dict) else {"role": "user", "content": str(messages[0])})
            elif "question" in data:
                prompts.append({"role": "user", "content": data["question"]})
            elif "instruction" in data:
                prompts.append({"role": "user", "content": data["instruction"]})
            else:
                raise DatasetError(f"数据集格式错误 (行 {line_num}): 缺少 messages/question/instruction 字段")
    if not prompts:
        raise DatasetError(f"数据集为空: {path}")
    return prompts
