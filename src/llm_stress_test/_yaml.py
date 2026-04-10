"""最小 YAML 子集解析器 — 仅处理本工具配置所需的 YAML 特性，零外部依赖。

支持特性：
  - 键值对: key: value
  - 嵌套映射（缩进）
  - 块序列: - item / - key: value
  - 行内流式序列: [a, b, c]
  - 行内流式映射: {a: b, c: d}
  - 标量: 字符串（带引号/不带）、整数、浮点数、布尔值、null
  - 注释: # 开头
  - 空行

不支持（本工具不需要）：
  - 锚点/别名 (& / *)
  - 多行字符串 (| / >)
  - 标签 (!tag)
  - 合并键 (<<)
"""
from __future__ import annotations


class YAMLError(Exception):
    """YAML 解析错误"""


# ── 公开 API ────────────────────────────────────────────


def safe_load(text: str):
    """解析 YAML 子集，返回 Python 对象。"""
    lines = _preprocess(text)
    if not lines:
        return {}
    value, _ = _parse_block(lines, 0, -1)
    return value


def dump(data, allow_unicode=True, sort_keys=True, default_flow_style=False) -> str:
    """将 Python 对象序列化为 YAML 字符串。"""
    lines: list[str] = []
    _dump_node(data, lines, 0, sort_keys)
    return "\n".join(lines) + "\n"


# ── 预处理 ──────────────────────────────────────────────


def _preprocess(text: str) -> list[tuple[int, str]]:
    """拆分为 (缩进, 内容) 列表，跳过空行和纯注释行。"""
    result: list[tuple[int, str]] = []
    for raw in text.splitlines():
        content = _strip_comment(raw)
        stripped = content.strip()
        if not stripped:
            continue
        indent = len(content) - len(content.lstrip())
        result.append((indent, stripped))
    return result


def _strip_comment(line: str) -> str:
    """去除行内注释，但保护引号内的 #。"""
    in_single = False
    in_double = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            # # 前面必须是空白或行首才算注释
            if i == 0 or line[i - 1] in (" ", "\t"):
                return line[:i].rstrip()
    return line.rstrip()


# ── 解析 ────────────────────────────────────────────────


def _parse_block(lines, pos, parent_indent):
    """解析一个缩进块（映射或序列）。"""
    if pos >= len(lines):
        return None, pos
    _indent, content = lines[pos]
    if content.startswith("- "):
        return _parse_sequence(lines, pos, _indent)
    else:
        return _parse_mapping(lines, pos, _indent)


def _parse_mapping(lines, pos, block_indent):
    """解析映射块。"""
    result = {}
    while pos < len(lines):
        indent, content = lines[pos]
        if indent < block_indent:
            break
        if indent > block_indent:
            # 不该出现——跳过，避免死循环
            pos += 1
            continue

        key, val_str = _split_key_value(content)
        pos += 1

        if val_str:
            result[key] = _parse_scalar_or_flow(val_str)
        else:
            # 值在下一行（嵌套块）
            if pos < len(lines) and lines[pos][0] > block_indent:
                result[key], pos = _parse_block(lines, pos, block_indent)
            else:
                result[key] = None

    return result, pos


def _parse_sequence(lines, pos, block_indent):
    """解析序列块。"""
    result = []
    while pos < len(lines):
        indent, content = lines[pos]
        if indent < block_indent:
            break
        if indent > block_indent:
            break
        if not content.startswith("- "):
            break

        item, pos = _parse_sequence_item(lines, pos, block_indent)
        result.append(item)

    return result, pos


def _parse_sequence_item(lines, pos, seq_indent):
    """解析单个序列项。"""
    _indent, content = lines[pos]
    item_content = content[2:].strip()

    if not item_content:
        # '- ' 后面什么都没有，值在下一行
        pos += 1
        if pos < len(lines) and lines[pos][0] > seq_indent:
            return _parse_block(lines, pos, seq_indent)
        return None, pos

    # 检查是否为 key: value 形式（字典项）
    if _looks_like_mapping_entry(item_content):
        key, val_str = _split_key_value(item_content)
        mapping: dict = {}
        pos += 1

        if val_str:
            mapping[key] = _parse_scalar_or_flow(val_str)
        else:
            # 值在下一行
            if pos < len(lines) and lines[pos][0] > seq_indent and not lines[pos][1].startswith("- "):
                mapping[key], pos = _parse_block(lines, pos, lines[pos][0] - 1)
            else:
                mapping[key] = None

        # 继续解析同一个映射项的后续键值对
        while pos < len(lines):
            next_indent, next_content = lines[pos]
            if next_indent <= seq_indent:
                break
            if next_content.startswith("- "):
                break
            k, v = _split_key_value(next_content)
            pos += 1
            if v:
                mapping[k] = _parse_scalar_or_flow(v)
            else:
                if pos < len(lines) and lines[pos][0] > next_indent:
                    mapping[k], pos = _parse_block(lines, pos, next_indent)
                else:
                    mapping[k] = None

        return mapping, pos
    else:
        pos += 1
        return _parse_scalar_or_flow(item_content), pos


def _looks_like_mapping_entry(text: str) -> bool:
    """判断文本是否像 key: value 格式（排除行内流式集合）。"""
    if text.startswith("[") or text.startswith("{"):
        return False
    # 在引号外查找 ': ' 或末尾 ':'
    in_single = False
    in_double = False
    for i, ch in enumerate(text):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == ":" and not in_single and not in_double:
            if i + 1 >= len(text) or text[i + 1] == " ":
                return True
    return False


def _split_key_value(content: str) -> tuple[str, str]:
    """拆分 'key: value' 为 (key, value_str)。"""
    in_single = False
    in_double = False
    for i, ch in enumerate(content):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == ":" and not in_single and not in_double:
            if i + 1 >= len(content) or content[i + 1] == " ":
                key = content[:i].strip()
                val = content[i + 1 :].strip() if i + 1 < len(content) else ""
                # 去除 key 外层引号
                if len(key) >= 2 and key[0] == key[-1] and key[0] in ('"', "'"):
                    key = key[1:-1]
                return key, val
    return content.strip(), ""


def _parse_scalar_or_flow(text: str):
    """解析标量值或流式集合。"""
    text = text.strip()
    if not text:
        return None
    if text.startswith("["):
        return _parse_flow_sequence(text)
    if text.startswith("{"):
        return _parse_flow_mapping(text)
    return _parse_scalar(text)


def _parse_flow_sequence(text: str) -> list:
    """解析行内序列 [a, b, c]。"""
    inner = text[1:-1].strip()
    if not inner:
        return []
    items = _split_flow_items(inner)
    return [_parse_scalar_or_flow(item.strip()) for item in items]


def _parse_flow_mapping(text: str) -> dict:
    """解析行内映射 {a: b, c: d}。"""
    inner = text[1:-1].strip()
    if not inner:
        return {}
    result = {}
    items = _split_flow_items(inner)
    for item in items:
        key, val = _split_key_value(item.strip())
        result[key] = _parse_scalar_or_flow(val) if val else None
    return result


def _split_flow_items(text: str) -> list[str]:
    """按逗号分割流式集合项，但尊重嵌套的 [] {} 和引号。"""
    items: list[str] = []
    depth = 0
    in_single = False
    in_double = False
    current: list[str] = []
    for ch in text:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif not in_single and not in_double:
            if ch in "[{":
                depth += 1
            elif ch in "]}":
                depth -= 1
            elif ch == "," and depth == 0:
                items.append("".join(current))
                current = []
                continue
        current.append(ch)
    if current:
        items.append("".join(current))
    return items


def _parse_scalar(text: str):
    """解析标量值。"""
    text = text.strip()
    if not text:
        return None

    # 引号字符串
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
        return text[1:-1]

    # null
    if text in ("null", "~", "Null", "NULL"):
        return None

    # 布尔
    if text in ("true", "True", "yes", "Yes", "on", "On", "TRUE", "YES", "ON"):
        return True
    if text in ("false", "False", "no", "No", "off", "Off", "FALSE", "NO", "OFF"):
        return False

    # 整数
    try:
        return int(text)
    except ValueError:
        pass

    # 浮点数
    try:
        return float(text)
    except ValueError:
        pass

    # 默认为字符串
    return text


# ── 序列化 ──────────────────────────────────────────────


def _dump_node(data, lines, indent, sort_keys):
    """递归序列化节点。"""
    prefix = " " * indent
    if isinstance(data, dict):
        keys = sorted(data.keys()) if sort_keys else list(data.keys())
        for key in keys:
            value = data[key]
            if isinstance(value, (dict, list)) and value:
                lines.append(f"{prefix}{_dump_key(key)}:")
                _dump_node(value, lines, indent + 2, sort_keys)
            else:
                lines.append(f"{prefix}{_dump_key(key)}: {_scalar_to_str(value)}")
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                first = True
                keys = sorted(item.keys()) if sort_keys else list(item.keys())
                for key in keys:
                    value = item[key]
                    if first:
                        lines.append(f"{prefix}- {_dump_key(key)}: {_scalar_to_str(value)}")
                        first = False
                    else:
                        lines.append(f"{prefix}  {_dump_key(key)}: {_scalar_to_str(value)}")
            else:
                lines.append(f"{prefix}- {_scalar_to_str(item)}")
    else:
        lines.append(f"{prefix}{_scalar_to_str(data)}")


def _dump_key(key) -> str:
    """序列化映射键。"""
    s = str(key)
    if any(c in s for c in ": #{}[],\"'") or s in (
        "true", "false", "null", "yes", "no",
        "True", "False", "Null", "Yes", "No",
    ):
        return f'"{s}"'
    return s


def _scalar_to_str(value) -> str:
    """标量值转字符串。"""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        if not value:
            return "''"
        # 需要引号的情况
        if any(c in value for c in ':#{}[],"\'') or value in (
            "true", "false", "null", "yes", "no",
            "True", "False", "Null", "Yes", "No",
        ):
            # 用双引号，转义内部双引号
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        # 纯数字字符串也需引号
        try:
            int(value)
            return f'"{value}"'
        except ValueError:
            pass
        try:
            float(value)
            return f'"{value}"'
        except ValueError:
            pass
        return value
    if isinstance(value, list):
        # 空列表或简单标量列表用行内格式
        inner = ", ".join(_scalar_to_str(v) for v in value)
        return f"[{inner}]"
    return str(value)
