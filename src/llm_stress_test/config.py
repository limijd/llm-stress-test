"""配置加载、校验、环境变量展开、脱敏"""
from __future__ import annotations
import copy
import os
import re
from pathlib import Path
from . import _yaml as yaml

class ConfigError(Exception):
    """配置错误"""

_VALID_ENGINES = {"evalscope", "native"}
_VALID_OPERATORS = {">=", "<=", ">", "<", "=="}
_REQUIRED_TOP_KEYS = ["target", "engine", "test", "pass_criteria", "degradation", "output"]
_REQUIRED_TARGET_KEYS = ["name", "api_url", "api_key", "model"]
_ENV_VAR_PATTERN = re.compile(r"^\$\{([^}]+)\}$")

DEFAULTS = {
    "request": {"stream": True, "extra_args": {}},
    "degradation": {"enabled": True, "start_concurrency": 50, "step": 10, "min_concurrency": 10},
    "output": {"dir": "./results", "formats": ["json", "csv", "html"], "charts": True},
}

def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"配置文件不存在: {path}")
    try:
        text = p.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
    except (yaml.YAMLError, Exception) as e:
        raise ConfigError(f"YAML 语法错误: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError("配置文件顶层必须是字典")
    for key, default in DEFAULTS.items():
        if key not in data:
            data[key] = copy.deepcopy(default)
        elif isinstance(default, dict):
            for dk, dv in default.items():
                data[key].setdefault(dk, dv)
    return data

def validate_config(cfg: dict) -> None:
    for key in _REQUIRED_TOP_KEYS:
        if key not in cfg:
            raise ConfigError(f"缺少必填字段: {key}")
    target = cfg["target"]
    for key in _REQUIRED_TARGET_KEYS:
        if key not in target:
            raise ConfigError(f"target 缺少必填字段: {key}")
    if cfg["engine"] not in _VALID_ENGINES:
        raise ConfigError(f"engine 必须是 {_VALID_ENGINES} 之一，当前值: {cfg['engine']}")
    test = cfg["test"]
    concurrency = test.get("concurrency", [])
    requests_per_level = test.get("requests_per_level", [])
    if len(concurrency) != len(requests_per_level):
        raise ConfigError(
            f"concurrency ({len(concurrency)} 项) 和 requests_per_level ({len(requests_per_level)} 项) 长度必须一致"
        )
    for criterion in cfg.get("pass_criteria", []):
        op = criterion.get("operator", "")
        if op not in _VALID_OPERATORS:
            raise ConfigError(f"不支持的操作符: {op}，支持: {_VALID_OPERATORS}")

def expand_env_vars(cfg: dict) -> dict:
    result = copy.deepcopy(cfg)
    _expand_recursive(result)
    return result

def _expand_recursive(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str):
                m = _ENV_VAR_PATTERN.match(value)
                if m:
                    var_name = m.group(1)
                    env_val = os.environ.get(var_name)
                    if env_val is None:
                        raise ConfigError(f"环境变量 {var_name} 未设置")
                    obj[key] = env_val
            elif isinstance(value, (dict, list)):
                _expand_recursive(value)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, str):
                m = _ENV_VAR_PATTERN.match(item)
                if m:
                    var_name = m.group(1)
                    env_val = os.environ.get(var_name)
                    if env_val is None:
                        raise ConfigError(f"环境变量 {var_name} 未设置")
                    obj[i] = env_val
            elif isinstance(item, (dict, list)):
                _expand_recursive(item)

def merge_cli_overrides(cfg: dict, overrides: dict) -> dict:
    result = copy.deepcopy(cfg)
    for key, value in overrides.items():
        parts = key.split(".")
        target = result
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return result

def sanitize_for_export(cfg: dict) -> dict:
    result = copy.deepcopy(cfg)
    api_key = result.get("target", {}).get("api_key", "")
    if api_key and not _ENV_VAR_PATTERN.match(api_key):
        result["target"]["api_key"] = "***REDACTED***"
    return result
