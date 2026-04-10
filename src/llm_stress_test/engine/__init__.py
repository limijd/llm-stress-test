"""引擎注册与工厂"""
from __future__ import annotations
from .base import BaseEngine


def get_engine(name: str) -> BaseEngine:
    if name == "native":
        from .native import NativeEngine
        return NativeEngine()
    elif name == "evalscope":
        from .evalscope import EvalScopeEngine
        return EvalScopeEngine()
    else:
        raise ValueError(f"未知引擎: {name}")
