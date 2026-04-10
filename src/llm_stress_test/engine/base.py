"""引擎抽象基类"""
from __future__ import annotations
from abc import ABC, abstractmethod
from ..models import EngineConfig, LevelResult


class BaseEngine(ABC):
    @abstractmethod
    def run(self, config: EngineConfig) -> LevelResult:
        ...

    @abstractmethod
    def check_available(self) -> tuple[bool, str]:
        ...
