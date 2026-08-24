"""Detector ABC — every Blue detector implements evaluate(trace) -> BlueVerdict.
unified_pipeline.py dispatches to the right detector by trace.family."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.common.schemas import AttackTrace, BlueVerdict


class Detector(ABC):
    family: str = "base"

    @abstractmethod
    def evaluate(self, trace: AttackTrace) -> BlueVerdict:
        raise NotImplementedError
