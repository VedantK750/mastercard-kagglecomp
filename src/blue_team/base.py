"""Detector ABC — every Blue detector implements evaluate(trace) -> BlueVerdict.
unified_pipeline.py dispatches to the right detector by trace.family."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from src.common.schemas import AttackTrace, BlueVerdict


class Detector(ABC):
    family: str = "base"
    trainable: bool = False  # True only for detectors evaluation/adaptive_loop.py refits per generation

    @abstractmethod
    def evaluate(self, trace: AttackTrace) -> BlueVerdict:
        raise NotImplementedError

    def fit(self, train_traces: List[AttackTrace]) -> None:
        """No-op by default. Trainable detectors override this to refit
        internal parameters (a classifier, a threshold) on an accumulating
        train pool. Called once per generation by evaluation/adaptive_loop.py
        with exactly one persistent detector instance per family — fitted
        state lives on the instance, so it must be constructed once and
        reused across generations, never recreated."""
        return None

    def calibrate(self, null_traces: List[AttackTrace]) -> None:
        """No-op by default. Detectors with a one-class component override
        this to fit thresholds on ATTACK-FREE traces only.

        Deliberately separate from fit(): fit() consumes labels and cannot
        generalize to a strategy absent from its training pool (the
        coefficient on the deciding feature is unidentified there), whereas
        calibrate() never sees a label, so an unseen strategy is scored on
        the same footing as a seen one. Callers must pass only traces with
        ground_truth_label=False — implementations assert this."""
        return None
