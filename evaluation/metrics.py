"""Shared evaluation harness: one Precision/Recall/F1/AUC table (with Wilson
confidence intervals, matching both source papers' reporting convention)
across every family that's been built, driven off the single AttackTrace /
BlueVerdict shapes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

from sklearn.metrics import roc_auc_score

from src.common.schemas import AttackTrace, BlueVerdict


def wilson_ci(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion — same convention
    Whispers of Wealth and Protocol-Level Attacks both used for ASR
    reporting."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.959963985  # z-score for 95% two-sided
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = p + z ** 2 / (2 * n)
    adj = z * math.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n)
    lower = (centre - adj) / denom
    upper = (centre + adj) / denom
    return (max(0.0, lower), min(1.0, upper))


@dataclass
class RateResult:
    successes: int
    n: int
    rate: float
    ci_low: float
    ci_high: float

    def __str__(self) -> str:
        return f"{self.rate:.1%} [{self.ci_low:.1%}, {self.ci_high:.1%}] ({self.successes}/{self.n})"


def success_rate(flags: Sequence[bool]) -> RateResult:
    n = len(flags)
    successes = sum(1 for f in flags if f)
    rate = successes / n if n else 0.0
    lo, hi = wilson_ci(successes, n)
    return RateResult(successes=successes, n=n, rate=rate, ci_low=lo, ci_high=hi)


@dataclass
class ClassificationResult:
    n: int
    precision: float
    recall: float
    f1: float
    auc: float
    tp: int
    fp: int
    tn: int
    fn: int

    def __str__(self) -> str:
        return (
            f"n={self.n}  P={self.precision:.3f}  R={self.recall:.3f}  "
            f"F1={self.f1:.3f}  AUC={self.auc:.3f}  "
            f"(TP={self.tp} FP={self.fp} TN={self.tn} FN={self.fn})"
        )


def classification_metrics(
    traces: List[AttackTrace], verdicts: List[BlueVerdict]
) -> ClassificationResult:
    """traces[i] and verdicts[i] must correspond to the same trace_id."""
    assert len(traces) == len(verdicts)
    y_true = [1 if t.ground_truth_label else 0 for t in traces]
    y_pred = [1 if v.predicted_label else 0 for v in verdicts]
    y_score = [v.risk_score for v in verdicts]

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    try:
        auc = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else float("nan")
    except ValueError:
        auc = float("nan")

    return ClassificationResult(
        n=len(traces), precision=precision, recall=recall, f1=f1, auc=auc,
        tp=tp, fp=fp, tn=tn, fn=fn,
    )
