"""Sequence detector for the Sequence Anomaly family. Rolling z-score +
velocity + category-drift checks over a trailing window of the transaction
history — the "Sequence detector" role in the Blue architecture.

Deliberately does NOT read `trace.metadata["preset"]`/`baseline_total` (Red's
own bookkeeping of where its baseline ends and its attack tail begins) —
that would be label leakage. Instead it picks a fixed trailing WINDOW off
the raw, time-ordered `transactions` list, exactly as a real fraud system
profiling an account's history would, and treats everything before that
window as the established baseline to compare against.

This structural choice is also what makes `low_and_slow` a genuine,
honestly-earned blind spot rather than an engineered one: it wasn't tuned to
be invisible, it stays inside WINDOW purely because its amounts/cadence are
close to the baseline profile, which is exactly the real-world weakness a
fixed-window rolling detector has against a slow drain.
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import List, Optional

from src.blue_team.base import Detector
from src.common.schemas import AttackTrace, BlueVerdict, Transaction

WINDOW = 5
MIN_BASELINE = 3
Z_THRESHOLD = 2.5
VELOCITY_RATIO_THRESHOLD = 2.0
CATEGORY_DRIFT_THRESHOLD = 0.5


def _avg_gap_seconds(seq: List[Transaction]) -> Optional[float]:
    if len(seq) < 2:
        return None
    deltas = [(seq[i + 1].timestamp - seq[i].timestamp).total_seconds() for i in range(len(seq) - 1)]
    return statistics.mean(deltas)


class SequenceAnomalyDetector(Detector):
    family = "sequence_anomaly"

    def evaluate(self, trace: AttackTrace) -> BlueVerdict:
        txns = sorted(trace.transactions, key=lambda t: t.timestamp)
        if len(txns) < MIN_BASELINE + 1:
            return BlueVerdict(
                trace_id=trace.trace_id, risk_score=0.0, predicted_label=False,
                explanation="sequence too short to profile",
            )

        window = min(WINDOW, len(txns) - MIN_BASELINE)
        window = max(window, 1)
        baseline, recent = txns[:-window], txns[-window:]

        amounts = [t.amount for t in baseline]
        mean_amt = statistics.mean(amounts)
        std_amt = statistics.pstdev(amounts) or 1.0
        recent_mean_amt = statistics.mean(t.amount for t in recent)
        amount_z = (recent_mean_amt - mean_amt) / std_amt

        baseline_gap = _avg_gap_seconds(baseline)
        recent_gap = _avg_gap_seconds(recent)
        velocity_ratio = (
            baseline_gap / recent_gap if (baseline_gap and recent_gap and recent_gap > 0) else 1.0
        )

        baseline_categories = Counter(t.category for t in baseline)
        dominant_category, _ = baseline_categories.most_common(1)[0]
        drift_fraction = sum(1 for t in recent if t.category != dominant_category) / len(recent)

        failed: List[str] = []
        if abs(amount_z) > Z_THRESHOLD:
            failed.append(f"amount_z::{amount_z:.2f} exceeds threshold {Z_THRESHOLD}")
        if velocity_ratio > VELOCITY_RATIO_THRESHOLD:
            failed.append(
                f"velocity_ratio::recent cadence {velocity_ratio:.1f}x faster than baseline "
                f"(threshold {VELOCITY_RATIO_THRESHOLD}x)"
            )
        if drift_fraction > CATEGORY_DRIFT_THRESHOLD:
            failed.append(
                f"category_drift::{drift_fraction:.0%} of recent txns outside dominant "
                f"category {dominant_category!r}"
            )

        risk_score = min(1.0, round(len(failed) / 3, 4))
        predicted_label = len(failed) > 0
        explanation = (
            "sequence anomaly detected: " + "; ".join(failed)
            if failed
            else f"recent {window} transactions consistent with baseline profile"
        )

        return BlueVerdict(
            trace_id=trace.trace_id, risk_score=risk_score, predicted_label=predicted_label,
            triggered_checks=failed, explanation=explanation,
        )
