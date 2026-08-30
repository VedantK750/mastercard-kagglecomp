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

`trainable = True`: evaluation/adaptive_loop.py can call `fit()` to refit a
small logistic-regression classifier over `[amount_z, velocity_ratio,
drift_fraction, cusum_max_neg]` on an accumulating labeled pool. Until
`fit()` has ever been called successfully (`self._clf is None`),
`evaluate()` falls back to the exact hardcoded three-threshold OR rule
above, byte-for-byte in LOGIC — but its three inputs now come from the
frozen-baseline split below, a deliberate bug fix, not a preserved-as-is
value. A fresh instance (never fit) is exactly what the held-out
generalization check needs, since reusing the main loop's continually-
retrained instance there would be leakage — it will already have seen
low_and_slow.

Two changes validated in `evaluation/feature_validation.py` (Part 2a) and
`evaluation/feature_validation_seq_v2.py` (C0-C5):

1. Frozen baseline, not a sliding complement of `recent`. The prior
   `baseline, recent = txns[:-WINDOW], txns[-WINDOW:]` split let a long
   low_and_slow tail leak into its own reference window — for a 23-txn
   trace (8 true baseline + 15 tail), `txns[:-5]` was 8 true baseline + 10
   tail transactions, diluting the very comparison meant to catch it.
   Fixed by anchoring `baseline` at the sequence START, capped at
   `BASELINE_WINDOW=8` — it never slides or grows into the tail
   regardless of how long the attack runs. `RECENT_WINDOW` still adapts
   down (never below 1) on short sequences (e.g. credential_ato's 8+4=12
   default) so no preset is excluded outright. This changes
   `amount_z`/`velocity_ratio`/`drift_fraction`'s VALUES for any sequence
   long enough for the old split to have differed (only low_and_slow, in
   this generator's current presets) — Generation-0's hardcoded-threshold
   LOGIC is unchanged, and empirically (`evaluation/phase3_reproduction.py`,
   re-run after this fix) its OUTPUT is unchanged too: low_and_slow's
   corrected amount_z values (~-1.6 to +0.5, see the validation script) are
   still far below the fixed Z_THRESHOLD=2.5 either way — that threshold
   was never tuned to catch a signal this subtle, buggy baseline or not.
   Still 0/8 low_and_slow caught, case_c_rate still 33.3%, byte-identical
   to before this fix. Only the TRAINED path below benefits from the
   correction — a data-driven boundary can use the improved AUC where a
   fixed threshold can't.
2. `cusum_max_neg`: a cumulative-sum control-chart statistic (slack
   `CUSUM_K=0.5`, standardized units) accumulated over every transaction
   after the frozen baseline — not just the trailing `RECENT_WINDOW` — so
   it accumulates evidence of a sustained small deviation across a long
   tail exactly where a single-window snapshot statistic cancels out.
   Validated as the one candidate with genuine incremental value (recall
   0.0->0.83, Case C 6->1 on the held-out test split). Documented
   robustness caveat: its discriminative power degrades smoothly and sits
   near the noise floor by `amount_multiplier~0.97` (see the validation
   script's robustness sweep), and it resets on any above-mean transaction
   Red could interleave — both real, quantified limits, not claimed away.
   `below_frac`/`longest_streak` and `dollar_rate_ratio` were also tested
   and are NOT included here: the former is fully redundant with
   `cusum_max_neg` once combined (identical ablation numbers), the latter
   never outperformed random (AUC 0.167) — both dropped, kept only as
   reported Phase B results, not live features.
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import List, Optional, Tuple

from sklearn.linear_model import LogisticRegression

from src.blue_team.base import Detector
from src.common.schemas import AttackTrace, BlueVerdict, Transaction

WINDOW = 5
MIN_BASELINE = 3
Z_THRESHOLD = 2.5
VELOCITY_RATIO_THRESHOLD = 2.0
CATEGORY_DRIFT_THRESHOLD = 0.5

BASELINE_WINDOW = 8   # frozen reference: chronologically first N txns, capped
RECENT_WINDOW = 5     # trailing window; adapts down (never below 1) on short sequences
CUSUM_K = 0.5          # CUSUM slack, in standardized (z-score) units


def _avg_gap_seconds(seq: List[Transaction]) -> Optional[float]:
    if len(seq) < 2:
        return None
    deltas = [(seq[i + 1].timestamp - seq[i].timestamp).total_seconds() for i in range(len(seq) - 1)]
    return statistics.mean(deltas)


def _frozen_baseline_recent_split(txns: List[Transaction]) -> Tuple[List[Transaction], List[Transaction]]:
    """Anchors `baseline` at the sequence START, capped at BASELINE_WINDOW —
    it never slides forward or grows as the tail lengthens (the fix for the
    contamination bug: a long low_and_slow tail can no longer leak into the
    reference window). `RECENT_WINDOW` adapts down (never below 1) only when
    the whole sequence is too short to fit a full baseline + a full recent
    window side by side (credential_ato's default preset produces only
    8+4=12 total) — this only ever shrinks `recent`, never re-widens
    `baseline` into the tail."""
    recent_n = min(RECENT_WINDOW, max(1, len(txns) - MIN_BASELINE))
    baseline_n = min(BASELINE_WINDOW, len(txns) - recent_n)
    return txns[:baseline_n], txns[-recent_n:]


def _cusum_max_neg(baseline: List[Transaction], post: List[Transaction], k: float = CUSUM_K) -> float:
    """Cumulative-sum control-chart statistic over every transaction AFTER
    the frozen baseline (not just `recent`) — accumulates evidence of a
    sustained small downward deviation across a long tail, which a single
    window-boundary z-score cancels out. Resets toward 0 on any above-mean
    transaction — a known, documented weakness, not hidden."""
    if not post:
        return 0.0
    mean = statistics.mean(t.amount for t in baseline)
    std = statistics.pstdev(t.amount for t in baseline) or 1.0
    s_neg = max_neg = 0.0
    for t in post:
        z = (t.amount - mean) / std
        s_neg = min(0.0, s_neg + z + k)
        max_neg = min(max_neg, s_neg)
    return max_neg


class SequenceAnomalyDetector(Detector):
    family = "sequence_anomaly"
    trainable = True

    def __init__(self) -> None:
        self._clf: Optional[LogisticRegression] = None

    def _features(self, trace: AttackTrace) -> Optional[Tuple[List[float], str, int]]:
        """Returns ([amount_z, velocity_ratio, drift_fraction, cusum_max_neg],
        dominant_category, window) or None if the sequence is too short to
        profile. `baseline` is the FROZEN split (see _frozen_baseline_recent_split
        above) — this is the fixed contamination bug, not a preserved-as-is
        computation."""
        txns = sorted(trace.transactions, key=lambda t: t.timestamp)
        if len(txns) < MIN_BASELINE + 1:
            return None

        baseline, recent = _frozen_baseline_recent_split(txns)
        post = txns[len(baseline):]  # everything after the frozen baseline, for cusum

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

        cusum_max_neg = _cusum_max_neg(baseline, post)

        return [amount_z, velocity_ratio, drift_fraction, cusum_max_neg], dominant_category, len(recent)

    def fit(self, train_traces: List[AttackTrace]) -> None:
        scored = [(t, self._features(t)) for t in train_traces]
        scored = [(t, f) for t, f in scored if f is not None]
        if not scored:
            return
        X = [f[0] for _, f in scored]
        y = [1 if t.ground_truth_label else 0 for t, _ in scored]
        if len(set(y)) < 2:
            return  # degenerate pool this generation — keep whatever was fit before
        self._clf = LogisticRegression(class_weight="balanced").fit(X, y)

    def evaluate(self, trace: AttackTrace) -> BlueVerdict:
        result = self._features(trace)
        if result is None:
            return BlueVerdict(
                trace_id=trace.trace_id, risk_score=0.0, predicted_label=False,
                explanation="sequence too short to profile",
            )
        (amount_z, velocity_ratio, drift_fraction, cusum_max_neg), dominant_category, window = result

        if self._clf is None:
            # Unchanged, byte-for-byte, IN LOGIC from the pre-fix formula — the
            # three inputs now come from the frozen-baseline split (a
            # deliberate bug fix, see module docstring). Verified via
            # phase3_reproduction.py that OUTPUT is also unchanged: the fixed
            # thresholds are coarse enough that low_and_slow's corrected
            # z-scores still don't cross them, either way. cusum_max_neg is
            # computed above but never read on this branch.
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
        else:
            feats = [amount_z, velocity_ratio, drift_fraction, cusum_max_neg]
            risk_score = float(self._clf.predict_proba([feats])[0][1])
            predicted_label = bool(self._clf.predict([feats])[0])
            failed = []
            if abs(amount_z) > Z_THRESHOLD:
                failed.append(f"amount_z::{amount_z:.2f} exceeds threshold {Z_THRESHOLD}")
            if velocity_ratio > VELOCITY_RATIO_THRESHOLD:
                failed.append(f"velocity_ratio::recent cadence {velocity_ratio:.1f}x faster than baseline")
            if drift_fraction > CATEGORY_DRIFT_THRESHOLD:
                failed.append(f"category_drift::{drift_fraction:.0%} outside dominant category {dominant_category!r}")
            if cusum_max_neg < -3.0:  # display-only threshold, not what drives predicted_label
                failed.append(f"cusum_max_neg::{cusum_max_neg:.2f} sustained below-baseline drift")

        explanation = (
            "sequence anomaly detected: " + "; ".join(failed)
            if predicted_label
            else f"recent {window} transactions consistent with baseline profile"
        )

        return BlueVerdict(
            trace_id=trace.trace_id, risk_score=risk_score, predicted_label=predicted_label,
            triggered_checks=failed, explanation=explanation,
        )
