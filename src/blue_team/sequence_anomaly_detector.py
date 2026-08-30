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

import math
import statistics
from collections import Counter
from typing import Dict, List, Optional, Tuple

from sklearn.linear_model import LogisticRegression

from src.blue_team.anomaly_layer import MechanismAnomalyLayer
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

# Feature vector order for the trained path — one entry per attack MECHANISM
# the schema can actually express (amount/category/timestamp are all we have),
# not one per preset. Every entry is two-sided (direction-agnostic) and
# normalized against the user's OWN frozen baseline, so a feature can't be
# blind to an attack that moves the opposite way from the one it was designed
# against — the exact defect the old one-sided cusum_max_neg had against
# credential_ato's upward burst.
MECHANISM_NAMES = (
    "pooled_shift_z", "cusum_norm", "amount_z_abs",
    "velocity_log_abs", "drift_fraction", "persistence_frac",
)


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


def _cusum_two_sided(
    baseline: List[Transaction], post: List[Transaction], k: float = CUSUM_K
) -> Tuple[float, float]:
    """Returns (max_pos, max_neg) — BOTH accumulator arms, not just the
    downward one. The prior single-arm version was structurally blind to
    credential_ato (an upward burst): its max_neg sat at ~0.0, identical to
    benign, so the feature could not see one of the three attack presets at
    all. Resets toward 0 on any transaction crossing back over the mean — a
    known, documented weakness (Red could interleave above-mean decoys to
    hold the accumulator near zero), not claimed away."""
    if not post:
        return 0.0, 0.0
    mean = statistics.mean(t.amount for t in baseline)
    std = statistics.pstdev(t.amount for t in baseline) or 1.0
    s_pos = s_neg = max_pos = max_neg = 0.0
    for t in post:
        z = (t.amount - mean) / std
        s_pos = max(0.0, s_pos + z - k)
        s_neg = min(0.0, s_neg + z + k)
        max_pos = max(max_pos, s_pos)
        max_neg = min(max_neg, s_neg)
    return max_pos, max_neg


def _pooled_shift_z(baseline: List[Transaction], post: List[Transaction]) -> float:
    """Two-sample z of the post-baseline MEAN against the frozen-baseline
    mean — |mean_post - mean_base| / (sigma_base * sqrt(1/n_post + 1/n_base)).

    This is the statistically optimal test for the specific thing
    `low_and_slow` does (a sustained constant level shift over a known
    split), and it strictly dominates `cusum_norm` for that case. CUSUM is
    designed to find a change-point at an UNKNOWN time in a stream and pays
    a real power penalty for that generality; here the frozen-baseline split
    already tells us where the boundary is, so we should not pay it.

    Both are kept: CUSUM still wins when a shift is confined to part of the
    tail (it maximizes over sub-runs) while this maximizes power against a
    whole-tail shift — they are complementary, not redundant.

    Validated against closed-form power analysis: predicted vs. measured
    detection at Bonferroni z=2.64 agrees within a few points across
    history lengths (0.90 multiplier at n_base=8: 18% predicted / 16%
    measured; at n_base=60: ~99% / 98%). That agreement is the evidence
    this feature is at the information ceiling, which in turn is what lets
    us attribute the residual low_and_slow blind spot to the simulator's
    8-transaction baseline rather than to detector design."""
    if len(post) < 2 or len(baseline) < 2:
        return 0.0
    mean_b = statistics.mean(t.amount for t in baseline)
    mean_p = statistics.mean(t.amount for t in post)
    std_b = statistics.pstdev(t.amount for t in baseline) or 1.0
    se = std_b * math.sqrt(1.0 / len(post) + 1.0 / len(baseline))
    return abs(mean_p - mean_b) / max(se, 1e-9)


def _cusum_norm(baseline: List[Transaction], post: List[Transaction]) -> float:
    """Two-sided CUSUM magnitude, normalized by sqrt(len(post)).

    The sqrt divisor matters and isn't cosmetic: under the null the
    accumulator behaves like a bounded random walk whose extreme grows ~sqrt(n),
    so an UN-normalized cusum partially encodes sequence LENGTH. In this
    simulator benign traces are always 14 transactions while low_and_slow is
    23, so length correlates with the label — an unnormalized statistic would
    be silently rewarded for reading trace length rather than behavior, and a
    threshold calibrated on 14-txn benign would not transfer to a 23-txn
    attack. Dividing by sqrt(n_post) makes the null approximately
    length-invariant, which is what lets one calibration serve all lengths."""
    if not post:
        return 0.0
    max_pos, max_neg = _cusum_two_sided(baseline, post)
    return max(max_pos, abs(max_neg)) / math.sqrt(len(post))


class SequenceAnomalyDetector(Detector):
    family = "sequence_anomaly"
    trainable = True

    def __init__(self) -> None:
        self._clf: Optional[LogisticRegression] = None
        self._anomaly = MechanismAnomalyLayer()

    def mechanism_scores(self, trace: AttackTrace) -> Optional[Dict[str, float]]:
        """The five two-sided, baseline-normalized mechanism deviations
        (MECHANISM_NAMES), or None if the sequence is too short to profile.

        Every value is >= 0 and direction-agnostic: "how far from this user's
        own established baseline, in any direction," never "higher/lower than
        baseline." That's what lets ONE calibration cover an upward burst
        (credential_ato) and a downward drain (low_and_slow) with the same
        statistic, and it's why the anomaly layer (src/blue_team/anomaly_layer.py)
        can score an attack strategy it was never trained on.

        Shared by both the trained path here and that one-class layer, so the
        two views of a trace are always computed from identical inputs."""
        txns = sorted(trace.transactions, key=lambda t: t.timestamp)
        if len(txns) < MIN_BASELINE + 1:
            return None

        baseline, recent = _frozen_baseline_recent_split(txns)
        post = txns[len(baseline):]  # everything after the frozen baseline

        amounts = [t.amount for t in baseline]
        mean_amt = statistics.mean(amounts)
        std_amt = statistics.pstdev(amounts) or 1.0

        recent_mean_amt = statistics.mean(t.amount for t in recent)
        amount_z_abs = abs((recent_mean_amt - mean_amt) / std_amt)

        baseline_gap = _avg_gap_seconds(baseline)
        recent_gap = _avg_gap_seconds(recent)
        # |log ratio| — symmetric in speed-up vs slow-down, where the raw
        # ratio was not (2x faster = 2.0, 2x slower = 0.5, unequal distances
        # from the no-change point of 1.0).
        velocity_log_abs = (
            abs(math.log(baseline_gap / recent_gap))
            if (baseline_gap and recent_gap and recent_gap > 0 and baseline_gap > 0)
            else 0.0
        )

        dominant_category, _ = Counter(t.category for t in baseline).most_common(1)[0]
        drift_fraction = sum(1 for t in post if t.category != dominant_category) / len(post)

        # Fraction of post-baseline transactions more than 1 sigma from the
        # baseline mean in EITHER direction. Complements cusum_norm: cusum
        # measures how far a sustained run drifts, this measures how much of
        # the tail participates — a short violent excursion and a long quiet
        # one separate on these two differently.
        persistence_frac = sum(1 for t in post if abs((t.amount - mean_amt) / std_amt) > 1.0) / len(post)

        return {
            "pooled_shift_z": _pooled_shift_z(baseline, post),
            "cusum_norm": _cusum_norm(baseline, post),
            "amount_z_abs": amount_z_abs,
            "velocity_log_abs": velocity_log_abs,
            "drift_fraction": drift_fraction,
            "persistence_frac": persistence_frac,
        }

    def _features(self, trace: AttackTrace) -> Optional[Tuple[List[float], str, int]]:
        """Returns (feature_vector_in_MECHANISM_NAMES_order, dominant_category,
        recent_window_size) or None if too short to profile."""
        scores = self.mechanism_scores(trace)
        if scores is None:
            return None
        txns = sorted(trace.transactions, key=lambda t: t.timestamp)
        baseline, recent = _frozen_baseline_recent_split(txns)
        dominant_category, _ = Counter(t.category for t in baseline).most_common(1)[0]
        return [scores[name] for name in MECHANISM_NAMES], dominant_category, len(recent)

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

    def calibrate(self, null_traces: List[AttackTrace]) -> None:
        """Fit the one-class layer's thresholds on ATTACK-FREE traces only.

        This is what gives the detector a cross-strategy floor. Measured in
        evaluation/generalization_suite.py: trained on credential_ato +
        sequence_shift only, the supervised path scores 0.00 recall on
        held-out low_and_slow (identifiability — the deciding coefficient is
        unconstrained by that pool), while this layer, having seen no attack
        labels at all, still scores 0.30/0.15 at strengths 0.85/0.90 and
        1.00 on both loud strategies."""
        assert all(not t.ground_truth_label for t in null_traces), (
            "calibrate() received an attack trace — the one-class layer must "
            "never see attack labels or it loses its cross-strategy property"
        )
        scores = [self.mechanism_scores(t) for t in null_traces]
        self._anomaly.calibrate([s for s in scores if s is not None])

    def _legacy_heuristic_checks(self, trace: AttackTrace) -> Tuple[List[str], str]:
        """The pre-learning fixed-threshold rule, FROZEN — signed amount_z and
        the signed one-sided velocity ratio, exactly as originally written.

        Kept separate from mechanism_scores() on purpose: those are now
        two-sided (|amount_z|, |log velocity|), and feeding them to this rule
        would silently CHANGE Generation-0 behavior — |log ratio| > log(2)
        also fires on a 2x SLOWDOWN, which the original `ratio > 2.0` never
        did. Isolating the legacy inputs here is what keeps every committed
        reproduction script's numbers byte-identical while the learned path
        moves to the better representation."""
        txns = sorted(trace.transactions, key=lambda t: t.timestamp)
        baseline, recent = _frozen_baseline_recent_split(txns)
        amounts = [t.amount for t in baseline]
        mean_amt = statistics.mean(amounts)
        std_amt = statistics.pstdev(amounts) or 1.0
        amount_z = (statistics.mean(t.amount for t in recent) - mean_amt) / std_amt

        baseline_gap = _avg_gap_seconds(baseline)
        recent_gap = _avg_gap_seconds(recent)
        velocity_ratio = (
            baseline_gap / recent_gap if (baseline_gap and recent_gap and recent_gap > 0) else 1.0
        )
        dominant_category, _ = Counter(t.category for t in baseline).most_common(1)[0]
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
        return failed, dominant_category

    def evaluate(self, trace: AttackTrace) -> BlueVerdict:
        result = self._features(trace)
        if result is None:
            return BlueVerdict(
                trace_id=trace.trace_id, risk_score=0.0, predicted_label=False,
                explanation="sequence too short to profile",
            )
        feats, dominant_category, window = result
        failed, _ = self._legacy_heuristic_checks(trace)

        if self._clf is None and not self._anomaly.is_calibrated:
            # Generation-0: the frozen legacy rule above, unchanged.
            risk_score = min(1.0, round(len(failed) / 3, 4))
            predicted_label = len(failed) > 0
        elif self._clf is None:
            # Calibrated but never fit — pure one-class. This is the correct
            # state for a cross-strategy evaluation, where training on the
            # held-out strategy is by definition not allowed.
            scores = dict(zip(MECHANISM_NAMES, feats))
            verdict = self._anomaly.verdict(trace, scores)
            return verdict
        else:
            risk_score = float(self._clf.predict_proba([feats])[0][1])
            predicted_label = bool(self._clf.predict([feats])[0])
            # Display-only: name whichever mechanisms are large, so a verdict
            # is explainable. These do NOT drive predicted_label (the fitted
            # classifier does) — they're the reason string, and they feed
            # AttackMemory.detection_reasons so Red's mutate() knows which
            # lever to move.
            named = dict(zip(MECHANISM_NAMES, feats))
            if named["cusum_norm"] > 2.0:
                failed.append(f"cusum_norm::{named['cusum_norm']:.2f} sustained deviation from baseline")
            if named["persistence_frac"] > 0.5:
                failed.append(f"persistence_frac::{named['persistence_frac']:.0%} of tail beyond 1 sigma")
            if named["velocity_log_abs"] > 0.15:
                failed.append(f"velocity_ratio::cadence shifted {named['velocity_log_abs']:.2f} log-units")

            # HYBRID: the two views are complementary, not competing, and the
            # generalization suite quantifies exactly how. Supervised is far
            # stronger on strategies it has labels for (strength-generalization
            # recall 0.72 vs the one-class layer's 0.17); the one-class layer
            # is the ONLY one with any signal on a strategy absent from
            # training (0.30/0.15 vs 0.00/0.00). OR-ing them keeps each
            # where it wins, at the cost of union false positives — a real
            # cost, reported, not hidden.
            if self._anomaly.is_calibrated:
                a_ratio, a_mech = self._anomaly.score(named)
                if a_ratio >= 1.0:
                    predicted_label = True
                    risk_score = max(risk_score, min(0.999, a_ratio / (1.0 + a_ratio)))
                    failed.append(
                        f"anomaly_layer::{a_mech} {a_ratio:.1f}x beyond null-calibrated threshold"
                    )

        explanation = (
            "sequence anomaly detected: " + "; ".join(failed)
            if predicted_label
            else f"recent {window} transactions consistent with baseline profile"
        )

        return BlueVerdict(
            trace_id=trace.trace_id, risk_score=risk_score, predicted_label=predicted_label,
            triggered_checks=failed, explanation=explanation,
        )
