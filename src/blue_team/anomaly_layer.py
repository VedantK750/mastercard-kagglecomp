"""One-class (null-calibrated) anomaly scoring — the generalization floor.

WHY THIS EXISTS. The supervised detectors in this package learn a boundary
from attack labels. That is provably insufficient for cross-strategy
generalization here, and the reason is identifiability, not model capacity:
in a training pool of `credential_ato` + `sequence_shift` only, the
`cusum_norm` mechanism carries no label information at all (credential_ato's
old one-sided value sat at ~0.0, indistinguishable from benign; benign's
values actually SPANNED and EXCEEDED sequence_shift's). The coefficient on
the one feature that detects `low_and_slow` is therefore unidentified — its
fitted value is set by sampling noise. That is why the same held-out
experiment returned 0/13 in one pool composition and 13/15 in another: we
were reading a coefficient-sign lottery as a result. No supervised model —
logistic regression, gradient boosting, or a neural network — can recover a
coefficient for a feature that is label-uninformative in its training set.

This layer sidesteps that entirely by never looking at attack labels. It
models NULL (attack-free) behavior only and flags deviation from it, so an
attack strategy it has never seen is scored on exactly the same footing as
one it has. Cross-strategy generalization becomes a property of the
construction rather than something we hope transfers.

DESIGN NOTES, each of which is a real problem found empirically:

- Two-sided inputs. Consumes `SequenceAnomalyDetector.mechanism_scores()`,
  whose entries are all >= 0 and direction-agnostic, so one calibration
  covers an upward burst and a downward drain alike.
- Bonferroni. Scoring m mechanisms and taking the max is an uncorrected
  multiple comparison; a nominal 5% per-mechanism rate measured 8-12%
  family-wise in testing. Per-mechanism quantiles are taken at
  1 - alpha/m so the FAMILY-wise rate is ~alpha.
- Degenerate nulls. `drift_fraction` is identically 0.0 across every benign
  trace this simulator produces, so a purely empirical quantile is 0.0 and
  ratio-scaling against it divides by zero (in testing this silently
  dropped `sequence_shift` detection to 32%). Each mechanism therefore has
  an absolute MINIMUM_MEANINGFUL floor, and the effective threshold is
  max(empirical_quantile, floor). The floors are set on DOMAIN grounds
  (below), never tuned against attack data — tuning them on attacks would
  reintroduce exactly the label dependence this layer exists to avoid.
- Length. `cusum_norm` is already sqrt(n)-normalized upstream because the
  null accumulator grows ~sqrt(n) and this simulator's benign traces (14
  txns) are shorter than low_and_slow (23). Without that, a threshold
  calibrated on benign would not transfer across lengths and the statistic
  would partly encode trace length, which correlates with the label.

WHAT THIS DOES NOT DO. It is an unsupervised deviation detector, so it
cannot distinguish "unusual" from "malicious" — a genuine change in a
user's spending habits scores the same as a slow drain. In production that
implies a re-baselining process; here it means the false-positive rate is a
real cost to report, not a rounding error to hide.
"""

from __future__ import annotations

import statistics
from typing import Dict, List, Optional

from src.common.schemas import AttackTrace, BlueVerdict

# Smallest deviation per mechanism that is meaningful on DOMAIN grounds,
# independent of any observed attack. These are floors on the threshold
# (they make detection HARDER, never easier) and exist so a degenerate
# all-zero null can't make every trace look anomalous.
MINIMUM_MEANINGFUL: Dict[str, float] = {
    "pooled_shift_z": 2.0,      # two-sample z of a whole-tail level shift
    "cusum_norm": 1.0,          # one sigma-equivalent of sustained drift per sqrt(txn)
    "amount_z_abs": 1.0,        # one sigma of level shift
    "velocity_log_abs": 0.223,  # |log(1.25)| — a 25% cadence change either way
    "drift_fraction": 0.25,     # a quarter of the tail outside the usual category
    "persistence_frac": 0.60,   # 60% of the tail beyond 1 sigma (benign runs ~43%)
}

DEFAULT_ALPHA = 0.05  # target FAMILY-wise false-positive rate across all mechanisms

# Multiplier on the scaled MAD for the small-sample robust bound. 3.0 is the
# conventional outlier cut (~99.7th percentile for a normal, and MORE
# conservative than that for the right-skewed distributions these mechanism
# scores actually have, which is the direction we want).
ROBUST_K = 3.0


def _robust_upper_bound(values: List[float], k: float = ROBUST_K) -> float:
    """median + k * 1.4826 * MAD — a stable upper bound that uses EVERY
    sample instead of a single extreme order statistic, so it degrades
    gracefully when the calibration set is small. 1.4826 rescales MAD to be
    a consistent estimator of sigma under normality.

    Falls back to the sample max when MAD is zero (a degenerate mechanism
    like drift_fraction, which is identically 0.0 across every benign trace
    this simulator produces) — in that case the MINIMUM_MEANINGFUL floor is
    what actually governs, which is exactly its purpose."""
    if not values:
        return 0.0
    med = statistics.median(values)
    mad = statistics.median([abs(v - med) for v in values])
    if mad <= 0:
        return max(values)
    return med + k * 1.4826 * mad


class MechanismAnomalyLayer:
    """Calibrate on null traces, then score any trace by its largest
    normalized mechanism deviation. `score >= 1.0` means at least one
    mechanism exceeded its calibrated threshold."""

    def __init__(self, alpha: float = DEFAULT_ALPHA) -> None:
        self.alpha = alpha
        self.thresholds: Dict[str, float] = {}
        self.n_calibration = 0
        # False until calibrated with enough samples for the empirical
        # quantile to be meaningful; surfaced so a caller/report can say so
        # rather than quietly presenting a small-sample threshold as if it
        # were a real 99th percentile.
        self.sample_sufficient = False

    def calibrate(self, null_scores: List[Dict[str, float]]) -> None:
        """`null_scores` are mechanism_scores() dicts from ATTACK-FREE traces
        only. Passing anything attack-derived here defeats the entire point
        of this layer — callers are responsible for that filtering, and the
        experiment harness asserts it.

        SMALL-SAMPLE HANDLING. An empirical quantile cannot estimate the
        q-th quantile from fewer than ~1/(1-q) samples: with m=6 mechanisms
        and alpha=0.05, q=0.9917 needs n>=120, but the live loop was calling
        this with 6 benign traces. `int(0.9917 * 6) = 5` simply returns the
        MAXIMUM of 6 points, which underestimates the true quantile badly and
        was the dominant cause of a measured ~13-20% null false-positive rate
        (7 of 8 false flags came from this layer, not the classifier).

        When the sample is too small, we fall back to a ROBUST estimator that
        uses every point (median + k*MAD) rather than one extreme order
        statistic, and take the larger of the two. Erring toward the higher
        threshold is the correct direction: it costs recall on marginal
        attacks rather than inventing false positives on benign users."""
        if not null_scores:
            return
        names = list(null_scores[0].keys())
        m = len(names)
        # Bonferroni: per-mechanism quantile so the FAMILY-wise rate is ~alpha
        q = 1.0 - (self.alpha / m)
        n = len(null_scores)
        sample_sufficient = n >= (1.0 / (1.0 - q))

        self.thresholds = {}
        for name in names:
            values = sorted(s[name] for s in null_scores)
            idx = min(len(values) - 1, int(q * len(values)))
            empirical = values[idx]

            candidates = [empirical, MINIMUM_MEANINGFUL.get(name, 0.0)]
            if not sample_sufficient:
                candidates.append(_robust_upper_bound(values))
            self.thresholds[name] = max(candidates)

        self.n_calibration = n
        self.sample_sufficient = sample_sufficient

    @property
    def is_calibrated(self) -> bool:
        return bool(self.thresholds)

    def score(self, scores: Dict[str, float]) -> tuple[float, Optional[str]]:
        """Returns (ratio_of_largest_exceedance, which_mechanism)."""
        if not self.is_calibrated:
            return 0.0, None
        best_ratio, best_name = 0.0, None
        for name, value in scores.items():
            threshold = self.thresholds.get(name)
            if not threshold:
                continue
            ratio = value / threshold
            if ratio > best_ratio:
                best_ratio, best_name = ratio, name
        return best_ratio, best_name

    def verdict(self, trace: AttackTrace, scores: Dict[str, float]) -> BlueVerdict:
        ratio, mechanism = self.score(scores)
        flagged = ratio >= 1.0
        # Squash the unbounded ratio into [0,1) for BlueVerdict.risk_score,
        # keeping it monotonic in the ratio so AUC/ranking is unaffected.
        risk = min(0.999, ratio / (1.0 + ratio)) if ratio > 0 else 0.0
        return BlueVerdict(
            trace_id=trace.trace_id,
            risk_score=risk,
            predicted_label=flagged,
            triggered_checks=(
                [f"{mechanism}::{scores[mechanism]:.2f} exceeds null threshold "
                 f"{self.thresholds[mechanism]:.2f} ({ratio:.1f}x)"]
                if flagged and mechanism else []
            ),
            explanation=(
                f"anomaly layer: {mechanism} deviates {ratio:.1f}x beyond the "
                f"null-calibrated threshold (calibrated on {self.n_calibration} attack-free traces)"
                if flagged
                else "anomaly layer: all mechanisms within null-calibrated bounds"
            ),
        )
