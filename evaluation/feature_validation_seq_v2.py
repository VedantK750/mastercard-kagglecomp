"""Phase B.2 — sequence_anomaly longitudinal feature validation (C0-C5).

Follow-up to `evaluation/feature_validation.py` (Phase B), scoped to the one
open blind spot Phase B left unresolved: `low_and_slow` generalization stayed
at 0/13 even after the accepted frozen-baseline fix. Validates three new
longitudinal candidate features (CUSUM, below-baseline-mean fraction/streak,
cumulative dollar-rate ratio) proposed after auditing the schema/generator —
see the diagnostic write-up in this session for the full candidate audit and
why merchant/category diversity, amount variance, and inter-arrival
regularity were ruled out before this file was written (all three are
provably identical between `low_and_slow` and benign by generator
construction, not just uninformative on the current sample).

Same discipline as Phase B: zero new LLM calls, same stable 70/30 split
(`evaluation.adaptive_loop.is_train`), same `evaluation.metrics` functions,
does NOT touch `src/blue_team/sequence_anomaly_detector.py` or
`evaluation/adaptive_loop.py` — this is validation-only. C0 is the Phase-B-
ACCEPTED frozen-baseline 3-feature set (`feat_seq_fixed`, imported unchanged
from `feature_validation.py`), used as the fixed reference point for every
other row.

Run: PYTHONPATH=. .venv/bin/python -m evaluation.feature_validation_seq_v2
"""

from __future__ import annotations

import random
import statistics
from typing import Dict, List, Optional, Tuple

from sklearn.linear_model import LogisticRegression

from evaluation.adaptive_loop import TRACE_PATH, is_train
from evaluation.feature_validation import (
    BASELINE_WINDOW,
    MIN_BASELINE,
    RECENT_WINDOW,
    _frozen_baseline_recent_split,
    feat_seq_fixed,
    generalization_check,
    generate_extra_sequence_samples,
)
from evaluation.metrics import classification_metrics, confusion_breakdown
from src.common.llm_client import get_call_count, reset_call_count
from src.common.schemas import AttackTrace, BlueVerdict
from src.common.trace_io import load_traces
from src.red_team.sequence_anomaly import SequenceAnomalyGenerator

CUSUM_K = 0.5  # standard CUSUM slack constant, in standardized (z-score) units

# ---------------------------------------------------------------------------
# New candidate features (C1-C3), each Optional[List[float]] over the
# ALREADY-ACCEPTED frozen baseline (Phase B, Part 2a) as their reference —
# building on top of accepted work rather than re-deriving a baseline here.
# ---------------------------------------------------------------------------


def cusum_max_neg(trace: AttackTrace) -> Optional[List[float]]:
    """Cumulative sum of standardized deviations below the frozen baseline
    mean, accumulated over every transaction AFTER the baseline (not just a
    trailing window) — the classic control-chart tool for a sustained small
    mean-shift, which a single-window z-score structurally can't detect."""
    txns = sorted(trace.transactions, key=lambda t: t.timestamp)
    if len(txns) < MIN_BASELINE + 1:
        return None
    baseline, _ = _frozen_baseline_recent_split(txns)
    post = txns[len(baseline):]
    if not post:
        return None
    mean = statistics.mean(t.amount for t in baseline)
    std = statistics.pstdev(t.amount for t in baseline) or 1.0
    s_neg = max_neg = 0.0
    for t in post:
        z = (t.amount - mean) / std
        s_neg = min(0.0, s_neg + z + CUSUM_K)
        max_neg = min(max_neg, s_neg)
    return [max_neg]


def below_frac_and_streak(trace: AttackTrace) -> Optional[List[float]]:
    """Fraction of post-baseline transactions below the baseline mean, and
    the longest consecutive run of such transactions — a simpler, more
    interpretable relative of cusum_max_neg (same underlying mechanism:
    directional persistence rather than magnitude)."""
    txns = sorted(trace.transactions, key=lambda t: t.timestamp)
    if len(txns) < MIN_BASELINE + 1:
        return None
    baseline, _ = _frozen_baseline_recent_split(txns)
    post = txns[len(baseline):]
    if not post:
        return None
    mean = statistics.mean(t.amount for t in baseline)
    below_frac = sum(1 for t in post if t.amount < mean) / len(post)
    longest = current = 0
    for t in post:
        if t.amount < mean:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return [below_frac, float(longest)]


def dollar_rate_ratio(trace: AttackTrace) -> Optional[List[float]]:
    """Post-baseline $/hour vs. baseline $/hour — a different axis than
    per-transaction amount: aggregate spend velocity rather than individual
    transaction size."""
    txns = sorted(trace.transactions, key=lambda t: t.timestamp)
    if len(txns) < MIN_BASELINE + 1:
        return None
    baseline, _ = _frozen_baseline_recent_split(txns)
    post = txns[len(baseline):]
    if len(post) < 2 or len(baseline) < 2:
        return None
    baseline_span_hrs = (baseline[-1].timestamp - baseline[0].timestamp).total_seconds() / 3600
    post_span_hrs = (post[-1].timestamp - post[0].timestamp).total_seconds() / 3600
    baseline_rate = sum(t.amount for t in baseline) / max(baseline_span_hrs, 1.0)
    post_rate = sum(t.amount for t in post) / max(post_span_hrs, 1.0)
    return [post_rate / max(baseline_rate, 0.01)]


# ---------------------------------------------------------------------------
# C0-C5 run matrix
# ---------------------------------------------------------------------------


def feat_c0(trace: AttackTrace) -> Optional[List[float]]:  # accepted Phase B fix, unchanged
    return feat_seq_fixed(trace)


def feat_c1(trace: AttackTrace) -> Optional[List[float]]:
    return cusum_max_neg(trace)


def feat_c2(trace: AttackTrace) -> Optional[List[float]]:
    return below_frac_and_streak(trace)


def feat_c3(trace: AttackTrace) -> Optional[List[float]]:
    return dollar_rate_ratio(trace)


def feat_c4(trace: AttackTrace) -> Optional[List[float]]:
    a, b = feat_seq_fixed(trace), cusum_max_neg(trace)
    return None if (a is None or b is None) else a + b


def feat_c5(trace: AttackTrace) -> Optional[List[float]]:
    a, b, c, d = (
        feat_seq_fixed(trace),
        cusum_max_neg(trace),
        below_frac_and_streak(trace),
        dollar_rate_ratio(trace),
    )
    if any(x is None for x in (a, b, c, d)):
        return None
    return a + b + c + d


RUN_MATRIX = [
    ("C0", "frozen-baseline 3 features (accepted, Phase B)", feat_c0),
    ("C1", "cusum_max_neg alone", feat_c1),
    ("C2", "below_frac + longest_streak alone", feat_c2),
    ("C3", "dollar_rate_ratio alone", feat_c3),
    ("C4", "C0 + cusum_max_neg", feat_c4),
    ("C5", "C0 + all three new candidates", feat_c5),
]


def fit_eval_breakdown(
    train: List[AttackTrace], test: List[AttackTrace], feature_fn, label: str
) -> Optional[Tuple[LogisticRegression, object, object]]:
    train_scored = [(t, feature_fn(t)) for t in train]
    train_scored = [(t, f) for t, f in train_scored if f is not None]
    test_scored = [(t, feature_fn(t)) for t in test]
    test_scored = [(t, f) for t, f in test_scored if f is not None]

    if not train_scored or not test_scored:
        print(f"  [{label}] insufficient data (train={len(train_scored)}, test={len(test_scored)}) — skipped")
        return None
    y_train = [1 if t.ground_truth_label else 0 for t, _ in train_scored]
    if len(set(y_train)) < 2:
        print(f"  [{label}] degenerate train pool (single class) — skipped")
        return None

    X_train = [f for _, f in train_scored]
    clf = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X_train, y_train)
    test_traces = [t for t, _ in test_scored]
    verdicts = [
        BlueVerdict(
            trace_id=t.trace_id,
            risk_score=float(clf.predict_proba([f])[0][1]),
            predicted_label=bool(clf.predict([f])[0]),
        )
        for t, f in test_scored
    ]
    cm = classification_metrics(test_traces, verdicts)
    cb = confusion_breakdown(test_traces, verdicts)
    n_excluded = (len(train) - len(train_scored)) + (len(test) - len(test_scored))
    excl = f"  (excluded {n_excluded} traces missing this feature)" if n_excluded else ""
    print(f"  [{label}]")
    print(
        f"    P={cm.precision:.3f}  R={cm.recall:.3f}  F1={cm.f1:.3f}  AUC={cm.auc:.3f}  "
        f"FPR={cb.false_positive_rate:.3f}  CaseC={cb.case_c}{excl}"
    )
    verdict_by_id = {v.trace_id: v for v in verdicts}
    segments: Dict[str, List[AttackTrace]] = {}
    for t in test_traces:
        seg = t.metadata.get("preset") if t.ground_truth_label else "benign"
        segments.setdefault(seg or "unknown", []).append(t)
    for seg in sorted(segments):
        seg_traces = segments[seg]
        seg_verdicts = [verdict_by_id[t.trace_id] for t in seg_traces]
        n_flagged = sum(1 for v in seg_verdicts if v.predicted_label)
        print(f"      {seg:15} n={len(seg_traces):3}  flagged={n_flagged:3}  rate={n_flagged/len(seg_traces):.2f}")
    return clf, cm, cb


def main() -> None:
    reset_call_count()
    traces = load_traces(TRACE_PATH)
    sa = [t for t in traces if t.family == "sequence_anomaly"]
    sa_train = [t for t in sa if is_train(t.trace_id)]
    sa_test = [t for t in sa if not is_train(t.trace_id)]
    print(f"n_train={len(sa_train)} n_test={len(sa_test)} (same stable split as Phase B)")

    print("\n=== C0-C5 ablation ===")
    fitted: Dict[str, Tuple[LogisticRegression, object, object]] = {}
    for code, desc, fn in RUN_MATRIX:
        result = fit_eval_breakdown(sa_train, sa_test, fn, f"{code} {desc}")
        if result is not None:
            fitted[code] = result

    calls = get_call_count()
    print(f"\nLLM calls through ablation: {calls} (must be 0)")

    # -- Incremental-value comparison: C0 vs C4 vs C5 -------------------------------
    print("\n=== C0 vs C4 vs C5 — incremental value check ===")
    for code in ("C0", "C4", "C5"):
        if code in fitted:
            _, cm, cb = fitted[code]
            print(f"  {code}: R={cm.recall:.3f} F1={cm.f1:.3f} AUC={cm.auc:.3f} FPR={cb.false_positive_rate:.3f} CaseC={cb.case_c}")
        else:
            print(f"  {code}: not fitted (see skip reason above)")

    # -- Robustness sweep: recall vs amount_multiplier, models fit above -----------
    print("\n=== Robustness sweep: recall vs amount_multiplier (models trained above, fresh held-out attacks per point, zero LLM calls) ===")
    gen = SequenceAnomalyGenerator()
    seed_ctx = gen.seed()[0]
    sweep = [0.85, 0.90, 0.92, 0.95, 0.97, 1.00]
    random.seed(2026083001)
    header = "  mult   " + "  ".join(f"{c}_recall(n)" for c in ("C0", "C4", "C5"))
    print(header)
    for mult in sweep:
        test_batch = []
        for _ in range(15):
            ctx = dict(seed_ctx)
            ctx["preset"] = "low_and_slow"
            ctx["amount_multiplier"] = mult
            ctx["spacing_multiplier"] = 0.9
            ctx["n_tail_txns"] = 15
            test_batch.append(gen.simulate(ctx, benign=False))
        row = []
        for code, _, fn in [r for r in RUN_MATRIX if r[0] in ("C0", "C4", "C5")]:
            if code not in fitted:
                row.append("n/a")
                continue
            clf, _, _ = fitted[code]
            scored = [(t, fn(t)) for t in test_batch]
            scored = [(t, f) for t, f in scored if f is not None]
            if not scored:
                row.append("n/a")
                continue
            preds = [bool(clf.predict([f])[0]) for _, f in scored]
            recall = sum(preds) / len(preds)
            row.append(f"{recall:.2f}({len(scored)})")
        print(f"  {mult:.2f}   " + "  ".join(row))

    calls_after_sweep = get_call_count()
    print(f"\nLLM calls through robustness sweep: {calls_after_sweep} (must be 0)")

    # -- Corrected generalization: credential_ato+sequence_shift -> unseen low_and_slow --
    print("\n=== Corrected generalization: train credential_ato+sequence_shift -> test unseen low_and_slow ===")
    random.seed(20260830)
    extra = generate_extra_sequence_samples(n_each=16)
    seq_pool = sa + extra
    n_ca = sum(1 for t in seq_pool if t.metadata.get("preset") == "credential_ato")
    n_ss = sum(1 for t in seq_pool if t.metadata.get("preset") == "sequence_shift")
    n_los = sum(1 for t in seq_pool if t.metadata.get("preset") == "low_and_slow")
    print(f"pool composition: credential_ato={n_ca}, sequence_shift={n_ss}, low_and_slow (held out)={n_los}")
    for code, desc, fn in RUN_MATRIX:
        generalization_check(
            seq_pool, {"credential_ato", "sequence_shift"}, {"low_and_slow"}, fn, f"{code} {desc}"
        )

    total_calls = get_call_count()
    print(f"\nTotal LLM calls this entire run: {total_calls} (must be 0)")
    assert total_calls == 0, "Phase B.2 must make zero new LLM calls — investigate before trusting these results"


if __name__ == "__main__":
    main()
