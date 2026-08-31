"""Phase B — feature validation harness (Addendum 2 to the master plan,
`/home/krish/.claude/plans/1-code-repository-purring-graham.md`).

Validates three candidate detector features against the EXISTING, already-
collected 126-trace dataset (`traces/adaptive_loop_traces.jsonl`) with ZERO
new LLM calls. Does NOT modify any `src/blue_team/*.py` file — every run
below computes a standalone feature vector from data already on each
`AttackTrace` (reusing existing detector helper methods read-only where
possible: `ReasoningAttackDetector._features`,
`IntentManipulationDetector._match_score`, `sequence_anomaly_detector._avg_gap_seconds`),
fits a throwaway `LogisticRegression`, and scores with the exact same
`evaluation.metrics` functions and the exact same stable 70/30 split
(`evaluation.adaptive_loop.is_train`) the live adaptive loop uses. Nothing
here is wired into the live detectors — that's Phase C, gated on this
script's results being reviewed and specific features being approved.

Run: PYTHONPATH=. .venv/bin/python -m evaluation.feature_validation
"""

from __future__ import annotations

import random
import statistics
from collections import Counter
from typing import Callable, List, Optional

from sklearn.linear_model import LogisticRegression

from evaluation.adaptive_loop import TRACE_PATH, is_train
from evaluation.metrics import ClassificationResult, ConfusionBreakdown, classification_metrics, confusion_breakdown
from src.blue_team.intent_manipulation_detector import IntentManipulationDetector
from src.blue_team.reasoning_attack_detector import ReasoningAttackDetector
from src.blue_team.sequence_anomaly_detector import _avg_gap_seconds
from src.common.llm_client import get_call_count, reset_call_count
from src.common.schemas import AttackTrace, BlueVerdict
from src.common.trace_io import load_traces
from src.red_team.sequence_anomaly import PRESETS, SequenceAnomalyGenerator
from src.red_team.vault_whisper import FRAMING_TECHNIQUES

FeatureFn = Callable[[AttackTrace], Optional[List[float]]]

_ra_detector = ReasoningAttackDetector()
_im_detector = IntentManipulationDetector()

# ---------------------------------------------------------------------------
# Reasoning attack (Vault Whisper) — Part 5 run matrix R1/R2/R3
# ---------------------------------------------------------------------------


def feat_ra_keyword(trace: AttackTrace) -> List[float]:
    # Take only the leading keyword pair: the live detector's _features() has
    # since grown identity_mismatch as a third entry (accepted in Phase C),
    # but R1 here is deliberately the PRE-Phase-C baseline this script exists
    # to measure against, so it must stay the original two features.
    feats, _ = _ra_detector._features(trace)
    return list(feats[:2])


def feat_ra_identity(trace: AttackTrace) -> List[float]:
    req = trace.metadata.get("requested_user_id")
    sess = trace.metadata.get("session_user_id")
    return [1.0 if (req is not None and req != sess) else 0.0]


def feat_ra_combined(trace: AttackTrace) -> List[float]:
    return feat_ra_keyword(trace) + feat_ra_identity(trace)


# ---------------------------------------------------------------------------
# Sequence anomaly — Part 5 run matrix A1/A2(=B1)/B2/B3
# Mirrors SequenceAnomalyDetector._features() exactly for the "buggy" (today,
# live) split; the "frozen" split is the Part 2a bug fix under validation.
# ---------------------------------------------------------------------------

WINDOW = 5
MIN_BASELINE = 3
BASELINE_WINDOW = 8
RECENT_WINDOW = 5


def _frozen_baseline_recent_split(txns: List) -> tuple[List, List]:
    """Anchors baseline at the sequence START and caps it at BASELINE_WINDOW
    — it never slides forward or grows as the tail lengthens, which is the
    actual fix for the contamination bug (a long low_and_slow tail can no
    longer leak into the reference window). RECENT_WINDOW adapts DOWN (never
    below 1) only when the whole sequence is too short to fit a full
    baseline + a full recent window side by side (credential_ato's default
    preset produces only 8 baseline + 4 tail = 12 total, below the naive
    8+5=13 floor) — this only ever shrinks `recent`, never re-widens
    `baseline` into the tail, so the fix's guarantee is preserved for every
    preset length actually produced by the generator, not just low_and_slow."""
    recent_n = min(RECENT_WINDOW, max(1, len(txns) - MIN_BASELINE))
    baseline_n = min(BASELINE_WINDOW, len(txns) - recent_n)
    return txns[:baseline_n], txns[-recent_n:]


def _seq_point(trace: AttackTrace, frozen: bool) -> Optional[List[float]]:
    txns = sorted(trace.transactions, key=lambda t: t.timestamp)
    if len(txns) < MIN_BASELINE + 1:
        return None
    if frozen:
        baseline, recent = _frozen_baseline_recent_split(txns)
    else:
        window = max(min(WINDOW, len(txns) - MIN_BASELINE), 1)
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

    dominant_category, _ = Counter(t.category for t in baseline).most_common(1)[0]
    drift_fraction = sum(1 for t in recent if t.category != dominant_category) / len(recent)
    return [amount_z, velocity_ratio, drift_fraction]


def _rolling_mean_slope(trace: AttackTrace, window: int = 5) -> Optional[float]:
    """Linear-regression slope of a rolling mean of amount over the FULL
    sequence, normalized by the frozen-baseline mean amount (scale-free,
    ~fractional drift per transaction). Deliberately NOT split into
    baseline/recent — it's a single continuous trend estimate, immune to the
    window-boundary contamination the point features above are affected by
    in "buggy" mode."""
    txns = sorted(trace.transactions, key=lambda t: t.timestamp)
    if len(txns) < MIN_BASELINE + 1:
        return None
    baseline, _ = _frozen_baseline_recent_split(txns)
    amounts = [t.amount for t in txns]
    rolling = [statistics.mean(amounts[max(0, i - window + 1) : i + 1]) for i in range(len(amounts))]
    n = len(rolling)
    xs = list(range(n))
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(rolling)
    denom = sum((x - x_mean) ** 2 for x in xs)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, rolling)) / denom if denom else 0.0
    baseline_mean = statistics.mean(t.amount for t in baseline)
    return slope / max(baseline_mean, 1.0)


def feat_seq_buggy(trace: AttackTrace) -> Optional[List[float]]:  # A1 — today's live code, reproduced
    return _seq_point(trace, frozen=False)


def feat_seq_fixed(trace: AttackTrace) -> Optional[List[float]]:  # A2 / B1 — frozen baseline
    return _seq_point(trace, frozen=True)


def feat_seq_slope_only(trace: AttackTrace) -> Optional[List[float]]:  # B2
    s = _rolling_mean_slope(trace)
    return None if s is None else [s]


def feat_seq_fixed_plus_slope(trace: AttackTrace) -> Optional[List[float]]:  # B3
    base = _seq_point(trace, frozen=True)
    s = _rolling_mean_slope(trace)
    if base is None or s is None:
        return None
    return base + [s]


# ---------------------------------------------------------------------------
# Intent manipulation — Part 5 run matrix I1/I2/I3 (attempt-detection track)
# ---------------------------------------------------------------------------


def feat_im_outcome(trace: AttackTrace) -> Optional[List[float]]:
    score = _im_detector._match_score(trace)
    return None if score is None else [1 - score]


def feat_im_decoy_reached(trace: AttackTrace) -> List[float]:
    return [1.0 if trace.metadata.get("decoy_reached_agent") else 0.0]


def feat_im_combined(trace: AttackTrace) -> Optional[List[float]]:
    score = _im_detector._match_score(trace)
    if score is None:
        return None
    return [1 - score, 1.0 if trace.metadata.get("decoy_reached_agent") else 0.0]


# ---------------------------------------------------------------------------
# Generic fit/eval — same LogisticRegression config, same evaluation.metrics
# functions the live adaptive loop uses; only the feature vector changes.
# ---------------------------------------------------------------------------


def fit_eval(
    train: List[AttackTrace], test: List[AttackTrace], feature_fn: FeatureFn, label: str
) -> Optional[tuple[ClassificationResult, ConfusionBreakdown]]:
    train_scored = [(t, feature_fn(t)) for t in train]
    train_scored = [(t, f) for t, f in train_scored if f is not None]
    test_scored = [(t, feature_fn(t)) for t in test]
    test_scored = [(t, f) for t, f in test_scored if f is not None]

    if not train_scored or not test_scored:
        print(f"  [{label}] insufficient data (train={len(train_scored)}, test={len(test_scored)}) — skipped")
        return None

    X_train = [f for _, f in train_scored]
    y_train = [1 if t.ground_truth_label else 0 for t, _ in train_scored]
    if len(set(y_train)) < 2:
        print(f"  [{label}] degenerate train pool (single class) — skipped")
        return None

    clf = LogisticRegression(class_weight="balanced").fit(X_train, y_train)
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
    excl_note = f"  (excluded {n_excluded} traces missing this feature)" if n_excluded else ""
    print(f"  [{label}]")
    print(f"    {cm}")
    print(f"    {cb}{excl_note}")
    return cm, cb


def generalization_check(
    pool: List[AttackTrace],
    held_in,
    held_out,
    feature_fn: FeatureFn,
    label: str,
    key: str = "preset",
) -> None:
    train = [t for t in pool if (not t.ground_truth_label) or (t.metadata.get(key) in held_in)]
    test = [t for t in pool if t.ground_truth_label and t.metadata.get(key) in held_out]
    fit_eval(train, test, feature_fn, label)


def generate_extra_sequence_samples(n_each: int = 16) -> List[AttackTrace]:
    """credential_ato + sequence_shift are underrepresented in the collected
    126-trace set (0-1 sequence_shift examples across prior runs) — too thin
    to populate a fair held-in training pool for the corrected Part 6
    generalization check. SequenceAnomalyGenerator has no LLM dependency, so
    generating more costs nothing; reuses the REAL generator (seed -> mutate
    -> simulate) rather than hand-rolling parameter dicts, so these are
    genuine (if extra) generator outputs, not synthetic shortcuts. Kept
    in-memory only — never written to traces/adaptive_loop_traces.jsonl, to
    avoid conflating with the 126-trace baseline used everywhere else."""
    gen = SequenceAnomalyGenerator()
    base_seed = gen.seed()[0]
    out: List[AttackTrace] = []
    for preset in ("credential_ato", "sequence_shift"):
        for _ in range(n_each):
            ctx = dict(base_seed)
            ctx["preset"] = preset
            ctx["_lock_preset"] = True   # controlled experiment: keep requested preset
            ctx["avg_amount"] = base_seed["avg_amount"] * random.uniform(0.85, 1.15)
            ctx["velocity_hours"] = base_seed["velocity_hours"] * random.uniform(0.85, 1.15)
            ctx = gen.mutate(ctx)
            out.append(gen.simulate(ctx, benign=False))
    return out


def main() -> None:
    reset_call_count()
    traces = load_traces(TRACE_PATH)
    train_all = [t for t in traces if is_train(t.trace_id)]
    test_all = [t for t in traces if not is_train(t.trace_id)]
    print(
        f"Loaded {len(traces)} traces ({len(train_all)} train / {len(test_all)} test — "
        f"stable split reused byte-for-byte from evaluation.adaptive_loop.is_train)"
    )

    # -- Part 5: reasoning_attack / Vault Whisper -----------------------------------
    print("\n=== Part 5 — Reasoning attack (Vault Whisper) ===")
    ra_train = [t for t in train_all if t.family == "reasoning_attack"]
    ra_test = [t for t in test_all if t.family == "reasoning_attack"]
    print(f"n_train={len(ra_train)} n_test={len(ra_test)}")
    fit_eval(ra_train, ra_test, feat_ra_keyword, "R1 [max_hits, total_hits] (today)")
    fit_eval(ra_train, ra_test, feat_ra_identity, "R2 [identity_mismatch]")
    fit_eval(ra_train, ra_test, feat_ra_combined, "R3 [max_hits, total_hits, identity_mismatch]")

    vw = [t for t in traces if t.sub_attack == "vault_whisper"]
    case_c_today = sum(1 for t in vw if t.ground_truth_label and t.cross_user_exposure)
    would_flip = sum(
        1
        for t in vw
        if t.ground_truth_label
        and t.cross_user_exposure
        and (t.metadata.get("requested_user_id") != t.metadata.get("session_user_id"))
    )
    print(
        f"  Direct check (Part 1 table, not model-mediated): {case_c_today}/{len(vw)} Vault Whisper "
        f"traces succeeded (all Case C today, since [0,0] never flags anything) -> "
        f"identity_mismatch alone would flag {would_flip}/{case_c_today} of them"
    )

    print("\n  -- bonus: does identity_mismatch generalize across held-out framing techniques? --")
    # VaultWhisperGenerator never produces benign traces (no benign=True path
    # in its simulate()) — a vault_whisper-only pool has zero negatives, so
    # fit() would see one class and degenerate. Pull benign examples from the
    # wider reasoning_attack family (branded_whisper's benign traces) instead,
    # same as what the live ReasoningAttackDetector.fit() would see; keep the
    # attack side restricted to vault_whisper, since identity_mismatch has no
    # meaning for branded_whisper (defaults to 0, would just add label noise).
    ra_benign = [t for t in traces if t.family == "reasoning_attack" and not t.ground_truth_label]
    ra_held_in = {FRAMING_TECHNIQUES[0], FRAMING_TECHNIQUES[1]}
    ra_held_out = set(FRAMING_TECHNIQUES) - ra_held_in
    generalization_check(
        vw + ra_benign, ra_held_in, ra_held_out, feat_ra_identity, "identity_mismatch only", key="technique"
    )

    # -- Part 5: sequence_anomaly ----------------------------------------------------
    print("\n=== Part 5 — Sequence anomaly ===")
    sa_train = [t for t in train_all if t.family == "sequence_anomaly"]
    sa_test = [t for t in test_all if t.family == "sequence_anomaly"]
    print(f"n_train={len(sa_train)} n_test={len(sa_test)}")
    fit_eval(sa_train, sa_test, feat_seq_buggy, "A1 buggy split, [amount_z,velocity_ratio,drift_frac] (today)")
    fit_eval(sa_train, sa_test, feat_seq_fixed, "A2/B1 frozen-baseline split, same 3 features")
    fit_eval(sa_train, sa_test, feat_seq_slope_only, "B2 [rolling_mean_slope] alone")
    fit_eval(sa_train, sa_test, feat_seq_fixed_plus_slope, "B3 frozen-baseline 3 features + rolling_mean_slope")

    # -- Part 5: intent_manipulation --------------------------------------------------
    print(
        "\n=== Part 5 — Intent manipulation (attempt-detection track — Case-C is 0% "
        "by construction for every run below; 'recall' here IS attempt-detection recall) ==="
    )
    im_train = [t for t in train_all if t.family == "intent_manipulation"]
    im_test = [t for t in test_all if t.family == "intent_manipulation"]
    print(f"n_train={len(im_train)} n_test={len(im_test)}")
    im_all = [t for t in traces if t.family == "intent_manipulation"]
    n_attempts = sum(1 for t in im_all if t.ground_truth_label)
    n_reached = sum(1 for t in im_all if t.metadata.get("decoy_reached_agent"))
    n_succeeded = sum(1 for t in im_all if t.metadata.get("attack_succeeded"))
    print(f"  Funnel (Part 3): attempts={n_attempts}, reached_agent={n_reached}, succeeded={n_succeeded}")
    fit_eval(im_train, im_test, feat_im_outcome, "I1 [1-intent_match_score] (today)")
    fit_eval(im_train, im_test, feat_im_decoy_reached, "I2 [decoy_reached_agent]")
    fit_eval(im_train, im_test, feat_im_combined, "I3 [1-intent_match_score, decoy_reached_agent]")

    calls_after_part5 = get_call_count()
    print(f"\nLLM calls through Part 5: {calls_after_part5} (must be 0)")

    # -- Part 6: corrected sequence_anomaly generalization ----------------------------
    print("\n=== Part 6 — Corrected sequence_anomaly generalization (train credential_ato+sequence_shift -> test unseen low_and_slow) ===")
    random.seed(20260830)  # reproducible sample generation; still zero LLM calls
    extra = generate_extra_sequence_samples(n_each=16)
    print(f"Generated {len(extra)} additional deterministic credential_ato/sequence_shift samples")

    seq_pool = [t for t in traces if t.family == "sequence_anomaly"] + extra
    n_ca = sum(1 for t in seq_pool if t.metadata.get("preset") == "credential_ato")
    n_ss = sum(1 for t in seq_pool if t.metadata.get("preset") == "sequence_shift")
    n_los = sum(1 for t in seq_pool if t.metadata.get("preset") == "low_and_slow")
    print(f"pool composition: credential_ato={n_ca}, sequence_shift={n_ss}, low_and_slow (held out)={n_los}")

    generalization_check(
        seq_pool, {"credential_ato", "sequence_shift"}, {"low_and_slow"}, feat_seq_fixed,
        "frozen-baseline 3 features only",
    )
    generalization_check(
        seq_pool, {"credential_ato", "sequence_shift"}, {"low_and_slow"}, feat_seq_fixed_plus_slope,
        "frozen-baseline 3 features + rolling_mean_slope",
    )

    print(
        "\nreasoning_attack / intent_manipulation generalization: kept at their existing honest "
        "caveats (4 Vault Whisper framings, 3 intent-manipulation techniques) — no sample inflation; "
        "see the bonus identity_mismatch check above for the one addition made to that family."
    )

    total_calls = get_call_count()
    print(f"\nTotal LLM calls this entire validation run: {total_calls} (must be 0)")
    assert total_calls == 0, "Phase B must make zero new LLM calls — investigate before trusting these results"


if __name__ == "__main__":
    main()
