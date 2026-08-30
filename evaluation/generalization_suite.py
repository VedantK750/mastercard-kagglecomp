"""Three-tier generalization suite for the sequence_anomaly family.

These three questions are DIFFERENT and must never be reported as one number.
Conflating them is the single easiest way to overclaim in this project, and we
already did it once: a live adaptive run reported "13/15 on unseen
low_and_slow" while a controlled experiment on the same question reported
0/13. The difference was pool composition, not capability.

  TIER 1  in-distribution
          Blue trains and tests on the same strategies. Split by LINEAGE ROOT
          (not trace_id) — children are near-copies of their parents, so a
          per-trace split puts near-twins on both sides.
          CLAIM: "Blue detects attacks like the ones it trains on."

  TIER 2  attack-strength generalization (same strategy, unseen parameters)
          Split by strength BIN. Two sub-cases, which are not equally hard:
            interpolation — test strengths lie inside the training range
            extrapolation — test strengths lie beyond it (weaker attacks)
          CLAIM: "Blue handles unseen parameterizations of a KNOWN strategy."
          NOT a claim about unseen strategies.

  TIER 3  cross-strategy generalization (entirely unseen strategy)
          Train on credential_ato + sequence_shift, test on low_and_slow,
          with ZERO low_and_slow anywhere in training — including replay-floor
          traces, which is exactly why the replay floor cannot be used to
          "fix" this number. Run for both Blue variants, because the contrast
          IS the finding:
            supervised  — expected to fail (identifiability, see below)
            anomaly     — never sees attack labels at all, so an unseen
                          strategy is on the same footing as a seen one.
          CLAIM: "Blue detects a strategy it has never been trained on."

WHY SUPERVISED CANNOT WIN TIER 3. In a credential_ato + sequence_shift
training pool, the mechanism that identifies low_and_slow carries no label
information — benign values span and exceed the attack values. The
coefficient on it is therefore unidentified and its fitted sign is set by
sampling noise. No supervised model (LR, boosting, or a network) recovers a
coefficient the data does not constrain. This suite is built to expose that,
not to paper over it.

Zero LLM calls: sequence_anomaly is fully deterministic.

Run: PYTHONPATH=. .venv/bin/python -m evaluation.generalization_suite
"""

from __future__ import annotations

import random
import statistics
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from sklearn.linear_model import LogisticRegression

from evaluation.adaptive_loop import is_train
from src.blue_team.anomaly_layer import MechanismAnomalyLayer
from src.blue_team.sequence_anomaly_detector import MECHANISM_NAMES, SequenceAnomalyDetector
from src.common.llm_client import get_call_count, reset_call_count
from src.common.schemas import AttackTrace
from src.red_team.sequence_anomaly import SequenceAnomalyGenerator

DETECTOR = SequenceAnomalyDetector()
GEN = SequenceAnomalyGenerator()
SEED_CTX = GEN.seed()[0]

# Strength grid. Sweeping amount alone is NOT a valid strength sweep: the
# low_and_slow preset also carries spacing_multiplier=0.9, so an amount-only
# sweep leaves a constant cadence signal in every sample and never actually
# produces a weak attack (this contaminated an earlier robustness curve —
# "recall 0.60 at amount_multiplier=1.00" was reading cadence, not amount).
# Each strength here moves BOTH levers toward benign together.
STRENGTH_GRID: Dict[str, Tuple[float, float]] = {
    "0.80": (0.80, 0.80),
    "0.85": (0.85, 0.85),
    "0.90": (0.90, 0.90),
    "0.92": (0.92, 0.92),
    "0.95": (0.95, 0.95),
    "0.97": (0.97, 0.97),
}


def make(
    preset: Optional[str], n: int, seed: int,
    amount_multiplier: Optional[float] = None,
    spacing_multiplier: Optional[float] = None,
    n_tail: int = 15, tag: Optional[str] = None,
) -> List[AttackTrace]:
    random.seed(seed)
    out: List[AttackTrace] = []
    for _ in range(n):
        ctx = dict(SEED_CTX)
        if preset is None:
            t = GEN.simulate(ctx, benign=True)
        else:
            ctx["preset"] = preset
            if amount_multiplier is not None:
                ctx["amount_multiplier"] = amount_multiplier
                ctx["spacing_multiplier"] = spacing_multiplier
                ctx["n_tail_txns"] = n_tail
            else:
                ctx = GEN.mutate(ctx)
            t = GEN.simulate(ctx, benign=False)
        if tag:
            t.metadata["strength_bin"] = tag
        out.append(t)
    return out


def _vec(t: AttackTrace) -> Optional[List[float]]:
    s = DETECTOR.mechanism_scores(t)
    return None if s is None else [s[n] for n in MECHANISM_NAMES]


def eval_supervised(train: Sequence[AttackTrace], test: Sequence[AttackTrace]) -> Optional[Dict[str, float]]:
    tr = [(t, _vec(t)) for t in train]
    tr = [(t, v) for t, v in tr if v is not None]
    te = [(t, _vec(t)) for t in test]
    te = [(t, v) for t, v in te if v is not None]
    if not tr or not te:
        return None
    y = [1 if t.ground_truth_label else 0 for t, _ in tr]
    if len(set(y)) < 2:
        return None
    clf = LogisticRegression(class_weight="balanced", max_iter=1000).fit([v for _, v in tr], y)
    preds = [bool(clf.predict([v])[0]) for _, v in te]
    return _metrics([t for t, _ in te], preds)


def eval_anomaly(null_traces: Sequence[AttackTrace], test: Sequence[AttackTrace], alpha: float = 0.05):
    """`null_traces` MUST be attack-free. The layer never sees a label, which
    is precisely why an unseen strategy is not disadvantaged."""
    assert all(not t.ground_truth_label for t in null_traces), \
        "anomaly layer calibration set contains attacks — that defeats its purpose"
    scores = [DETECTOR.mechanism_scores(t) for t in null_traces]
    scores = [s for s in scores if s is not None]
    layer = MechanismAnomalyLayer(alpha=alpha)
    layer.calibrate(scores)
    te = [(t, DETECTOR.mechanism_scores(t)) for t in test]
    te = [(t, s) for t, s in te if s is not None]
    if not te:
        return None, layer
    preds = [layer.score(s)[0] >= 1.0 for _, s in te]
    return _metrics([t for t, _ in te], preds), layer


def _metrics(traces: Sequence[AttackTrace], preds: Sequence[bool]) -> Dict[str, float]:
    tp = sum(1 for t, p in zip(traces, preds) if t.ground_truth_label and p)
    fp = sum(1 for t, p in zip(traces, preds) if not t.ground_truth_label and p)
    fn = sum(1 for t, p in zip(traces, preds) if t.ground_truth_label and not p)
    tn = sum(1 for t, p in zip(traces, preds) if not t.ground_truth_label and not p)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "n": len(traces), "recall": rec, "precision": prec,
        "f1": 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0,
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def show(label: str, m: Optional[Dict[str, float]]) -> None:
    if m is None:
        print(f"  {label:52} — skipped (insufficient/degenerate data)")
        return
    print(f"  {label:52} R={m['recall']:.2f} P={m['precision']:.2f} "
          f"F1={m['f1']:.2f} FPR={m['fpr']:.2f}  (n={m['n']}, TP={m['tp']} FN={m['fn']} FP={m['fp']})")


def main() -> None:
    reset_call_count()

    benign_train = make(None, 60, seed=1)
    benign_test = make(None, 40, seed=2)
    # Length-matched nulls: the low_and_slow CODEPATH with both levers at 1.0,
    # i.e. no actual attack. Needed because ordinary benign traces are 14 txns
    # while low_and_slow is 23, and a threshold calibrated only on short
    # traces would not transfer across lengths.
    null_long = make("low_and_slow", 60, seed=3, amount_multiplier=1.0, spacing_multiplier=1.0)
    for t in null_long:
        t.ground_truth_label = False
        t.metadata["attack_succeeded"] = False
        t.metadata["preset"] = None
        t.metadata["is_length_matched_null"] = True
    null_long_test = make("low_and_slow", 40, seed=4, amount_multiplier=1.0, spacing_multiplier=1.0)
    for t in null_long_test:
        t.ground_truth_label = False
        t.metadata["attack_succeeded"] = False
        t.metadata["preset"] = None
        t.metadata["is_length_matched_null"] = True

    calib = benign_train + null_long

    # ---------------- TIER 1 ----------------
    print("=" * 78)
    print("TIER 1 — IN-DISTRIBUTION (same strategies; split by LINEAGE ROOT)")
    print("  Claim: Blue detects attacks like the ones it trains on.")
    print("=" * 78)
    pool: List[AttackTrace] = []
    for preset, s in (("credential_ato", 11), ("sequence_shift", 12), ("low_and_slow", 13)):
        pool += make(preset, 40, seed=s)
    pool += benign_train + null_long
    tr = [t for t in pool if is_train(t.metadata.get("root_id") or t.trace_id)]
    te = [t for t in pool if not is_train(t.metadata.get("root_id") or t.trace_id)]
    print(f"  train n={len(tr)}  test n={len(te)}")
    show("supervised LR (6 mechanism features)", eval_supervised(tr, te))
    m, _ = eval_anomaly([t for t in tr if not t.ground_truth_label], te)
    show("anomaly layer (null-calibrated, no attack labels)", m)

    # ---------------- TIER 2 ----------------
    print()
    print("=" * 78)
    print("TIER 2 — ATTACK-STRENGTH GENERALIZATION (low_and_slow; unseen PARAMETERS)")
    print("  Claim: Blue handles unseen parameterizations of a KNOWN strategy.")
    print("  NOTE: both levers (amount AND cadence) move together — an")
    print("        amount-only sweep leaves a constant cadence signal and never")
    print("        actually produces a weak attack.")
    print("=" * 78)
    by_bin = {
        tag: make("low_and_slow", 40, seed=100 + i, amount_multiplier=a, spacing_multiplier=sp, tag=tag)
        for i, (tag, (a, sp)) in enumerate(STRENGTH_GRID.items())
    }
    for name, train_bins, test_bins in (
        ("INTERPOLATION (test inside train range)", ["0.80", "0.90", "0.97"], ["0.85", "0.92"]),
        ("EXTRAPOLATION (test weaker than any seen)", ["0.80", "0.85", "0.90"], ["0.92", "0.95", "0.97"]),
    ):
        tr = [t for b in train_bins for t in by_bin[b]] + benign_train + null_long
        te = [t for b in test_bins for t in by_bin[b]] + benign_test + null_long_test
        print(f"\n  {name}   train={train_bins} test={test_bins}")
        show("supervised LR", eval_supervised(tr, te))
        m, _ = eval_anomaly(calib, te)
        show("anomaly layer", m)
        print("    per-strength recall (attacks only):")
        for b in test_bins:
            sub = by_bin[b]
            sm = eval_supervised(tr, sub + benign_test)
            am, _ = eval_anomaly(calib, sub + benign_test)
            print(f"      {b}: supervised R={sm['recall']:.2f}   anomaly R={am['recall']:.2f}" if sm and am else f"      {b}: n/a")

    # ---------------- TIER 3 ----------------
    print()
    print("=" * 78)
    print("TIER 3 — CROSS-STRATEGY (train credential_ato+sequence_shift -> test UNSEEN low_and_slow)")
    print("  ZERO low_and_slow in training, including replay floor.")
    print("=" * 78)
    ca = make("credential_ato", 40, seed=21)
    ss = make("sequence_shift", 40, seed=22)
    tr = ca + ss + benign_train + null_long
    assert not any(t.metadata.get("preset") == "low_and_slow" for t in tr), "TIER 3 LEAK"
    for tag in ("0.85", "0.90", "0.95"):
        te = by_bin[tag] + benign_test + null_long_test
        print(f"\n  held-out low_and_slow @ strength {tag}")
        show("supervised LR (trained on loud presets only)", eval_supervised(tr, te))
        m, _ = eval_anomaly(calib, te)
        show("anomaly layer (never saw ANY attack)", m)

    print("\n  sanity — same anomaly layer on the two SEEN loud strategies:")
    for nm, g in (("credential_ato", ca), ("sequence_shift", ss)):
        m, _ = eval_anomaly(calib, g + benign_test)
        show(f"anomaly layer on {nm}", m)

    calls = get_call_count()
    print(f"\nTotal LLM calls: {calls} (must be 0)")
    assert calls == 0, "sequence_anomaly is deterministic — any LLM call here is a bug"


if __name__ == "__main__":
    main()
