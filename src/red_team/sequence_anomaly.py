"""Sequence Anomaly — merges 1.B (credential/ATO), 3.A (low-and-slow), and
3.B (sequence manipulation) into one family, per the plan's explicit
decision not to build three near-identical trajectory generators. One Red
state-machine produces a transaction trajectory (baseline history + an
attack tail) against a synthetic legitimate-spending profile; three named
presets, each a fixed parameter grid, not a search/RL optimizer:

- credential_ato   : sudden burst of large, out-of-category transactions in
                      a short window after a period of normal activity —
                      the classic, loud account-takeover signature.
- low_and_slow      : tail transactions kept statistically INDISTINGUISHABLE
                      from the baseline profile (amount, category, cadence
                      all close to normal) — deliberately designed to stay
                      inside whatever rolling window a naive detector uses.
                      This is the realistic "should often evade" preset.
- sequence_shift    : amount and cadence stay near-baseline, but the
                      category drifts step by step away from the user's
                      typical spend — no single transaction looks wrong,
                      the trend does.

No LLM involved — this is a structural/statistical attack on transaction
history, not a reasoning-layer manipulation.

`mutate()` is a real feedback-driven hill-climb (adaptive co-evolution loop,
evaluation/adaptive_loop.py): each preset's amount/spacing/tail-length/drift
levers are context fields, not literals, and a caught round's
`detection_reasons` (from `AttackMemory`) nudge the specific lever that
caused the flag — velocity_ratio flagged -> spread transactions out;
amount_z flagged -> pull the amount back toward baseline; category_drift
flagged -> reduce how many tail transactions actually drift category.
"""

from __future__ import annotations

import random
from datetime import timedelta
from typing import Any, Dict, List, Optional

from src.common.ap2_env import new_trace_id, new_txn_id, now
from src.common.feedback import AttackMemory
from src.common.realism import sequence_realism
from src.common.schemas import AttackTrace, CredentialState, IntentObject, RedScore, Transaction
from src.red_team.base import RedGenerator

PRESETS = ["credential_ato", "low_and_slow", "sequence_shift"]

_PRESET_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "credential_ato": {"amount_multiplier": 7.0, "spacing_multiplier": 1.0, "n_tail_txns": 4},
    "low_and_slow": {"amount_multiplier": 0.85, "spacing_multiplier": 0.9, "n_tail_txns": 15},
    "sequence_shift": {"amount_multiplier": 1.1, "spacing_multiplier": 1.0, "n_tail_txns": 6, "drift_steps": 4},
}
_DRIFT_POOL = ["household", "household", "electronics", "luxury_goods"]

_PRESET_DETECTION_PRIOR = {  # Red's own prior guess at how loud each preset is
    "credential_ato": 0.7,
    "sequence_shift": 0.5,
    "low_and_slow": 0.15,
}

# G3 — a real, explicit, testable success criterion. Previously
# `attack_succeeded = preset is not None`, i.e. True whenever an attack
# existed, which made ASR 1.0 by construction and left Case C carrying no
# information for this family. Success now means the trajectory actually
# extracted meaningful value: at least this many "normal transactions worth"
# of spend in the attack tail. A 4-transaction low-and-slow at 0.85x the
# usual amount does NOT clear this bar, and should not — barely touching an
# account is an attempt, not a successful fraud.
FRAUD_SUCCESS_TXN_EQUIVALENTS = 5.0

# G7 — the searchable parameter space, with bounds. These are the dimensions
# that actually change the SHAPE of a trajectory rather than nudging it.
# Bounds are deliberately conservative: n_tail_txns is capped well below the
# point where the volume-plausibility term in realism.py would dominate, so
# Red explores real trajectory shapes instead of rediscovering the
# transaction-count exploit that G1/G2 just closed.
_PARAM_BOUNDS: Dict[str, tuple] = {
    "amount_multiplier": (0.5, 8.0),
    "spacing_multiplier": (0.3, 3.0),
    "n_tail_txns": (3, 16),
    "drift_steps": (0, 4),
}

# G5 — probability that a mutation crosses to a different preset. Small on
# purpose: most children stay related to their parent (preserving the lineage
# semantics the reward and the train/test split depend on), but a lineage is
# no longer permanently trapped in whichever preset it started with.
PRESET_SWITCH_PROB = 0.15
_PRESET_NOVELTY = {  # static fallback only — evasion.py's population search uses measured novelty instead
    "credential_ato": 0.4,   # well-known pattern
    "sequence_shift": 0.7,
    "low_and_slow": 1.0,     # stealthiest / most novel evasion
}


def _clip_positive(x: float) -> float:
    return max(1.0, x)


def _sequence_shift_categories(typical_category: str, n_tail: int, drift_steps: int) -> List[str]:
    drift_steps = max(0, min(drift_steps, len(_DRIFT_POOL)))
    n_typical_lead = max(0, n_tail - drift_steps)
    cats = [typical_category] * n_typical_lead + _DRIFT_POOL[:drift_steps]
    if len(cats) < n_tail:
        cats += [typical_category] * (n_tail - len(cats))
    return cats[:n_tail]


class SequenceAnomalyGenerator(RedGenerator):
    family = "sequence_anomaly"
    sub_attack = None  # preset lives in metadata; no SubAttackLiteral entry needed (not injection-based)

    def seed(self) -> List[Dict[str, Any]]:
        """G4 — seven distinct spending PROFILES, not one profile with cosmetic
        variations. They differ on the dimensions that actually change what a
        detector sees: typical amount (12 -> 480), dispersion (tight vs. wide),
        cadence (6h -> 168h between transactions), and history length (6 -> 14
        baseline transactions).

        This matters because every result before this change was effectively
        n=1 in scenario space: we measured variance across mutations of one
        grocery shopper, never across situations. A low-and-slow drain against
        a high-variance luxury spender is a materially different detection
        problem from the same attack against a tight, regular grocery
        account, and only the second was ever being tested.

        `scenario_id` is carried into trace metadata so scenario coverage is
        reportable and the train/test split can be audited against it."""
        return [
            {
                "scenario_id": "grocery_regular",
                "session_user_id": "user_carol", "agent_id": "shopping_agent_1",
                "typical_category": "groceries",
                "avg_amount": 45.0, "std_amount": 15.0, "velocity_hours": 20.0, "n_baseline": 8,
            },
            {
                "scenario_id": "daily_coffee",
                "session_user_id": "user_dan", "agent_id": "shopping_agent_1",
                "typical_category": "dining",
                "avg_amount": 12.0, "std_amount": 3.0, "velocity_hours": 8.0, "n_baseline": 14,
            },
            {
                "scenario_id": "commuter_transit",
                "session_user_id": "user_erin", "agent_id": "shopping_agent_1",
                "typical_category": "transport",
                "avg_amount": 28.0, "std_amount": 6.0, "velocity_hours": 12.0, "n_baseline": 12,
            },
            {
                "scenario_id": "household_bulk",
                "session_user_id": "user_frank", "agent_id": "shopping_agent_1",
                "typical_category": "household",
                "avg_amount": 130.0, "std_amount": 55.0, "velocity_hours": 72.0, "n_baseline": 8,
            },
            {
                "scenario_id": "electronics_occasional",
                "session_user_id": "user_grace", "agent_id": "shopping_agent_1",
                "typical_category": "electronics",
                "avg_amount": 310.0, "std_amount": 120.0, "velocity_hours": 168.0, "n_baseline": 6,
            },
            {
                "scenario_id": "luxury_highvariance",
                "session_user_id": "user_henry", "agent_id": "shopping_agent_1",
                "typical_category": "luxury_goods",
                "avg_amount": 480.0, "std_amount": 260.0, "velocity_hours": 120.0, "n_baseline": 7,
            },
            {
                "scenario_id": "smallbiz_supplies",
                "session_user_id": "user_iris", "agent_id": "shopping_agent_1",
                "typical_category": "office_supplies",
                "avg_amount": 95.0, "std_amount": 30.0, "velocity_hours": 6.0, "n_baseline": 14,
            },
        ]

    def searchable_params(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # scenario_id is included so two structurally-identical attacks against
        # DIFFERENT spending profiles are not treated as duplicates — post-G4
        # they are genuinely different attacks, and collapsing them would make
        # the novelty signal (and the dedup) wrong.
        return {
            "preset": context.get("preset"),
            "scenario_id": context.get("scenario_id"),
            "amount_multiplier": context.get("amount_multiplier"),
            "spacing_multiplier": context.get("spacing_multiplier"),
            "n_tail_txns": context.get("n_tail_txns"),
            "drift_steps": context.get("drift_steps"),
        }

    def mutate(self, seed_context: Dict[str, Any], feedback: Optional[Any] = None) -> Dict[str, Any]:
        context = dict(seed_context)
        prior_preset = context.get("preset")

        # G5 — break the one-way door. Previously `context.get("preset") or
        # random.choice(PRESETS)` only randomized off a BARE seed, so once a
        # lineage carried a preset every descendant inherited it forever and
        # Red could never explore across strategies. A lineage that converged
        # on low_and_slow was permanently unable to try credential_ato, which
        # is what starved Blue's training pool of the other presets.
        #
        # Resampling is deliberately a minority event: most children stay
        # related to their parent (the reward, the survivor selection, and the
        # lineage-based train/test split all depend on that), but the door now
        # opens both ways.
        # `_lock_preset` pins the strategy for CONTROLLED experiments. Without
        # it, a batch requested as "credential_ato" silently acquires some
        # low_and_slow members via the switching below — which is correct and
        # desirable during adaptive search, but is a leak in a cross-strategy
        # generalization test where the held-out strategy must appear NOWHERE
        # in training. evaluation/generalization_suite.py's TIER 3 assertion
        # caught exactly this, and the lock is the fix rather than weakening
        # the assertion.
        locked = bool(context.get("_lock_preset"))
        switched = False
        if prior_preset is None:
            preset = random.choice(PRESETS)
        elif locked:
            preset = prior_preset
        elif random.random() < PRESET_SWITCH_PROB:
            alternatives = [p for p in PRESETS if p != prior_preset]
            preset = random.choice(alternatives)
            switched = True
        else:
            preset = prior_preset

        if switched:
            # Carry no stale levers across a strategy boundary — a
            # credential_ato amount_multiplier of 7.0 is meaningless for
            # low_and_slow and would land the child far outside the new
            # preset's plausible region.
            for key in _PRESET_DEFAULTS[prior_preset]:
                context.pop(key, None)
        for key, value in _PRESET_DEFAULTS[preset].items():
            context.setdefault(key, value)

        # Baseline exploratory jitter on every call (not just when caught) —
        # this family has no LLM sampling to fall back on for diversity, so
        # without an unconditional jitter, two population members mutated
        # from the same parent with no/plain-string feedback (e.g. a
        # dedup retry) would produce byte-identical candidates.
        context["amount_multiplier"] = context["amount_multiplier"] * random.uniform(0.95, 1.05)
        context["spacing_multiplier"] = context["spacing_multiplier"] * random.uniform(0.95, 1.05)

        # G7 — the structural levers are now genuinely searched, not just
        # declared. n_tail_txns changes trajectory SHAPE (a 4-transaction
        # smash-and-grab vs. a 16-transaction drain are qualitatively
        # different attacks, not numeric variants of one), and drift_steps
        # controls how far sequence_shift's category walk actually travels.
        # Both were previously fixed at their preset default for the entire
        # life of a run.
        if random.random() < 0.5:
            context["n_tail_txns"] = int(context.get("n_tail_txns", 6)) + random.choice((-2, -1, 1, 2))
        if preset == "sequence_shift" and random.random() < 0.5:
            context["drift_steps"] = int(context.get("drift_steps", 4)) + random.choice((-1, 1))

        if isinstance(feedback, AttackMemory) and feedback.blue_detected:
            for reason in feedback.detection_reasons:
                if reason == "velocity_ratio":
                    context["spacing_multiplier"] = context["spacing_multiplier"] * 1.2
                elif reason == "amount_z":
                    context["amount_multiplier"] = 1 + (context["amount_multiplier"] - 1) * 0.8
                elif reason == "category_drift":
                    context["drift_steps"] = max(0, int(context.get("drift_steps", 0)) - 1)
                elif reason in ("cusum_norm", "persistence_frac"):
                    # Sustained-deviation reasons: shorten the tail so there
                    # is less for an accumulating statistic to accumulate.
                    context["n_tail_txns"] = int(context.get("n_tail_txns", 6)) - 1

        # Clamp every searched lever into its bounds. This is what stops the
        # search walking off into the reward-exploitation region G1/G2 closed
        # — bounds are enforced here rather than trusted to the reward alone.
        for key, (lo, hi) in _PARAM_BOUNDS.items():
            if key in context and context[key] is not None:
                value = max(lo, min(hi, context[key]))
                context[key] = int(round(value)) if isinstance(lo, int) else value

        context["preset"] = preset
        context["preset_switched"] = switched
        return context

    def _baseline_txns(self, context: Dict[str, Any]) -> List[Transaction]:
        txns = []
        t = now() - timedelta(hours=context["velocity_hours"] * context["n_baseline"])
        for _ in range(context["n_baseline"]):
            amount = round(_clip_positive(random.gauss(context["avg_amount"], context["std_amount"] * 0.4)), 2)
            txns.append(
                Transaction(
                    txn_id=new_txn_id(),
                    agent_id=context["agent_id"],
                    merchant_id="baseline_merchant",
                    amount=amount,
                    category=context["typical_category"],
                    timestamp=t,
                )
            )
            t += timedelta(hours=context["velocity_hours"] + random.uniform(-2, 2))
        return txns

    def _attack_tail(self, context: Dict[str, Any], preset: str, start_time) -> List[Transaction]:
        txns: List[Transaction] = []
        t = start_time
        defaults = _PRESET_DEFAULTS[preset]
        amount_multiplier = context.get("amount_multiplier", defaults["amount_multiplier"])
        spacing_multiplier = context.get("spacing_multiplier", defaults["spacing_multiplier"])
        n_tail = int(context.get("n_tail_txns", defaults["n_tail_txns"]))

        if preset == "credential_ato":
            for _ in range(n_tail):
                amount = round(
                    _clip_positive(random.gauss(context["avg_amount"] * amount_multiplier, context["avg_amount"] * 1.5)), 2
                )
                txns.append(
                    Transaction(
                        txn_id=new_txn_id(), agent_id=context["agent_id"], merchant_id="unknown_merchant_burst",
                        amount=amount, category="electronics", timestamp=t,
                    )
                )
                t += timedelta(minutes=random.uniform(5, 15) * spacing_multiplier)

        elif preset == "low_and_slow":
            for _ in range(n_tail):
                amount = round(
                    _clip_positive(random.gauss(context["avg_amount"] * amount_multiplier, context["std_amount"] * 0.4)), 2
                )
                txns.append(
                    Transaction(
                        txn_id=new_txn_id(), agent_id=context["agent_id"], merchant_id="baseline_merchant",
                        amount=amount, category=context["typical_category"], timestamp=t,
                    )
                )
                t += timedelta(hours=context["velocity_hours"] * spacing_multiplier + random.uniform(-2, 2))

        elif preset == "sequence_shift":
            drift_steps = int(context.get("drift_steps", defaults["drift_steps"]))
            cats = _sequence_shift_categories(context["typical_category"], n_tail, drift_steps)
            for cat in cats:
                amount = round(
                    _clip_positive(random.gauss(context["avg_amount"] * amount_multiplier, context["std_amount"] * 0.4)), 2
                )
                txns.append(
                    Transaction(
                        txn_id=new_txn_id(), agent_id=context["agent_id"], merchant_id="drift_merchant",
                        amount=amount, category=cat, timestamp=t,
                    )
                )
                t += timedelta(hours=context["velocity_hours"] * spacing_multiplier + random.uniform(-2, 2))

        return txns

    def _benign_continuation(self, context: Dict[str, Any], start_time) -> List[Transaction]:
        # Configurable so Blue can be given LENGTH-MATCHED benign traces.
        # Default 6 keeps every existing reproduction script byte-identical.
        # This matters: with a fixed 6, every benign trace is 14 transactions
        # while low_and_slow is 23, so an attack-free 23-txn sequence is
        # out-of-distribution for a detector trained only on short benign
        # examples — measured at a 25% false-positive rate on pure nulls that
        # differ from benign ONLY in length. Trace length is not evidence of
        # fraud, and a detector must not be allowed to learn that it is.
        txns = []
        t = start_time
        for _ in range(int(context.get("n_benign_continuation", 6))):
            amount = round(_clip_positive(random.gauss(context["avg_amount"], context["std_amount"] * 0.4)), 2)
            txns.append(
                Transaction(
                    txn_id=new_txn_id(), agent_id=context["agent_id"], merchant_id="baseline_merchant",
                    amount=amount, category=context["typical_category"], timestamp=t,
                )
            )
            t += timedelta(hours=context["velocity_hours"] + random.uniform(-2, 2))
        return txns

    def simulate(self, context: Dict[str, Any], benign: bool = False) -> AttackTrace:
        preset = None if benign else context.get("preset", random.choice(PRESETS))

        baseline = self._baseline_txns(context)
        tail_start = baseline[-1].timestamp + timedelta(hours=context["velocity_hours"] / 2)
        tail = (
            self._benign_continuation(context, tail_start)
            if benign
            else self._attack_tail(context, preset, tail_start)
        )
        all_txns = baseline + tail

        credential_state = {
            "credential_ato": CredentialState.COMPROMISED_UNKNOWN,
            "low_and_slow": CredentialState.COMPROMISED_LOW_VALUE,
            "sequence_shift": CredentialState.COMPROMISED_MIMIC,
        }.get(preset, CredentialState.LEGITIMATE)

        intent = IntentObject(
            category=context["typical_category"],
            max_amount=context["avg_amount"] * 2,
            raw_user_statement=(
                f"Ongoing account activity for {context['session_user_id']}, "
                f"typical category {context['typical_category']}."
            ),
        )

        baseline_total = sum(t.amount for t in baseline)
        tail_total = sum(t.amount for t in tail)

        # G3 — real success criterion. Was `preset is not None` (True whenever
        # an attack existed), which pinned ASR at 1.0 and made Case C
        # uninformative for this family. Success now requires the trajectory
        # to have actually extracted meaningful value, measured in
        # "normal transactions worth" of this specific account's spend, so the
        # bar scales with the profile rather than being an absolute figure
        # that means different things to a coffee buyer and a luxury spender.
        #
        # Explicitly NOT a function of whether Blue caught it — attack_present,
        # attack_succeeded, and blue_detected stay independent.
        txn_equivalents = tail_total / max(context["avg_amount"], 1e-9)
        attack_succeeded = bool(preset is not None and txn_equivalents >= FRAUD_SUCCESS_TXN_EQUIVALENTS)

        realism = sequence_realism(baseline, tail, baseline_mean_amount=context["avg_amount"]) if preset else 1.0

        red_score = RedScore.compute(
            intent_deviation=1.0 if preset else 0.0,
            payment_impact=(tail_total / max(baseline_total, 1.0)) if preset else 0.0,
            realism=realism,
            novelty=_PRESET_NOVELTY.get(preset, 0.0),
            detection_probability=_PRESET_DETECTION_PRIOR.get(preset, 0.05),
        )

        return AttackTrace(
            trace_id=new_trace_id("sequence_anomaly"),
            family="sequence_anomaly",
            sub_attack=None,
            objective="payment_manipulation",
            injection_channel=None,
            ground_truth_label=not benign,
            user_intent=intent,
            agent_reasoning_trace=[
                {
                    "agent": "MerchantPaymentProcessorAgent",
                    "action": "process_sequence",
                    "preset": preset,
                    "credential_state": credential_state.value,
                    "n_baseline": len(baseline),
                    "n_tail": len(tail),
                }
            ],
            transactions=all_txns,
            final_transaction=all_txns[-1],
            red_score=red_score,
            metadata={
                "preset": preset,
                "scenario_id": context.get("scenario_id"),          # G4 scenario coverage
                "preset_switched": context.get("preset_switched", False),  # G5 transitions
                "credential_state": credential_state.value,
                "attack_succeeded": attack_succeeded,
                "txn_equivalents_extracted": round(txn_equivalents, 2),  # G3 — why success/failure
                "realism": realism,                                  # G1 — measured, not constant
                "n_tail_txns": len(tail),                            # G7 — searched, so recorded
                "drift_steps": context.get("drift_steps"),
                "baseline_total": round(baseline_total, 2),
                "tail_total": round(tail_total, 2),
            },
        )
