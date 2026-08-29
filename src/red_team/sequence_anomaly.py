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
"""

from __future__ import annotations

import random
import uuid
from datetime import timedelta
from typing import Any, Dict, List, Optional

from src.common.ap2_env import new_trace_id, new_txn_id, now
from src.common.schemas import AttackTrace, CredentialState, IntentObject, RedScore, Transaction
from src.red_team.base import RedGenerator

PRESETS = ["credential_ato", "low_and_slow", "sequence_shift"]

_PRESET_DETECTION_PRIOR = {  # Red's own prior guess at how loud each preset is
    "credential_ato": 0.7,
    "sequence_shift": 0.5,
    "low_and_slow": 0.15,
}
_PRESET_NOVELTY = {
    "credential_ato": 0.4,   # well-known pattern
    "sequence_shift": 0.7,
    "low_and_slow": 1.0,     # stealthiest / most novel evasion
}


def _clip_positive(x: float) -> float:
    return max(1.0, x)


class SequenceAnomalyGenerator(RedGenerator):
    family = "sequence_anomaly"
    sub_attack = None  # preset lives in metadata; no SubAttackLiteral entry needed (not injection-based)

    def seed(self) -> List[Dict[str, Any]]:
        return [
            {
                "session_user_id": "user_carol",
                "agent_id": "shopping_agent_1",
                "typical_category": "groceries",
                "avg_amount": 45.0,
                "std_amount": 15.0,
                "velocity_hours": 20.0,  # ~1 transaction per 20 hours at baseline
                "n_baseline": 8,
            }
        ]

    def mutate(self, seed_context: Dict[str, Any], feedback: Optional[str] = None) -> Dict[str, Any]:
        # Deterministic family — preset selection is a fixed parameter grid,
        # not LLM-driven, so `feedback` (from evasion.py's loop) has nothing
        # to steer. Accepted only to satisfy the RedGenerator interface.
        context = dict(seed_context)
        context["preset"] = random.choice(PRESETS)
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

        if preset == "credential_ato":
            for _ in range(4):
                amount = round(_clip_positive(random.gauss(context["avg_amount"] * 7, context["avg_amount"] * 1.5)), 2)
                txns.append(
                    Transaction(
                        txn_id=new_txn_id(), agent_id=context["agent_id"], merchant_id="unknown_merchant_burst",
                        amount=amount, category="electronics", timestamp=t,
                    )
                )
                t += timedelta(minutes=random.uniform(5, 15))

        elif preset == "low_and_slow":
            for _ in range(15):
                amount = round(
                    _clip_positive(random.gauss(context["avg_amount"] * 0.85, context["std_amount"] * 0.4)), 2
                )
                txns.append(
                    Transaction(
                        txn_id=new_txn_id(), agent_id=context["agent_id"], merchant_id="baseline_merchant",
                        amount=amount, category=context["typical_category"], timestamp=t,
                    )
                )
                t += timedelta(hours=context["velocity_hours"] * 0.9 + random.uniform(-2, 2))

        elif preset == "sequence_shift":
            drift_categories = [
                context["typical_category"], context["typical_category"],
                "household", "household", "electronics", "luxury_goods",
            ]
            for cat in drift_categories:
                amount = round(_clip_positive(random.gauss(context["avg_amount"] * 1.1, context["std_amount"] * 0.4)), 2)
                txns.append(
                    Transaction(
                        txn_id=new_txn_id(), agent_id=context["agent_id"], merchant_id="drift_merchant",
                        amount=amount, category=cat, timestamp=t,
                    )
                )
                t += timedelta(hours=context["velocity_hours"] + random.uniform(-2, 2))

        return txns

    def _benign_continuation(self, context: Dict[str, Any], start_time) -> List[Transaction]:
        txns = []
        t = start_time
        for _ in range(6):
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
        attack_succeeded = preset is not None  # nothing in this env blocks the tail from executing

        red_score = RedScore.compute(
            intent_deviation=1.0 if preset else 0.0,
            payment_impact=(tail_total / max(baseline_total, 1.0)) if preset else 0.0,
            realism=0.85,
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
                "credential_state": credential_state.value,
                "attack_succeeded": attack_succeeded,
                "baseline_total": round(baseline_total, 2),
                "tail_total": round(tail_total, 2),
            },
        )
