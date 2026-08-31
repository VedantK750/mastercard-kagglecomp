"""Measured realism scores for Red's reward — replaces the hardcoded
constants that previously sat in every generator (`realism=0.85` / `0.9`).

WHY THIS EXISTS. `RedScore.compute()` multiplies realism into the reward, so
it is one of the terms that is supposed to keep Red honest: an attack that
could never happen in a real payment system should not score well no matter
how effectively it evades. With realism pinned to a constant, that term could
not penalize anything — Red was free to produce arbitrarily absurd attacks at
zero cost. A CEM optimizer given a slightly wider search space did exactly
that, converging on a 55-transaction "attack" whose per-transaction amounts
were indistinguishable from normal spending (see the RL-viability study).

WHAT IS AND ISN'T BEING MEASURED. These functions score **plausibility, not
maliciousness**. A credential-takeover burst is loud, obvious, and highly
malicious — and also completely realistic, because real ATO looks exactly
like that. It must score HIGH here. What scores low is behavior no real
account or attacker would produce: transaction volumes that dwarf the
account's entire history, superhuman transaction cadence, degenerate or
empty generated content, prices outside any plausible market range.

Every score is deterministic, needs no LLM, and lands in [0, 1]. Each is a
product of independent per-dimension factors, so one implausible dimension is
enough to pull the score down — matching the multiplicative logic of the
reward it feeds.
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional, Sequence

from src.common.schemas import Transaction

# --------------------------------------------------------------------------
# Sequence anomaly
# --------------------------------------------------------------------------
# Dimensions measured (documented per the G1 requirement):
#   1. volume     — attack-tail transaction count relative to the account's
#                   own established history. A compromise window that emits
#                   several times more activity than the account has ever
#                   produced is not a plausible compromise; it is just the
#                   account's normal life, relabelled.
#   2. cadence    — the fastest sustained inter-transaction gap. Sub-minute
#                   sustained purchasing is not physically plausible even for
#                   an automated attacker hitting a real payment rail.
#   3. amounts    — non-positive amounts, or amounts orders of magnitude
#                   outside the account's own profile.
# Deliberately NOT measured: how loud, how large, or how detectable the
# attack is. Those are maliciousness, not plausibility.

VOLUME_PLAUSIBLE_RATIO = 2.0    # tail may be up to 2x the baseline history length
MIN_PLAUSIBLE_GAP_SECONDS = 60.0
MAX_PLAUSIBLE_AMOUNT_MULTIPLE = 50.0  # vs. the account's mean transaction


def sequence_realism(
    baseline: Sequence[Transaction],
    tail: Sequence[Transaction],
    baseline_mean_amount: Optional[float] = None,
) -> float:
    if not tail:
        return 1.0  # nothing attack-shaped happened; benign traces are plausible by definition
    if not baseline:
        return 0.5

    mean_amt = baseline_mean_amount or statistics.mean(t.amount for t in baseline)

    # 1. volume plausibility — smooth decay past the threshold rather than a
    #    cliff, so Red gets a usable gradient instead of a wall.
    ratio = len(tail) / max(len(baseline), 1)
    volume = 1.0 if ratio <= VOLUME_PLAUSIBLE_RATIO else VOLUME_PLAUSIBLE_RATIO / ratio

    # 2. cadence plausibility — measured on the MEDIAN gap, so one coincidental
    #    near-simultaneous pair doesn't condemn an otherwise plausible tail.
    cadence = 1.0
    if len(tail) >= 2:
        gaps = [
            (tail[i + 1].timestamp - tail[i].timestamp).total_seconds()
            for i in range(len(tail) - 1)
        ]
        median_gap = statistics.median(gaps)
        if median_gap < MIN_PLAUSIBLE_GAP_SECONDS:
            cadence = max(0.05, median_gap / MIN_PLAUSIBLE_GAP_SECONDS)

    # 3. amount plausibility — fraction of tail transactions inside a plausible
    #    band for this account.
    ok = sum(
        1 for t in tail
        if t.amount > 0 and t.amount <= mean_amt * MAX_PLAUSIBLE_AMOUNT_MULTIPLE
    )
    amounts = ok / len(tail)

    return round(max(0.0, min(1.0, volume * cadence * amounts)), 4)


# --------------------------------------------------------------------------
# LLM-generated text families (branded_whisper, vault_whisper,
# intent_manipulation)
# --------------------------------------------------------------------------
# Dimensions measured:
#   1. non-degenerate — empty or near-empty output means the LLM refused or
#      was safety-blocked; that is not a real attack, and previously scored
#      identically to a well-formed one.
#   2. length band    — content far outside the plausible length for its type
#      would not survive contact with a real merchant catalog or chat UI.
# Deliberately NOT measured: how adversarial the text is.

def text_realism(text: Optional[str], min_chars: int = 40, max_chars: int = 1200) -> float:
    if not text or not text.strip():
        return 0.0  # degenerate: refused / blocked / empty
    n = len(text.strip())
    if n < min_chars:
        return max(0.1, n / min_chars)
    if n > max_chars:
        return max(0.1, max_chars / n)
    return 1.0


def price_realism(price: float, reference_price: float, max_multiple: float = 5.0) -> float:
    """A decoy priced 50x the legitimate comparable would never be a credible
    listing. Symmetric: implausibly cheap is as suspicious as implausibly
    dear."""
    if price <= 0 or reference_price <= 0:
        return 0.0
    ratio = max(price / reference_price, reference_price / price)
    return 1.0 if ratio <= max_multiple else round(max(0.1, max_multiple / ratio), 4)


# --------------------------------------------------------------------------
# Delegation abuse
# --------------------------------------------------------------------------
# Dimensions measured:
#   1. amount plausibility vs. the delegated cap. Exceeding a cap 3x is
#      ordinary authorization abuse; exceeding it 10,000x is not a
#      transaction any real processor would carry.
# The other five violation types are structural (wrong agent, wrong category,
# expired window, forged edge) and are all perfectly plausible as written, so
# they score 1.0 — there is nothing implausible about them to measure, and
# inventing a penalty would be noise dressed as rigor.

def delegation_realism(txn_amount: float, edge_max_amount: float) -> float:
    if txn_amount <= 0:
        return 0.0
    if edge_max_amount <= 0:
        return 0.5
    overshoot = txn_amount / edge_max_amount
    return 1.0 if overshoot <= 10.0 else round(max(0.1, 10.0 / overshoot), 4)
