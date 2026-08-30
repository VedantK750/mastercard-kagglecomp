"""Persistent cross-generation attack memory for the adaptive co-evolution
loop — `evaluation/adaptive_loop.py` constructs exactly ONE
`AttackMemoryStore` per family before the generation loop starts and never
resets it. Without this, the evolutionary search can (and will) repeatedly
rediscover the same trajectory across generations, and "novelty" has
nothing real to be measured against.

Two similarity strategies, matching the two kinds of generators in this
project — no new dependency, no extra LLM/API calls for either:

- Numeric/deterministic families (sequence_anomaly's tunable levers,
  intent_manipulation's decoy_price, delegation_abuse's violation_type):
  quantized-parameter distance.
- LLM-text families (branded_whisper/vault_whisper/intent_manipulation's
  generated copy): stdlib difflib similarity ratio over normalized text.

Callers pass only the family-specific SEARCHABLE subset of a candidate's
parameters (not the full mutate()-produced context dict, which may contain
unrelated fixed fields or non-hashable values like VaultWhisper's
credential_store) — see each generator's mutate() for what it passes.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from src.common.feedback import AttackMemory

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = _PUNCT_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def _quantize_params(parameters: Dict[str, Any]) -> Dict[str, Any]:
    quantized: Dict[str, Any] = {}
    for key, value in parameters.items():
        if isinstance(value, bool) or value is None or isinstance(value, str):
            quantized[key] = value
        elif isinstance(value, (int, float)):
            quantized[key] = round(float(value), 1)
        # non-hashable/complex values (dicts, lists) are silently skipped —
        # callers are expected to pass only the searchable numeric/string subset
    return quantized


def _numeric_similarity(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    keys = set(a) & set(b)
    if not keys:
        return 0.0
    diffs = []
    for key in keys:
        va, vb = a[key], b[key]
        if va is None or vb is None:
            diffs.append(0.0 if va == vb else 1.0)
        elif isinstance(va, bool) or isinstance(vb, bool) or isinstance(va, str) or isinstance(vb, str):
            diffs.append(0.0 if va == vb else 1.0)
        else:
            denom = max(abs(va), abs(vb), 1.0)
            diffs.append(min(1.0, abs(va - vb) / denom))
    avg_diff = sum(diffs) / len(diffs)
    return max(0.0, 1.0 - avg_diff)


class AttackMemoryStore:
    def __init__(self) -> None:
        self._entries: List[AttackMemory] = []
        self._param_entries: Dict[str, List[Dict[str, Any]]] = {}
        self._text_normed: Dict[str, List[str]] = {}

    def max_similarity(
        self, family: str, parameters: Dict[str, Any], text: Optional[str] = None
    ) -> float:
        """0.0 = nothing like it seen before for this family, 1.0 = exact
        duplicate. Drives both is_duplicate() and the measured novelty_score
        (novelty = 1 - this)."""
        if text is not None:
            normed = _normalize_text(text)
            priors = self._text_normed.get(family, [])
            if not priors:
                return 0.0
            return max(SequenceMatcher(None, normed, p).ratio() for p in priors)

        quantized = _quantize_params(parameters)
        priors = self._param_entries.get(family, [])
        if not priors:
            return 0.0
        return max(_numeric_similarity(quantized, p) for p in priors)

    def is_duplicate(
        self, family: str, parameters: Dict[str, Any], text: Optional[str] = None, threshold: float = 0.85
    ) -> bool:
        return self.max_similarity(family, parameters, text) >= threshold

    def record(self, entry: AttackMemory, parameters: Dict[str, Any], text: Optional[str] = None) -> None:
        self._entries.append(entry)
        if text is not None:
            self._text_normed.setdefault(entry.family, []).append(_normalize_text(text))
        else:
            self._param_entries.setdefault(entry.family, []).append(_quantize_params(parameters))

    def history_for(self, family: str) -> List[AttackMemory]:
        """Red's own 'what have I already tried?' query."""
        return [e for e in self._entries if e.family == family]

    def all_entries(self) -> List[AttackMemory]:
        return list(self._entries)


class BlueReplayMemory:
    """Training-coverage bookkeeping for Blue — deliberately NOT the same
    store as `AttackMemoryStore` above.

    The two have OPPOSITE objectives, and conflating them caused a real bug:
    the first stratified-replay-floor implementation checked candidates
    against Red's `AttackMemoryStore` before accepting them, so near-
    duplicates were rejected. But Red's store exists to SUPPRESS similarity
    (that's what makes its novelty score and distinct-evasion count
    meaningful), while Blue's training wants the opposite — mass and coverage,
    where near-duplicates are harmless and often useful. The result was a
    floor that filled 1 of 8 requested slots for `low_and_slow` and left the
    generalization numbers uninterpretable.

    So: Red dedups, Blue accumulates. This class tracks per-segment counts so
    a floor can be topped up to a target, and tags hard negatives (traces Red
    got past Blue) so their contribution stays auditable. It intentionally
    holds no similarity logic at all."""

    def __init__(self) -> None:
        self._by_segment: Dict[str, List[str]] = {}
        self._hard_negatives: set[str] = set()

    def record(self, segment: Optional[str], trace_id: str, is_hard_negative: bool = False) -> None:
        self._by_segment.setdefault(segment or "unknown", []).append(trace_id)
        if is_hard_negative:
            self._hard_negatives.add(trace_id)

    def count(self, segment: str) -> int:
        return len(self._by_segment.get(segment, []))

    def deficit(self, segment: str, target: int) -> int:
        return max(0, target - self.count(segment))

    def coverage(self, segment_universe: List[str], target: int) -> Dict[str, int]:
        """Per-segment shortfall against the floor — what top-up must close."""
        return {seg: self.deficit(seg, target) for seg in segment_universe}

    def mark_hard_negative(self, trace_id: str) -> None:
        self._hard_negatives.add(trace_id)

    def is_hard_negative(self, trace_id: str) -> bool:
        return trace_id in self._hard_negatives

    @property
    def n_hard_negatives(self) -> int:
        return len(self._hard_negatives)
