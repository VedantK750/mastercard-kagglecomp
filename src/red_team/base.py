"""RedGenerator ABC — the seed -> mutate -> simulate -> score shape every
attack family implements. Concrete generators live one file per family
(branded_whisper.py, vault_whisper.py, intent_manipulation.py, ...).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.common.schemas import AttackTrace


class RedGenerator(ABC):
    family: str = "base"
    text_field: Optional[str] = None  # context key holding LLM-generated text, for memory-store dedup

    @abstractmethod
    def seed(self) -> List[Dict[str, Any]]:
        """Return a small set of hand-written seed contexts (benign goal +
        attack-relevant content) this generator starts from."""
        raise NotImplementedError

    @abstractmethod
    def mutate(self, seed_context: Dict[str, Any], feedback: Optional[Any] = None) -> Dict[str, Any]:
        """Produce one variant context from a seed (or a prior mutation).
        `feedback` is either a plain string (the legacy single-shot
        run_evasion_search path — a caught-phrases avoidance instruction) or
        an `AttackMemory` (src/common/feedback.py, the population-search
        path — structured detection_reasons/reward a generator can act on
        precisely) — this is what turns a plain mutation loop into detector
        evasion / hill-climbing."""
        raise NotImplementedError

    def searchable_params(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """The subset of `context` relevant for AttackMemoryStore dedup/
        novelty comparison (src/common/memory.py) — used only when
        `text_field` is None. Default: every primitive scalar value in
        context. Deterministic/numeric generators should override this to
        return just their tunable search levers (not fixed scaffolding
        fields), or the constant fields dilute the similarity signal."""
        return {k: v for k, v in context.items() if isinstance(v, (str, int, float, bool)) or v is None}

    @abstractmethod
    def simulate(self, context: Dict[str, Any]) -> AttackTrace:
        """Run the mutated context through the AP2 simulation environment
        and return a fully-populated AttackTrace (red_score included)."""
        raise NotImplementedError

    def generate_batch(self, n: int) -> List[AttackTrace]:
        """Default batch driver: seed contexts are cycled and mutated until
        n traces are produced. Subclasses may override for family-specific
        batching (e.g. fixed-seed baseline reproduction runs)."""
        seeds = self.seed()
        traces: List[AttackTrace] = []
        i = 0
        while len(traces) < n:
            base = seeds[i % len(seeds)]
            context = self.mutate(base)
            traces.append(self.simulate(context))
            i += 1
        return traces
