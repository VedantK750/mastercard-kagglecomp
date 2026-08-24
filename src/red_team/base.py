"""RedGenerator ABC — the seed -> mutate -> simulate -> score shape every
attack family implements. Concrete generators live one file per family
(branded_whisper.py, vault_whisper.py, intent_manipulation.py, ...).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from src.common.schemas import AttackTrace


class RedGenerator(ABC):
    family: str = "base"

    @abstractmethod
    def seed(self) -> List[Dict[str, Any]]:
        """Return a small set of hand-written seed contexts (benign goal +
        attack-relevant content) this generator starts from."""
        raise NotImplementedError

    @abstractmethod
    def mutate(self, seed_context: Dict[str, Any]) -> Dict[str, Any]:
        """Produce one variant context from a seed (or a prior mutation)."""
        raise NotImplementedError

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
