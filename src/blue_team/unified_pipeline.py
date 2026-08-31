"""G9 — one routing layer for family -> generator/detector dispatch.

Before this, every evaluation script rebuilt the same mapping by hand:
baseline_reproduction, phase2/phase3_reproduction, adaptive_loop,
feature_validation and generalization_suite each constructed their own
detectors and each knew independently which detector serves which family.
Adding a family meant editing all of them, and the `reasoning_attack`
special case (two generators, one shared detector) was re-derived every time.

This is deliberately a THIN layer. It centralizes the routing table and the
evaluate-a-batch loop; it does NOT own experiment design, training policy, or
metrics interpretation, because those legitimately differ between scripts —
the adaptive loop refits per generation, the generalization suite must use
fresh never-fit detectors, and the reproduction scripts must stay on the
untrained Generation-0 path. Forcing those through one abstraction would
break the very properties they exist to test.

Existing scripts are NOT rewritten to use this: their numbers are published
and the risk of silently perturbing them outweighs the duplication saved.
New code should route through here.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

from src.blue_team.base import Detector
from src.blue_team.delegation_abuse_detector import DelegationAbuseDetector
from src.blue_team.intent_manipulation_detector import IntentManipulationDetector
from src.blue_team.reasoning_attack_detector import ReasoningAttackDetector
from src.blue_team.sequence_anomaly_detector import SequenceAnomalyDetector
from src.common.schemas import AttackTrace, BlueVerdict
from src.red_team.base import RedGenerator
from src.red_team.branded_whisper import BrandedWhisperGenerator
from src.red_team.delegation_abuse import DelegationAbuseGenerator
from src.red_team.intent_manipulation import IntentManipulationGenerator
from src.red_team.sequence_anomaly import SequenceAnomalyGenerator
from src.red_team.vault_whisper import VaultWhisperGenerator

# family -> (detector factory, generator factories). reasoning_attack maps to
# TWO generators sharing ONE detector, which is the case every script was
# re-deriving by hand.
FAMILY_REGISTRY: Dict[str, Tuple[Callable[[], Detector], List[Callable[[], RedGenerator]]]] = {
    "reasoning_attack": (ReasoningAttackDetector, [BrandedWhisperGenerator, VaultWhisperGenerator]),
    "intent_manipulation": (IntentManipulationDetector, [IntentManipulationGenerator]),
    "delegation_abuse": (DelegationAbuseDetector, [DelegationAbuseGenerator]),
    "sequence_anomaly": (SequenceAnomalyDetector, [SequenceAnomalyGenerator]),
}

FAMILIES = list(FAMILY_REGISTRY)

# How to label a trace's segment within its family, for per-segment reporting.
SEGMENT_KEY: Dict[str, Callable[[AttackTrace], Optional[str]]] = {
    "reasoning_attack": lambda t: t.sub_attack,
    "intent_manipulation": lambda t: t.metadata.get("technique"),
    "delegation_abuse": lambda t: t.metadata.get("violation_type"),
    "sequence_anomaly": lambda t: t.metadata.get("preset"),
}


def make_detector(family: str) -> Detector:
    if family not in FAMILY_REGISTRY:
        raise KeyError(f"unknown family {family!r}; known: {FAMILIES}")
    return FAMILY_REGISTRY[family][0]()


def make_generators(family: str) -> List[RedGenerator]:
    if family not in FAMILY_REGISTRY:
        raise KeyError(f"unknown family {family!r}; known: {FAMILIES}")
    return [factory() for factory in FAMILY_REGISTRY[family][1]]


def segment_of(trace: AttackTrace) -> Optional[str]:
    fn = SEGMENT_KEY.get(trace.family)
    return fn(trace) if fn else None


def scenario_of(trace: AttackTrace) -> Optional[str]:
    """G4 scenario identity, uniform across families."""
    return trace.metadata.get("scenario_id")


def evaluate_traces(
    traces: Sequence[AttackTrace], detectors: Optional[Dict[str, Detector]] = None
) -> List[BlueVerdict]:
    """Route each trace to its family's detector. Accepts a caller-supplied
    detector map so a script can pass its own already-trained instances —
    fitted state lives on the instance, and silently constructing fresh
    untrained ones here would discard it."""
    detectors = detectors if detectors is not None else {}
    out: List[BlueVerdict] = []
    for trace in traces:
        det = detectors.get(trace.family)
        if det is None:
            det = make_detector(trace.family)
            detectors[trace.family] = det
        out.append(det.evaluate(trace))
    return out


def scenario_coverage(traces: Sequence[AttackTrace]) -> Dict[str, Dict[str, int]]:
    """G4 reporting: how many traces per (family, scenario_id). Surfaces the
    case where a run technically has many seeds but only exercised one."""
    cov: Dict[str, Dict[str, int]] = {}
    for t in traces:
        cov.setdefault(t.family, {})
        key = scenario_of(t) or "unknown"
        cov[t.family][key] = cov[t.family].get(key, 0) + 1
    return cov
