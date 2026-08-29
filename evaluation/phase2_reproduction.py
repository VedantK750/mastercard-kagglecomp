"""Day-4 gate: exercise families 1.D (Delegation/Authorization Abuse) and 1.C
(Intent Manipulation) the same way baseline_reproduction.py exercises 1.A —
benign + attack trials, real Blue detectors, classification metrics and the
4-case confusion breakdown for each family plus the combined set.

Run: PYTHONPATH=. .venv/bin/python evaluation/phase2_reproduction.py
"""

from __future__ import annotations

from src.blue_team.delegation_abuse_detector import DelegationAbuseDetector
from src.blue_team.intent_manipulation_detector import IntentManipulationDetector
from src.common.trace_io import write_traces
from src.red_team.delegation_abuse import VIOLATION_TYPES, DelegationAbuseGenerator
from src.red_team.intent_manipulation import DECOY_TECHNIQUES, IntentManipulationGenerator
from evaluation.metrics import classification_metrics, confusion_breakdown

N_BENIGN_PER_FAMILY = 10


def run_delegation_abuse():
    gen = DelegationAbuseGenerator()
    det = DelegationAbuseDetector()
    seed = gen.seed()[0]

    traces = [gen.simulate(seed, benign=True) for _ in range(N_BENIGN_PER_FAMILY)]
    for violation_type in VIOLATION_TYPES:
        for _ in range(2):  # 2 trials per violation type = 12 attack traces
            ctx = dict(seed)
            ctx["violation_type"] = violation_type
            traces.append(gen.simulate(ctx, benign=False))

    verdicts = [det.evaluate(t) for t in traces]
    return traces, verdicts


def run_intent_manipulation():
    gen = IntentManipulationGenerator()
    det = IntentManipulationDetector()
    seed = gen.seed()[0]

    traces = [gen.simulate(seed, benign=True) for _ in range(N_BENIGN_PER_FAMILY)]
    for _ in range(len(DECOY_TECHNIQUES) * 3):  # 3 trials per technique = 9 attack traces
        ctx = gen.mutate(seed)
        traces.append(gen.simulate(ctx, benign=False))

    verdicts = [det.evaluate(t) for t in traces]
    return traces, verdicts


def main():
    print("=== 1.D Delegation Abuse ===")
    da_traces, da_verdicts = run_delegation_abuse()
    print(classification_metrics(da_traces, da_verdicts))
    print(confusion_breakdown(da_traces, da_verdicts))

    print("\n=== 1.C Intent Manipulation ===")
    im_traces, im_verdicts = run_intent_manipulation()
    print(classification_metrics(im_traces, im_verdicts))
    print(confusion_breakdown(im_traces, im_verdicts))
    technique_outcomes = {}
    for t in im_traces:
        tech = t.metadata.get("technique")
        if tech is None:
            continue
        technique_outcomes.setdefault(tech, []).append(t.metadata["attack_succeeded"])
    for tech, outcomes in technique_outcomes.items():
        print(f"  {tech}: {sum(outcomes)}/{len(outcomes)} succeeded")

    all_traces = da_traces + im_traces
    all_verdicts = da_verdicts + im_verdicts
    write_traces(all_traces, "traces/phase2_traces.jsonl", mode="w")
    print(f"\nWrote {len(all_traces)} traces to traces/phase2_traces.jsonl")

    print("\n=== Combined 1.D + 1.C ===")
    print(classification_metrics(all_traces, all_verdicts))
    print(confusion_breakdown(all_traces, all_verdicts))


if __name__ == "__main__":
    main()
