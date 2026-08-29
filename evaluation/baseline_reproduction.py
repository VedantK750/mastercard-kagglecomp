"""Day-2 gate: reproduce Whispers of Wealth's Branded Whisper and Vault
Whisper results on our AP2 simulation before any automated variant
generation is trusted. Prints a Table-1-style comparison against the
paper's published figures and writes all traces to traces/attack_traces.jsonl.

Run: PYTHONPATH=. .venv/bin/python evaluation/baseline_reproduction.py
"""

from __future__ import annotations

from src.blue_team.reasoning_attack_detector import ReasoningAttackDetector
from src.common.trace_io import write_traces
from src.red_team.branded_whisper import BrandedWhisperGenerator
from src.red_team.vault_whisper import VaultWhisperGenerator
from evaluation.metrics import classification_metrics, confusion_breakdown, success_rate

N_TRIALS = 10  # matches the paper's trial count for direct comparability


def run_branded_whisper_baseline():
    gen = BrandedWhisperGenerator()
    seed_ctx = gen.seed()[0]

    benign_traces = [gen.simulate(seed_ctx, benign=True) for _ in range(N_TRIALS)]
    attack_traces = [gen.simulate(seed_ctx, benign=False) for _ in range(N_TRIALS)]

    benign_result = success_rate([t.metadata["attack_succeeded"] for t in benign_traces])
    attack_result = success_rate([t.metadata["attack_succeeded"] for t in attack_traces])

    return benign_traces, attack_traces, benign_result, attack_result


def run_vault_whisper_baseline():
    gen = VaultWhisperGenerator()
    seed_ctx = gen.seed()[0]

    attack_traces = [gen.simulate(seed_ctx) for _ in range(N_TRIALS)]
    exposure_result = success_rate([t.cross_user_exposure for t in attack_traces])

    return attack_traces, exposure_result


def main():
    print(f"=== Branded Whisper baseline reproduction (n={N_TRIALS}) ===")
    benign_traces, bw_attack_traces, benign_result, attack_result = run_branded_whisper_baseline()
    print(f"Benign condition ASR : {benign_result}   (paper: 0%)")
    print(f"Attack condition ASR : {attack_result}   (paper: 100%)")

    print(f"\n=== Vault Whisper baseline reproduction (n={N_TRIALS}) ===")
    vw_attack_traces, exposure_result = run_vault_whisper_baseline()
    print(f"Cross-user exposure rate : {exposure_result}   (paper: 20%)")

    all_traces = benign_traces + bw_attack_traces + vw_attack_traces
    write_traces(all_traces, "traces/attack_traces.jsonl", mode="w")
    print(f"\nWrote {len(all_traces)} traces to traces/attack_traces.jsonl")

    print(f"\n=== ReasoningAttackDetector (Blue) evaluated over all {len(all_traces)} traces ===")
    detector = ReasoningAttackDetector()
    verdicts = [detector.evaluate(t) for t in all_traces]
    print(classification_metrics(all_traces, verdicts))
    print(confusion_breakdown(all_traces, verdicts))


if __name__ == "__main__":
    main()
