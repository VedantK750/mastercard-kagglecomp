"""Day-5 gate: exercise the Sequence Anomaly family (merged 1.B ATO / 3.A
low-and-slow / 3.B sequence-shift) — benign + all three presets, the real
rolling-window Blue detector, classification metrics and the 4-case
confusion breakdown, broken out per preset (the presets are expected to
behave very differently: ATO and sequence_shift should be loudly caught,
low_and_slow is the designed-in blind spot).

Run: PYTHONPATH=. .venv/bin/python evaluation/phase3_reproduction.py
"""

from __future__ import annotations

from src.blue_team.sequence_anomaly_detector import SequenceAnomalyDetector
from src.common.trace_io import write_traces
from src.red_team.sequence_anomaly import PRESETS, SequenceAnomalyGenerator
from evaluation.metrics import classification_metrics, confusion_breakdown

N_BENIGN = 10
N_PER_PRESET = 8


def main():
    gen = SequenceAnomalyGenerator()
    det = SequenceAnomalyDetector()
    seed = gen.seed()[0]

    traces = [gen.simulate(seed, benign=True) for _ in range(N_BENIGN)]
    for preset in PRESETS:
        for _ in range(N_PER_PRESET):
            ctx = dict(seed)
            ctx["preset"] = preset
            traces.append(gen.simulate(ctx, benign=False))

    verdicts = [det.evaluate(t) for t in traces]

    print("=== Sequence Anomaly — combined ===")
    print(classification_metrics(traces, verdicts))
    print(confusion_breakdown(traces, verdicts))

    print("\n=== Per-preset detection rate ===")
    verdict_by_id = {v.trace_id: v for v in verdicts}
    for preset in PRESETS:
        preset_traces = [t for t in traces if t.metadata.get("preset") == preset]
        caught = sum(1 for t in preset_traces if verdict_by_id[t.trace_id].predicted_label)
        print(f"  {preset}: {caught}/{len(preset_traces)} detected")

    write_traces(traces, "traces/phase3_traces.jsonl", mode="w")
    print(f"\nWrote {len(traces)} traces to traces/phase3_traces.jsonl")


if __name__ == "__main__":
    main()
