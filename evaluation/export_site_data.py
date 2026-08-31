"""Curates a committed snapshot for the project website.

`traces/*.jsonl` and `evaluation/results/*.csv` are gitignored, so a fresh
clone — or any deploy — has nothing to display. This script selects a small,
representative slice and writes it to docs/site_data.json, which IS committed.
That file is the single source of truth for the site: docs/build_site.py reads
it and nothing else, so the page can never silently drift from the run that
produced it.

Curation aims for a spread of OUTCOMES, not just families: successful and
failed attacks, caught and missed, so Case C is visible in the UI rather than
only described. Blue verdicts are computed here (once, offline) using the same
detectors the evaluation harness uses.

Run: PYTHONPATH=. .venv/bin/python evaluation/export_site_data.py
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from src.blue_team.sequence_anomaly_detector import MECHANISM_NAMES, SequenceAnomalyDetector
from src.blue_team.unified_pipeline import make_detector, scenario_of, segment_of
from src.common.schemas import AttackTrace
from src.common.scoring import attack_succeeded
from src.common.trace_io import load_traces

TRACES = "traces/adaptive_loop_traces.jsonl"
SUMMARY = "evaluation/results/adaptive_loop_summary.csv"
BREAKDOWN = "evaluation/results/adaptive_loop_breakdown.csv"
OUT = Path("docs/site_data.json")

PER_FAMILY = 14  # curated traces per family


def _four_outcomes(trace: AttackTrace, detected: bool) -> str:
    if not trace.ground_truth_label:
        return "false_positive" if detected else "clean"
    succeeded = attack_succeeded(trace)
    if succeeded and detected:
        return "case_b"
    if succeeded and not detected:
        return "case_c"
    if not succeeded and detected:
        return "case_a"
    return "case_d"


def _narrative(trace: AttackTrace) -> List[Dict[str, Any]]:
    """The step-by-step story the Trace Theatre plays back. Real values only —
    the agent's verbatim output is shown, never a paraphrase, because the whole
    point is that a judge can see what the model actually did."""
    steps: List[Dict[str, Any]] = []
    fam, sub = trace.family, trace.sub_attack

    steps.append({
        "actor": "user", "icon": "user", "title": "User intent",
        "body": trace.user_intent.raw_user_statement,
    })

    injected = [c for c in trace.external_content if c.contains_injection]
    if injected:
        steps.append({
            "actor": "attacker", "icon": "attack",
            "title": "Adversarial content enters the agent's context",
            "body": injected[0].text,
            "tag": injected[0].injection_technique or "",
            "danger": True,
        })

    for step in trace.agent_reasoning_trace:
        action = step.get("action", "")
        if action == "find_products":
            n = len(step.get("candidates", []))
            steps.append({
                "actor": "merchant", "icon": "merchant",
                "title": "Merchant Agent · find_products()",
                "body": f"{n} candidate product(s) returned to the agent"
                        + ("  —  decoy passed the category/price filter"
                           if trace.metadata.get("decoy_reached_agent") else ""),
            })
        elif action == "select_product":
            steps.append({
                "actor": "agent", "icon": "agent",
                "title": "Shopping Agent · select_product()",
                "body": (step.get("raw_output") or "").strip()[:420],
                "verbatim": True,
            })
        elif action == "request_credentials":
            steps.append({
                "actor": "agent", "icon": "agent",
                "title": "Shopping Agent · request_credentials()",
                "body": (step.get("raw_output") or "").strip()[:420],
                "verbatim": True,
            })
        elif action == "process_payment":
            steps.append({
                "actor": "processor", "icon": "payment",
                "title": "Payment Processor · process_payment()",
                "body": f"executing edge {step.get('executing_authorization_edge')}  ·  "
                        f"violation: {step.get('violation_type')}",
                "danger": True,
            })
        elif action == "process_sequence":
            steps.append({
                "actor": "processor", "icon": "payment",
                "title": "Payment Processor · transaction sequence",
                "body": f"{step.get('n_baseline')} baseline + {step.get('n_tail')} attack-tail "
                        f"transactions  ·  credential state: {step.get('credential_state')}",
            })

    if sub == "vault_whisper":
        req = trace.metadata.get("requested_user_id")
        sess = trace.metadata.get("session_user_id")
        steps.append({
            "actor": "vault", "icon": "vault", "title": "Credentials Provider",
            "body": f"requested: {req}     session: {sess}",
            "tag": "IDENTITY MISMATCH" if req != sess else "identity consistent",
            "danger": req != sess,
        })

    if trace.mandates:
        steps.append({
            "actor": "mandate", "icon": "mandate",
            "title": f"{len(trace.mandates)} mandate(s) signed",
            "body": " · ".join(f"{m.mandate_type}: signature_valid={m.signature_valid}"
                               for m in trace.mandates),
            "thesis": all(m.signature_valid for m in trace.mandates),
        })
    return steps


def _sequence_series(trace: AttackTrace) -> Dict[str, Any] | None:
    if trace.family != "sequence_anomaly" or not trace.transactions:
        return None
    txns = sorted(trace.transactions, key=lambda t: t.timestamp)
    base = int(trace.agent_reasoning_trace[0].get("n_baseline", 8)) if trace.agent_reasoning_trace else 8
    return {
        "amounts": [round(t.amount, 2) for t in txns],
        "baseline_n": base,
        "categories": [t.category for t in txns],
    }


def build() -> None:
    traces = load_traces(TRACES)
    detectors = {}
    verdicts = {}
    for t in traces:
        det = detectors.setdefault(t.family, make_detector(t.family))
        verdicts[t.trace_id] = det.evaluate(t)

    # ---- curate: spread across families AND outcomes -----------------------
    by_family: Dict[str, List[AttackTrace]] = defaultdict(list)
    for t in traces:
        by_family[t.family].append(t)

    curated: List[Dict[str, Any]] = []
    for fam, group in by_family.items():
        buckets: Dict[str, List[AttackTrace]] = defaultdict(list)
        for t in group:
            buckets[_four_outcomes(t, verdicts[t.trace_id].predicted_label)].append(t)
        picked: List[AttackTrace] = []
        # round-robin across outcome buckets so no single outcome dominates
        order = ["case_c", "case_b", "case_a", "case_d", "clean", "false_positive"]
        i = 0
        while len(picked) < PER_FAMILY and any(buckets[o] for o in order):
            b = buckets[order[i % len(order)]]
            if b:
                picked.append(b.pop(0))
            i += 1

        for t in picked:
            v = verdicts[t.trace_id]
            det = detectors[t.family]
            mech = None
            if isinstance(det, SequenceAnomalyDetector):
                s = det.mechanism_scores(t)
                if s:
                    mech = {k: round(s[k], 3) for k in MECHANISM_NAMES}
            curated.append({
                "id": t.trace_id,
                "family": t.family,
                "sub": t.sub_attack or (t.metadata.get("preset") or ""),
                "segment": segment_of(t) or "—",
                "scenario": scenario_of(t) or "—",
                "generation": t.generation,
                "objective": t.objective,
                "present": t.ground_truth_label,
                "succeeded": attack_succeeded(t),
                "detected": v.predicted_label,
                "outcome": _four_outcomes(t, v.predicted_label),
                "risk": round(v.risk_score, 3),
                "checks": v.triggered_checks[:4],
                "explanation": v.explanation,
                "steps": _narrative(t),
                "series": _sequence_series(t),
                "mech": mech,
                "signature_valid": all(m.signature_valid for m in t.mandates) if t.mandates else None,
            })

    # ---- corpus-level counters (from ALL traces, not the curated subset) ----
    n_present = sum(1 for t in traces if t.ground_truth_label)
    n_succeeded = sum(1 for t in traces if t.ground_truth_label and attack_succeeded(t))
    n_case_c = sum(1 for t in traces
                   if _four_outcomes(t, verdicts[t.trace_id].predicted_label) == "case_c")
    scen = {f: len({scenario_of(t) for t in g if scenario_of(t)}) for f, g in by_family.items()}

    rounds = list(csv.DictReader(open(SUMMARY))) if Path(SUMMARY).exists() else []
    breakdown = list(csv.DictReader(open(BREAKDOWN))) if Path(BREAKDOWN).exists() else []

    payload = {
        "meta": {
            "n_traces": len(traces),
            "n_present": n_present,
            "n_succeeded": n_succeeded,
            "n_case_c": n_case_c,
            "families": len(by_family),
            "scenarios": sum(scen.values()),
            "by_family": {f: len(g) for f, g in by_family.items()},
            "scenarios_by_family": scen,
            "curated": len(curated),
        },
        "traces": curated,
        "rounds": rounds,
        "breakdown": breakdown,
        # Static results from the controlled suites — these are authoritative
        # over any single live run and are reported verbatim.
        "generalization": [
            {"tier": "1 · In-distribution", "q": "Attacks like those trained on?",
             "sup": 0.95, "one": 0.87, "hyb": 0.95},
            {"tier": "2 · Strength interpolation", "q": "Unseen parameters, inside the trained range",
             "sup": 0.72, "one": 0.17, "hyb": 0.72},
            {"tier": "2 · Strength extrapolation", "q": "Unseen parameters, weaker than any seen",
             "sup": 0.41, "one": 0.04, "hyb": 0.41},
            {"tier": "3 · Cross-strategy @0.85", "q": "An entirely unseen strategy",
             "sup": 0.05, "one": 0.25, "hyb": 0.25},
            {"tier": "3 · Cross-strategy @0.90", "q": "Unseen strategy, weaker",
             "sup": 0.00, "one": 0.15, "hyb": 0.15},
            {"tier": "Sanity · unseen LOUD strategies", "q": "One-class, zero attack labels",
             "sup": None, "one": 1.00, "hyb": 1.00},
        ],
        "fpr_curve": [
            {"mult": "0.85", "recall": 0.54, "lo": 0.47, "hi": 0.60, "null": False},
            {"mult": "0.90", "recall": 0.27, "lo": 0.21, "hi": 0.33, "null": False},
            {"mult": "0.92", "recall": 0.14, "lo": 0.09, "hi": 0.19, "null": False},
            {"mult": "0.95", "recall": 0.10, "lo": 0.06, "hi": 0.14, "null": False},
            {"mult": "0.97", "recall": 0.03, "lo": 0.01, "hi": 0.06, "null": False},
            {"mult": "1.00", "recall": 0.04, "lo": 0.02, "hi": 0.08, "null": True},
        ],
        "power": [
            {"strength": "0.90", "history": "8 txns (current)", "pred": "18%", "meas": "16%"},
            {"strength": "0.90", "history": "30 txns", "pred": "76%", "meas": "78%"},
            {"strength": "0.90", "history": "60 txns", "pred": "~99%", "meas": "98%"},
            {"strength": "0.95", "history": "8 txns (current)", "pred": "4%", "meas": "2%"},
        ],
        "baseline": [
            {"attack": "Branded Whisper (ASR)", "paper": "100%", "ours": "0%",
             "note": "Our victim model resisted ranking injection entirely"},
            {"attack": "Vault Whisper (exposure)", "paper": "20%", "ours": "100%",
             "note": "Far more vulnerable to identity override"},
        ],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT}  ({kb:.0f} KB)")
    print(f"  {len(traces)} traces -> {len(curated)} curated")
    print(f"  outcome spread: {Counter(c['outcome'] for c in curated)}")


if __name__ == "__main__":
    build()
