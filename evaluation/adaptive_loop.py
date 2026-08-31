"""Adaptive Red/Blue co-evolution loop — the addendum to the Day-2/4/5
baseline reproductions. Runs `src/red_team/evasion.py`'s
`run_population_search` generation over generation for the 3 learnable
families (reasoning_attack via Branded+Vault Whisper sharing one
ReasoningAttackDetector, intent_manipulation, sequence_anomaly), refitting
each family's detector on an accumulating 70/30 train/test pool, plus the
`delegation_abuse` control family (no population search, no fit — it's
already a complete deterministic verifier with nothing to learn).

Per generation, per learnable family:
  1. run_population_search() for attack candidates (deduped against that
     family's AttackMemoryStore) + a small benign batch for FPR.
  2. Route every trace into the family's running train/test pool by a
     stable per-trace_id hash split — test-split traces are NEVER used to
     fit anything, for the life of the run.
  3. Pre-fit check: this generation's new candidates against the
     detector's CURRENT (not yet updated) state — "Before Blue update: MISS".
  4. detector.fit(train_pool) — refit on the full accumulated train pool.
  5. Post-fit recovery check: re-evaluate the same pre-fit Case-C traces —
     count how many flip to caught — "After Blue update: CATCH".
  6. classification_metrics/confusion_breakdown on the full accumulated
     TEST-split pool.
  7. One row appended to the round-by-round summary table.

Ends with a held-out generalization check per learnable family (train on
some attack variants, test on ones never trained on), reusing already-
collected traces — zero new LLM/simulate calls.

Run: PYTHONPATH=. .venv/bin/python -m evaluation.adaptive_loop
"""

from __future__ import annotations

import csv
import hashlib
import os
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from src.blue_team.base import Detector
from src.blue_team.delegation_abuse_detector import DelegationAbuseDetector
from src.blue_team.intent_manipulation_detector import IntentManipulationDetector
from src.blue_team.reasoning_attack_detector import ReasoningAttackDetector
from src.blue_team.sequence_anomaly_detector import SequenceAnomalyDetector
from src.common.feedback import AttackMemory
from src.common.memory import AttackMemoryStore, BlueReplayMemory
from src.common.schemas import AttackTrace
from src.common.scoring import attack_succeeded
from src.common.trace_io import write_traces
from src.red_team.base import RedGenerator
from src.red_team.branded_whisper import INJECTION_TECHNIQUES, BrandedWhisperGenerator
from src.red_team.delegation_abuse import VIOLATION_TYPES, DelegationAbuseGenerator
from src.red_team.evasion import run_population_search
from src.red_team.intent_manipulation import DECOY_TECHNIQUES, IntentManipulationGenerator
from src.common.llm_client import get_call_count, get_call_count_by_model, reset_call_count
from src.red_team.sequence_anomaly import PRESETS, SequenceAnomalyGenerator
from src.red_team.vault_whisper import FRAMING_TECHNIQUES, VaultWhisperGenerator
from evaluation.metrics import classification_metrics, confusion_breakdown

# Overridable via env vars so a smoke run doesn't require editing this file —
# e.g. AL_GENERATIONS=1 AL_POPULATION_SIZE=2 AL_ROUNDS_PER_GEN=1 for the
# smallest smoke configuration.
GENERATIONS = int(os.getenv("AL_GENERATIONS", "3"))
POPULATION_SIZE = int(os.getenv("AL_POPULATION_SIZE", "3"))
ROUNDS_PER_GEN = int(os.getenv("AL_ROUNDS_PER_GEN", "2"))
N_BENIGN_PER_GEN = 3
TRAIN_FRACTION = 0.7

TRACE_PATH = "traces/adaptive_loop_traces.jsonl"
SUMMARY_PATH = "evaluation/results/adaptive_loop_summary.csv"
BREAKDOWN_PATH = "evaluation/results/adaptive_loop_breakdown.csv"
BREAKDOWN_FIELDS = ["generation", "family", "segment", "n_test", "n_caught", "recall"]
CSV_FIELDS = [
    "generation", "family", "n_train_pool", "n_test_pool", "red_asr",
    "blue_recall_test", "blue_fpr_test", "case_c_test", "recovered_case_c", "f1_test",
    "distinct_evasions", "mean_reward", "mean_novelty", "floor_topups",
    "hard_negatives_total", "hard_negatives_in_train",
]

# Addendum 3 — stratified replay floor (see plan file). Independent of Red's
# adaptive search: guarantees each KNOWN segment has at least this many
# examples in train_pool, regardless of what the live population currently
# favors. Smaller for LLM-costed families to bound the added budget; free
# for sequence_anomaly (no LLM dependency).
FLOOR_PER_SEGMENT_SEQUENCE = int(os.getenv("AL_FLOOR_SEQUENCE", "8"))
FLOOR_PER_SEGMENT_REASONING = int(os.getenv("AL_FLOOR_REASONING", "6"))
FLOOR_PER_SEGMENT_INTENT = int(os.getenv("AL_FLOOR_INTENT", "6"))


def is_train(trace_id: str) -> bool:
    return int(hashlib.sha256(trace_id.encode()).hexdigest(), 16) % 100 < int(TRAIN_FRACTION * 100)


def is_train_trace(trace: AttackTrace) -> bool:
    """Split on the LINEAGE ROOT, falling back to trace_id for traces with no
    lineage (benign batches, control family, replay-floor examples).

    Splitting per-trace leaks: run_population_search mutates children from
    surviving parents with a small jitter, so a parent in train and its
    near-identical child in test is effectively train-on-test. Hashing the
    root keeps every descendant of a lineage on the same side."""
    return is_train(trace.metadata.get("root_id") or trace.trace_id)


def _embed_memory(trace: AttackTrace, memory: AttackMemory) -> None:
    trace.metadata["parent_trace_id"] = memory.parent_id
    trace.metadata["memory_parameters"] = memory.parameters
    trace.metadata["novelty_score"] = memory.novelty_score
    trace.metadata["reward"] = memory.reward
    trace.metadata["is_duplicate"] = memory.is_duplicate


class LearnableFamily:
    def __init__(
        self,
        family: str,
        generators: List[RedGenerator],
        detector: Detector,
        segment_key,
        segment_universe: List[str],
        generate_segment_example: Callable[[str], Tuple[Dict[str, Any], AttackTrace, RedGenerator]],
        floor_per_segment: int,
    ):
        self.family = family
        self.generators = generators
        self.detector = detector
        self.segment_key = segment_key  # trace -> str, e.g. preset/technique — for the per-segment breakdown
        self.segment_universe = segment_universe  # the FULL known-segment list, not just what's been observed
        self.generate_segment_example = generate_segment_example  # segment -> (context, trace, generator)
        self.floor_per_segment = floor_per_segment
        self.memory_store = AttackMemoryStore()   # RED: novelty / dedup / distinct evasions
        self.replay_memory = BlueReplayMemory()   # BLUE: training coverage / hard negatives
        self.survivors: Dict[int, List[Dict[str, Any]]] = {}
        self.train_pool: List[AttackTrace] = []
        self.test_pool: List[AttackTrace] = []


def _benign_batch(gen: RedGenerator, generation: int, n: int) -> List[AttackTrace]:
    """Benign traces for FPR tracking. For sequence_anomaly, half the batch is
    LENGTH-MATCHED to the attack traces (a longer benign continuation).

    Without this, every benign example is 14 transactions while low_and_slow
    is 23, so an attack-free 23-txn sequence is out-of-distribution for Blue
    — measured 25% false positives on pure nulls differing from benign ONLY
    in length. That inflates apparent recall and understates FPR, because
    trace length becomes a usable proxy for the label. It is not one."""
    seeds = gen.seed()
    out = []
    for i in range(n):
        # G4: rotate scenarios so benign traces span profiles too. A benign
        # set drawn from ONE profile makes Blue's "normal" far narrower than
        # reality, which inflates apparent separation from attacks.
        ctx = dict(seeds[i % len(seeds)])
        if getattr(gen, "family", None) == "sequence_anomaly" and i % 2 == 1:
            ctx["n_benign_continuation"] = 15  # matches low_and_slow's tail length
        try:
            t = gen.simulate(ctx, benign=True)
        except TypeError:
            return []  # this generator's simulate() has no benign kwarg (VaultWhisperGenerator)
        t.generation = generation
        out.append(t)
    return out


# Explicit parameter GRID for replay-floor generation, walked round-robin.
# NOT one-step jitter: the original floor mutated once off a fixed preset
# default, and `0.85 * uniform(0.95,1.05)` spans 0.8075-0.8925, which the
# memory store's 1-decimal quantization collapsed into ~2 distinct buckets.
# Every later attempt then looked like a near-duplicate and was rejected, so
# the floor filled 1 of 8 requested low_and_slow slots. A grid guarantees
# real, reproducible spread across the strength axis Red actually searches.
_FLOOR_GRID = [
    (0.80, 0.80, 12), (0.85, 0.85, 15), (0.90, 0.90, 18),
    (0.92, 0.92, 15), (0.95, 0.95, 20), (0.97, 0.97, 12),
    (1.05, 1.05, 15), (1.15, 1.15, 12),  # upward variants — attacks aren't only drains
]
_floor_grid_cursor = {"i": 0}


def generate_sequence_segment(segment: str) -> Tuple[Dict[str, Any], AttackTrace, RedGenerator]:
    """Zero LLM cost. Walks _FLOOR_GRID round-robin so successive calls for
    the same segment land on genuinely different parameter regions instead of
    re-drawing the same quantized bucket."""
    gen = SequenceAnomalyGenerator()
    seeds = gen.seed()
    ctx = dict(seeds[_floor_grid_cursor["i"] % len(seeds)])  # G4: vary scenario too
    ctx["preset"] = segment
    amt, spacing, n_tail = _FLOOR_GRID[_floor_grid_cursor["i"] % len(_FLOOR_GRID)]
    _floor_grid_cursor["i"] += 1
    if segment == "low_and_slow":
        # Only this preset is defined by a subtle multiplier; the other two
        # have their own structural signatures (burst / category drift) that
        # these multipliers would distort rather than diversify.
        ctx["amount_multiplier"] = amt
        ctx["spacing_multiplier"] = spacing
        ctx["n_tail_txns"] = n_tail
    else:
        ctx = gen.mutate(ctx)
    trace = gen.simulate(ctx, benign=False)
    return ctx, trace, gen


def generate_reasoning_segment(segment: str) -> Tuple[Dict[str, Any], AttackTrace, RedGenerator]:
    """segment is a sub_attack ("branded_whisper"/"vault_whisper"), matching
    the granularity segment_key already reports at — dispatches to whichever
    generator produces it. Their own mutate()'s internal technique choice is
    left random, matching normal Red behavior."""
    gen: RedGenerator = BrandedWhisperGenerator() if segment == "branded_whisper" else VaultWhisperGenerator()
    seeds = gen.seed()
    ctx = dict(seeds[_floor_grid_cursor["i"] % len(seeds)])  # G4
    ctx = gen.mutate(ctx)
    trace = gen.simulate(ctx, benign=False) if segment == "branded_whisper" else gen.simulate(ctx)
    return ctx, trace, gen


def generate_intent_segment(segment: str) -> Tuple[Dict[str, Any], AttackTrace, RedGenerator]:
    """IntentManipulationGenerator.mutate() unconditionally randomizes
    `technique` (unlike sequence_anomaly's mutate(), it doesn't check
    context.get("technique") first) — forcing a segment here would require
    changing that file, which Addendum 3 explicitly rules out. Instead this
    intentionally re-implements just the technique-forced slice of mutate()
    (same prompt template, same category/price post-processing, imported
    from the module rather than copy-pasted) with no evasion-constraint
    branch, since a floor example isn't reacting to any prior detection —
    its only job is being a representative, real example of `segment`."""
    from src.red_team.intent_manipulation import _DECOY_PROMPT, _parse_listing
    from src.common.llm_client import RED_MODEL, chat

    gen = IntentManipulationGenerator()
    seeds = gen.seed()
    ctx = dict(seeds[_floor_grid_cursor["i"] % len(seeds)])  # G4
    prompt = _DECOY_PROMPT.format(
        technique=segment,
        raw_user_statement=ctx["raw_user_statement"],
        category=ctx["category"],
        brand=ctx["brand"],
        max_amount=ctx["max_amount"],
    )
    rewritten = chat(
        messages=[{"role": "user", "content": prompt}], model=RED_MODEL, temperature=0.9, max_tokens=400
    ).strip()
    title, description = _parse_listing(rewritten, ctx["decoy_title"], ctx["decoy_description"])
    ctx["technique"] = segment
    ctx["decoy_title"] = title
    ctx["decoy_description"] = description
    ctx["decoy_category"] = ctx["category"]
    if segment == "category_confusion":
        ctx["decoy_category"] = "accessories"
    elif segment == "price_illusion":
        ctx["decoy_price"] = round(ctx["max_amount"] * 0.98, 2)
    trace = gen.simulate(ctx, benign=False)
    return ctx, trace, gen


def top_up_replay_floor(fam: LearnableFamily, generation: int) -> List[AttackTrace]:
    """Addendum 3: independent of Red's adaptive search — guarantees every
    KNOWN segment has at least `fam.floor_per_segment` examples in
    train_pool, regardless of what the live population currently favors.
    Never touches test_pool. Kept in a list SEPARATE from the generation's
    `new_traces` until the very end (see run_generation) so red_asr/
    mean_reward/mean_novelty — meant to characterize what Red's adaptive
    search itself discovered — are structurally unaffected by this
    scaffolding, not just filtered by convention."""
    existing_counts = Counter(fam.segment_key(t) for t in fam.train_pool if t.ground_truth_label)
    topped_up: List[AttackTrace] = []
    for segment in fam.segment_universe:
        deficit = fam.floor_per_segment - existing_counts.get(segment, 0)
        for _ in range(max(0, deficit)):
            _context, trace, _gen = fam.generate_segment_example(segment)
            trace.generation = generation
            trace.metadata["source"] = "replay_floor"
            # NO dedup check, and nothing recorded into fam.memory_store.
            # That store is RED's discovery memory: its whole job is to
            # suppress similarity so novelty and distinct-evasion counts mean
            # something. Blue's training wants the opposite — coverage and
            # mass, where near-duplicates are harmless. Routing floor traces
            # through Red's dedup is exactly what broke the first version
            # (1 of 8 low_and_slow slots filled). Floor traces are already
            # excluded from red_asr/mean_reward/mean_novelty/distinct_evasions,
            # so they cannot inflate any Red-side metric by being here.
            fam.replay_memory.record(segment, trace.trace_id)
            topped_up.append(trace)
    return topped_up


def run_generation(
    fam: LearnableFamily, generation: int, csv_rows: List[Dict[str, Any]], breakdown_rows: List[Dict[str, Any]]
) -> None:
    new_traces: List[AttackTrace] = []
    memory_by_id: Dict[str, AttackMemory] = {}

    for gen_obj in fam.generators:
        # G4: rotate the starting scenario by generation so a multi-generation
        # run covers several profiles instead of re-running one. Survivors are
        # still carried forward, so lineages that already exist keep evolving —
        # this only changes where NEW lineages start.
        all_seeds = gen_obj.seed()
        seed_ctx = all_seeds[(generation - 1) % len(all_seeds)]
        prior = fam.survivors.get(id(gen_obj))
        gres = run_population_search(
            gen_obj, fam.detector, fam.memory_store, seed_ctx, generation,
            population_size=POPULATION_SIZE, rounds=ROUNDS_PER_GEN, prior_survivors=prior,
        )
        fam.survivors[id(gen_obj)] = gres.survivors
        for trace, memory in zip(gres.all_traces, gres.all_memories):
            _embed_memory(trace, memory)
            memory_by_id[trace.trace_id] = memory
        new_traces.extend(gres.all_traces)
        new_traces.extend(_benign_batch(gen_obj, generation, N_BENIGN_PER_GEN))

    # pre-fit verdicts (detector state BEFORE this generation's fit())
    pre_fit_verdicts = {t.trace_id: fam.detector.evaluate(t) for t in new_traces}
    case_c_before = [
        t for t in new_traces
        if t.ground_truth_label and attack_succeeded(t) and not pre_fit_verdicts[t.trace_id].predicted_label
    ]
    # Item 6 — hard-negative curriculum. A Case-C trace (attack SUCCEEDED and
    # Blue MISSED it) is precisely the example Blue most needs to learn from:
    # Red found a live blind spot. Tag them so their contribution is
    # auditable, then let the normal lineage split route them.
    #
    # Integrity constraint: a hard negative is a TRAINING signal supporting an
    # in-distribution adaptive claim ("Blue closes blind spots Red discovers"),
    # never a generalization claim. It must never be force-fed into the train
    # pool against its lineage assignment — doing so would put a near-twin of
    # a test trace into training. The split below is left untouched on
    # purpose; `recovered_case_c` then measures whether Blue actually closed
    # the ones it was allowed to see.
    for t in case_c_before:
        t.metadata["hard_negative"] = True
        fam.replay_memory.mark_hard_negative(t.trace_id)

    for t in new_traces:
        (fam.train_pool if is_train_trace(t) else fam.test_pool).append(t)

    # Addendum 3: replay-floor top-up — train_pool only, computed AFTER the
    # adaptive routing above (so it reacts to this generation's real
    # shortfall) and kept in its own list until the final write, so it never
    # touches red_asr/mean_reward/mean_novelty below.
    floor_traces = top_up_replay_floor(fam, generation)
    fam.train_pool.extend(floor_traces)

    if fam.detector.trainable:
        fam.detector.fit(fam.train_pool)
        # One-class calibration on the TRAIN pool's attack-free traces only.
        # Kept strictly separate from fit(): this is what gives the detector
        # a floor against strategies Red's population has drifted away from
        # (or never produced), which a label-supervised boundary cannot have.
        null_train = [t for t in fam.train_pool if not t.ground_truth_label]
        if null_train:
            fam.detector.calibrate(null_train)

    recovered_case_c = sum(1 for t in case_c_before if fam.detector.evaluate(t).predicted_label)

    test_verdicts = [fam.detector.evaluate(t) for t in fam.test_pool]
    cm = classification_metrics(fam.test_pool, test_verdicts)
    cb = confusion_breakdown(fam.test_pool, test_verdicts)

    attack_candidates = [t for t in new_traces if t.ground_truth_label]
    red_asr = (
        sum(1 for t in attack_candidates if attack_succeeded(t)) / len(attack_candidates)
        if attack_candidates else 0.0
    )

    verdict_by_id = {v.trace_id: v for v in test_verdicts}
    case_c_test_traces = [
        t for t in fam.test_pool
        if t.ground_truth_label and attack_succeeded(t) and not verdict_by_id[t.trace_id].predicted_label
    ]
    distinct_evasions = sum(
        1 for t in case_c_test_traces if not memory_by_id.get(t.trace_id, AttackMemory(
            attack_id="", parent_id=None, family=fam.family, generation=generation, parameters={},
            attack_present=True, attack_succeeded=True, blue_detected=False, blue_score=0.0,
        )).is_duplicate
    )

    rewards = [t.metadata.get("reward") for t in new_traces if t.metadata.get("reward") is not None]
    novelties = [t.metadata.get("novelty_score") for t in new_traces if t.metadata.get("novelty_score") is not None]

    csv_rows.append({
        "generation": generation, "family": fam.family,
        "n_train_pool": len(fam.train_pool), "n_test_pool": len(fam.test_pool),
        "red_asr": round(red_asr, 4), "blue_recall_test": round(cm.recall, 4),
        "blue_fpr_test": round(cb.false_positive_rate, 4),
        "case_c_test": cb.case_c, "recovered_case_c": recovered_case_c,
        "f1_test": round(cm.f1, 4), "distinct_evasions": distinct_evasions,
        "mean_reward": round(statistics.mean(rewards), 4) if rewards else 0.0,
        "mean_novelty": round(statistics.mean(novelties), 4) if novelties else 0.0,
        "floor_topups": len(floor_traces),
        "hard_negatives_total": fam.replay_memory.n_hard_negatives,
        # How many of Red's discovered blind spots Blue was actually ALLOWED
        # to train on. The gap between these two columns is the lineage
        # split doing its job, not data being lost.
        "hard_negatives_in_train": sum(
            1 for t in fam.train_pool if t.metadata.get("hard_negative")
        ),
    })

    # Per-segment (preset/technique) recall on the TEST split, cumulative —
    # disentangles "Blue got worse" from "the test mix shifted toward the
    # segment Blue was already worst at" (see item #6 of the diagnostic
    # pass: recall trends must be checked against segment composition, not
    # read as pure capability change).
    segments: Dict[str, List[AttackTrace]] = {}
    for t in fam.test_pool:
        if not t.ground_truth_label:
            continue
        segments.setdefault(fam.segment_key(t) or "unknown", []).append(t)
    for seg, seg_traces in segments.items():
        seg_verdicts = [verdict_by_id[t.trace_id] for t in seg_traces]
        n_caught = sum(1 for v in seg_verdicts if v.predicted_label)
        breakdown_rows.append({
            "generation": generation, "family": fam.family, "segment": seg,
            "n_test": len(seg_traces), "n_caught": n_caught,
            "recall": round(n_caught / len(seg_traces), 4) if seg_traces else 0.0,
        })

    write_traces(new_traces + floor_traces, TRACE_PATH, mode="a")


def run_control_generation(gen: DelegationAbuseGenerator, det: DelegationAbuseDetector,
                            generation: int, train_pool: List[AttackTrace], test_pool: List[AttackTrace],
                            csv_rows: List[Dict[str, Any]]) -> None:
    seeds = gen.seed()
    seed_ctx = seeds[(generation - 1) % len(seeds)]  # G4
    new_traces = [gen.simulate(dict(seeds[i % len(seeds)]), benign=True) for i in range(3)]
    for violation_type in VIOLATION_TYPES:
        ctx = dict(seed_ctx)
        ctx["violation_type"] = violation_type
        new_traces.append(gen.simulate(ctx, benign=False))
    for t in new_traces:
        t.generation = generation
        (train_pool if is_train_trace(t) else test_pool).append(t)

    test_verdicts = [det.evaluate(t) for t in test_pool]
    cm = classification_metrics(test_pool, test_verdicts)
    cb = confusion_breakdown(test_pool, test_verdicts)
    attack_candidates = [t for t in new_traces if t.ground_truth_label]
    red_asr = (
        sum(1 for t in attack_candidates if attack_succeeded(t)) / len(attack_candidates)
        if attack_candidates else 0.0
    )

    csv_rows.append({
        "generation": generation, "family": "delegation_abuse (control — provably complete, no fit)",
        "n_train_pool": len(train_pool), "n_test_pool": len(test_pool),
        "red_asr": round(red_asr, 4), "blue_recall_test": round(cm.recall, 4),
        "blue_fpr_test": round(cb.false_positive_rate, 4),
        "case_c_test": cb.case_c, "recovered_case_c": 0, "f1_test": round(cm.f1, 4), "distinct_evasions": 0,
        "mean_reward": "n/a", "mean_novelty": "n/a", "floor_topups": 0,
        "hard_negatives_total": 0, "hard_negatives_in_train": 0,
    })
    write_traces(new_traces, TRACE_PATH, mode="a")


def run_generalization_check(fam: LearnableFamily, held_in, held_out, key: str) -> Tuple[int, int]:
    """Fresh (never-fit) detector of fam's class, trained on pooled traces
    matching `held_in` PLUS every benign (ground_truth_label=False) trace —
    fit() needs both classes or its degenerate-pool guard silently skips
    fitting and evaluate() falls back to the untrained heuristic, which
    would make this check meaningless — tested on pooled attack traces
    matching `held_out`. Returns (n_test, n_caught), or (0, 0) if the fit
    was degenerate (e.g. too few benign examples collected). Reuses
    already-collected traces, zero new calls."""
    pool = fam.train_pool + fam.test_pool
    train = [t for t in pool if (not t.ground_truth_label) or (t.metadata.get(key) in held_in)]
    test = [t for t in pool if t.ground_truth_label and t.metadata.get(key) in held_out]
    if not train or not test:
        return (0, 0)
    fresh = type(fam.detector)()
    fresh.fit(train)
    if getattr(fresh, "_clf", None) is None:
        return (0, 0)  # degenerate fit — not a fair generalization check
    # Calibrate the one-class layer too, so this check measures the ARCHITECTURE
    # actually deployed rather than the supervised half alone. Legitimate here:
    # calibration sees only attack-free traces, never the held-out strategy, so
    # it cannot leak the thing being tested. This matters because the supervised
    # half is expected to score ~0 on a strategy absent from its training pool
    # (its deciding coefficient is unidentified there) — reporting only that
    # would understate the deployed detector.
    null_train = [t for t in train if not t.ground_truth_label]
    if null_train:
        fresh.calibrate(null_train)
    caught = sum(1 for t in test if fresh.evaluate(t).predicted_label)
    return (len(test), caught)


def main() -> None:
    Path("traces").mkdir(exist_ok=True)
    Path("evaluation/results").mkdir(parents=True, exist_ok=True)
    write_traces([], TRACE_PATH, mode="w")  # truncate once, here — every later write is mode="a"
    reset_call_count()

    reasoning_attack = LearnableFamily(
        "reasoning_attack", [BrandedWhisperGenerator(), VaultWhisperGenerator()], ReasoningAttackDetector(),
        segment_key=lambda t: t.sub_attack,
        segment_universe=["branded_whisper", "vault_whisper"],
        generate_segment_example=generate_reasoning_segment,
        floor_per_segment=FLOOR_PER_SEGMENT_REASONING,
    )
    intent_manipulation = LearnableFamily(
        "intent_manipulation", [IntentManipulationGenerator()], IntentManipulationDetector(),
        segment_key=lambda t: t.metadata.get("technique"),
        segment_universe=list(DECOY_TECHNIQUES),
        generate_segment_example=generate_intent_segment,
        floor_per_segment=FLOOR_PER_SEGMENT_INTENT,
    )
    sequence_anomaly = LearnableFamily(
        "sequence_anomaly", [SequenceAnomalyGenerator()], SequenceAnomalyDetector(),
        segment_key=lambda t: t.metadata.get("preset"),
        segment_universe=list(PRESETS),
        generate_segment_example=generate_sequence_segment,
        floor_per_segment=FLOOR_PER_SEGMENT_SEQUENCE,
    )
    learnable = [reasoning_attack, intent_manipulation, sequence_anomaly]

    delegation_gen = DelegationAbuseGenerator()
    delegation_det = DelegationAbuseDetector()
    delegation_train: List[AttackTrace] = []
    delegation_test: List[AttackTrace] = []

    csv_rows: List[Dict[str, Any]] = []
    breakdown_rows: List[Dict[str, Any]] = []

    for generation in range(1, GENERATIONS + 1):
        print(f"=== Generation {generation}/{GENERATIONS} ===")
        for fam in learnable:
            run_generation(fam, generation, csv_rows, breakdown_rows)
            print(f"  {fam.family}: {csv_rows[-1]}")
        run_control_generation(delegation_gen, delegation_det, generation, delegation_train, delegation_test, csv_rows)
        print(f"  delegation_abuse (control): {csv_rows[-1]}")
        print(f"  [LLM calls so far: {get_call_count()} — {get_call_count_by_model()}]")

        with open(SUMMARY_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(csv_rows)
        with open(BREAKDOWN_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=BREAKDOWN_FIELDS)
            writer.writeheader()
            writer.writerows(breakdown_rows)

    print(f"\nWrote round-by-round summary to {SUMMARY_PATH}")
    print(f"Wrote per-segment recall breakdown to {BREAKDOWN_PATH}")
    print(f"Total LLM calls this run: {get_call_count()} — by model: {get_call_count_by_model()}")

    print(
        "\n=== Robustness against attack-strength variation (FINAL LIVE sequence_anomaly "
        "detector — the one this run actually retrained across all generations; NOT a "
        "held-out/generalization check) ==="
    )
    if getattr(sequence_anomaly.detector, "_clf", None) is not None:
        probe_gen = SequenceAnomalyGenerator()
        probe_seed = probe_gen.seed()[0]
        random.seed(2026083001)
        for mult in (0.85, 0.90, 0.92, 0.95, 0.97, 1.00):
            batch = []
            for _ in range(15):
                ctx = dict(probe_seed)
                ctx["preset"] = "low_and_slow"
                # BOTH levers move together. Sweeping amount while pinning
                # spacing_multiplier at 0.9 (as this did originally) leaves a
                # constant cadence signal in every sample, so the "weakest"
                # point is not actually a weak attack and the curve reads
                # higher than the truth — an earlier version of this sweep
                # reported 0.60 recall at multiplier 1.00, which was the
                # detector reading leftover cadence, not amount.
                ctx["amount_multiplier"] = mult
                ctx["spacing_multiplier"] = mult
                ctx["n_tail_txns"] = 15
                batch.append(probe_gen.simulate(ctx, benign=False))
            verdicts = [sequence_anomaly.detector.evaluate(t) for t in batch]
            rate = sum(1 for v in verdicts if v.predicted_label) / len(verdicts)
            # At multiplier 1.00 BOTH levers are neutral, so this batch is a
            # pure NULL — statistically identical to benign, differing only in
            # tail length. Its flag rate is a FALSE-POSITIVE rate, not recall,
            # and labelling it "recall" would read as detection of an attack
            # that isn't there.
            label = "FPR (null control)" if mult >= 1.0 else "recall"
            print(f"  amount_multiplier={mult:.2f}: {label}={rate:.2f} (n={len(verdicts)})")
    else:
        print("  sequence_anomaly detector never successfully fit this run — skipping robustness sweep")
    print(
        "  NOTE: this measures in-distribution robustness of the detector actually deployed "
        "by this run. It is NOT evidence of generalization — see the held-out check below, "
        "which uses a fresh, never-live-trained detector on presets never trained on at all."
    )

    print("\n=== Held-out generalization checks (reusing collected traces, 0 new LLM calls) ===")
    sa_held_out = {"low_and_slow"}
    sa_held_in = set(PRESETS) - sa_held_out
    n_test, caught = run_generalization_check(
        sequence_anomaly, held_in=sa_held_in, held_out=sa_held_out, key="preset",
    )
    print(f"sequence_anomaly: trained on {sorted(sa_held_in)} only -> "
          f"caught {caught}/{n_test} previously-unseen low_and_slow examples")

    ra_held_in = {INJECTION_TECHNIQUES[0], INJECTION_TECHNIQUES[1], FRAMING_TECHNIQUES[0], FRAMING_TECHNIQUES[1]}
    ra_held_out = (set(INJECTION_TECHNIQUES) | set(FRAMING_TECHNIQUES)) - ra_held_in
    n_test, caught = run_generalization_check(
        reasoning_attack, held_in=ra_held_in, held_out=ra_held_out, key="technique",
    )
    print(f"reasoning_attack: trained on {sorted(ra_held_in)} -> "
          f"caught {caught}/{n_test} previously-unseen technique examples")

    im_held_out = {"price_illusion"}
    im_held_in = set(DECOY_TECHNIQUES) - im_held_out
    n_test, caught = run_generalization_check(
        intent_manipulation, held_in=im_held_in, held_out=im_held_out, key="technique",
    )
    print(f"intent_manipulation (thin, 3-bucket experiment): trained on 2 techniques -> "
          f"caught {caught}/{n_test} previously-unseen price_illusion examples")


if __name__ == "__main__":
    main()
