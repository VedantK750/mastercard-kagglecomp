# Agentic-Commerce Fraud Red-Team Repo — Mastercard Innovation Challenge 2026

## Context

Submission for the Mastercard Innovation Challenge 2026 (GFF, Sept 9–11 Mumbai). The brief asks for
an end-to-end **Identify → Generate → Defend** red-team/blue-team system covering GenAI-powered
payment fraud, judged on diversity, fidelity, detection efficacy, novelty, and real-world feasibility.
`/home/krish/MasterCard_Kaggle` is currently empty except two research papers (already digested this
session) and a `.env` with live LLM credentials — this is a from-scratch build with a **hard 7-day
deadline**.

Two papers ground the design directly:
- **"Protocol-Level Attacks on Agentic Commerce Platforms"** — RC-1..RC-6 taxonomy of structural
  (model-independent) vs. semantic (model-dependent) attacks, plus the PCAT defense pattern (P1
  response-integrity, P2 caller-identity/DID binding, P3 secure-channel, P4 atomic payment state, P5
  tool-call authorization). This taxonomy grounds family **1.D** (see §3).
- **"Whispers of Wealth"** — red-teams Google's AP2 at the reasoning layer using a real four-agent
  implementation (Gemini-2.5-Flash + Google ADK), and is now the literal **foundation** of family
  **1.A**, not just a citation. Its core finding: cryptographic mandate signing protects *execution*,
  not the *decision* that constructs the mandate — Branded Whisper (indirect injection, 100% ASR/10
  trials) and Vault Whisper (direct injection, 20% success/10 trials) both produce cryptographically
  valid mandates despite manipulated outcomes. The paper explicitly names its own limitation —
  manually crafted prompts, 10 trials, no automated generation — as future work. **That gap is this
  project's core novelty claim**: automate what the paper did by hand, at scale, adaptively, against
  an evolving Blue detector.

Scope decisions made across this conversation, most recent first:
1. **The AP2 paper's exact four-agent architecture and mandate chain becomes the 1.A simulator's
   foundation** (§2), not a generic shopping-agent abstraction.
2. **Branded Whisper and Vault Whisper are reproduced as literal baseline attacks** (§3, 1.A.1/1.A.2),
   with our own measured ASR reported alongside the paper's 100%/20% figures — then used as seed
   attacks for automated Red-generated variants.
3. Family **1.A is reframed as "Agentic Reasoning Attacks"**, an umbrella covering both attacks that
   manipulate *transactions* (Branded Whisper, intent drift) and attacks that manipulate *data access*
   (Vault Whisper) — two distinct Red objectives, not one.
4. The paper's own named future-work threats (cross-agent message tampering, mandate replay,
   compromised intermediaries, data poisoning) are mapped onto the existing family roadmap (§3) rather
   than treated as a disconnected addition.
5. Depth-first on agentic-commerce (1.A–1.D), LLM generation via a custom OpenAI-SDK-compatible
   endpoint (`.env`: `LLM_API_KEY`, `LLM_ENDPOINT=https://aicredits.in/v1`), PaySim as the base
   legitimate-distribution dataset if/when Tier-2 work is reached, and a specific 6-phase build order
   (§4) that proves the Red-vs-Blue loop on the simplest family first, then layers modalities on top
   of the shared harness.

## 1. Repo Structure

```
/home/krish/MasterCard_Kaggle/
├── .env                       # gitignored — LLM_API_KEY, LLM_ENDPOINT (never commit)
├── .gitignore
├── README.md                  # includes the paper-vs-us novelty comparison (§3.1)
├── requirements.txt
├── docs/identify/
│   ├── 00_taxonomy_overview.md        # master table + paper future-work -> family roadmap mapping (§3.5)
│   ├── 01_agentic_reasoning_attacks.md    # 1.A.1-1.A.5, the AP2 paper reproduction + extensions
│   ├── 02_detector_evasion.md
│   ├── 03_intent_manipulation.md
│   ├── 04_delegation_authorization_abuse.md   # cites Paper 1's RC-1..RC-5 as grounding
│   ├── 05_sequence_anomaly.md
│   ├── 06_multi_agent.md
│   ├── 07_synthetic_identity.md
│   ├── 08_attack_composer.md
│   └── 09_kyc_deepfake_voice.md
├── src/
│   ├── common/
│   │   ├── schemas.py         # single pydantic model set, AP2-native (§2)
│   │   ├── ap2_env.py         # ShoppingAgent, MerchantAgent, MerchantPaymentProcessorAgent, CredentialsProviderAgent + mandate chain
│   │   ├── llm_client.py      # openai client wrapped around LLM_ENDPOINT
│   │   └── trace_io.py        # JSONL read/write for AttackTrace
│   ├── red_team/
│   │   ├── base.py            # RedGenerator ABC: seed() -> mutate() -> simulate() -> score()
│   │   ├── evasion.py         # detector-evasion loop wrapper (§4 Phase 1)
│   │   ├── branded_whisper.py     # 1.A.1 — paper reproduction + auto variant generation
│   │   ├── vault_whisper.py       # 1.A.2 — paper reproduction + auto variant generation
│   │   ├── reasoning_attacks.py   # 1.A.3-1.A.5: intent drift / context poisoning / cross-agent injection (stretch, shares AP2 chain machinery)
│   │   ├── intent_manipulation.py # 1.C — non-injection-triggered variant, same IntentMatchScore
│   │   ├── delegation_abuse.py
│   │   ├── sequence_anomaly.py
│   │   ├── multi_agent.py
│   │   ├── synthetic_identity.py
│   │   └── composer.py
│   ├── blue_team/
│   │   ├── base.py             # Detector ABC: evaluate(trace) -> BlueVerdict
│   │   ├── reasoning_attack_detector.py   # serves 1.A.1-1.A.5, dual objective (payment vs. data-exposure)
│   │   ├── intent_manipulation_detector.py
│   │   ├── delegation_abuse_detector.py
│   │   ├── sequence_anomaly_detector.py
│   │   └── unified_pipeline.py
│   └── data/seed_content/
├── notebooks/
│   ├── 01_branded_and_vault_whisper_baseline.ipynb   # reproduces the paper's Table 1 first, then auto-generates variants
│   ├── 02_reasoning_attacks_and_evasion.ipynb
│   ├── 03_intent_and_delegation.ipynb
│   ├── 04_sequence_anomaly.ipynb
│   └── 99_full_pipeline_eval.ipynb
├── evaluation/
│   ├── metrics.py              # Precision/Recall/F1/AUC + Wilson CIs (matches both papers' reporting convention)
│   ├── adaptive_loop.py        # BASELINE -> RED GEN -> AP2 SIM -> BLUE DETECT -> RED EVOLVE -> BLUE RETRAIN -> UNSEEN-ATTACK EVAL (§3.4)
│   └── results/
└── traces/
    └── sample_attack_traces.jsonl
```

## 2. Shared Substrate — AP2-Native (build Day 1)

The simulator models the paper's actual architecture, not a generic marketplace: four agents and a
three-mandate chain, because the entire point of both Whisper attacks is that **the mandate chain
signs successfully regardless of the attack** — that has to be a first-class, observable fact in the
trace, not an implementation detail we abstract away.

`src/common/ap2_env.py`:
```python
class ShoppingAgent: ...            # user-facing coordinator, captures intent, orchestrates
class MerchantAgent: ...            # product discovery/pricing, returns candidate products
class MerchantPaymentProcessorAgent: ...  # verifies/processes the signed Payment Mandate
class CredentialsProviderAgent: ...       # manages/retrieves user credentials + shipping data
```
Workflow: `User intent -> Intent Mandate -> Product Selection (Shopping<->Merchant Agent) -> Cart
Mandate -> Payment Mandate -> Payment Execution (Merchant Payment Processor Agent)`, with the
Credentials Provider Agent invoked during information gathering.

`src/common/schemas.py` (pydantic v2), extended for the AP2 chain and dual Red objectives:

```python
class Mandate(BaseModel):
    mandate_type: Literal["intent", "cart", "payment"]
    content_hash: str
    signature_valid: bool          # stays True under successful attacks — this IS the paper's finding
    approved_by_user: bool

class IntentObject(BaseModel):
    category: str; brand: Optional[str]; max_amount: float
    quantity: int; geography: Optional[str]; urgency: Optional[str]
    raw_user_statement: str

class ExternalContentItem(BaseModel):
    source_url: str; text: str
    contains_injection: bool
    injection_technique: Optional[str]   # e.g. "hidden_ranking_directive" (Branded Whisper), "identity_override" (Vault Whisper)
    hop_index: int                       # which AP2 agent touched this content: 0=Merchant, 1=Shopping, 2=CredentialsProvider, 3=PaymentProcessor

class DelegationEdge(BaseModel):
    from_agent: str; to_agent: str
    allowed_categories: List[str]; max_amount: float
    merchant_category_codes: List[str]
    valid_from: datetime; valid_until: datetime; purpose: str
    trust_weight: float = 1.0

class AuthorizationGraph(BaseModel):
    nodes: List[str]; edges: List[DelegationEdge]

class Transaction(BaseModel):
    txn_id: str; agent_id: str; merchant_id: str
    amount: float; category: str; timestamp: datetime
    executing_authorization_edge: Optional[str]

class RedScore(BaseModel):
    intent_deviation: float; payment_impact: float
    realism: float; novelty: float; detection_probability: float
    r_red: float

class AttackTrace(BaseModel):
    trace_id: str
    family: Literal["reasoning_attack", "intent_manipulation", "delegation_abuse",
                     "sequence_anomaly", "multi_agent", "synthetic_identity", "composite"]
    sub_attack: Optional[str] = None     # "branded_whisper" | "vault_whisper" | "intent_drift" |
                                          # "context_poisoning" | "cross_agent_injection" | None
    objective: Literal["payment_manipulation", "data_exposure"] = "payment_manipulation"
    injection_channel: Optional[Literal["indirect_external_content", "direct_user_prompt"]] = None
    ground_truth_label: bool
    user_intent: IntentObject
    external_content: List[ExternalContentItem] = []
    mandates: List[Mandate] = []
    authorization_graph: Optional[AuthorizationGraph] = None
    agent_reasoning_trace: List[Dict] = []
    transactions: List[Transaction]
    final_transaction: Transaction
    exposed_data: Optional[Dict[str, Any]] = None       # populated only for objective="data_exposure"
    cross_user_exposure: bool = False                    # ground truth for Vault-Whisper-style traces
    red_score: RedScore
    evasion_rounds: List[RedScore] = []
    composed_of: List[str] = []
    metadata: Dict[str, Any] = {}
```

`injection_channel` is what distinguishes Branded Whisper (`indirect_external_content` — attacker
controls product metadata, has no AP2 signatures/credentials/payment infra) from Vault Whisper
(`direct_user_prompt` — attacker interacts normally with the Shopping Agent and tries to override
identity/data-access constraints). `objective` is what distinguishes "did the transaction get
manipulated" from "was sensitive data disclosed" — Blue's verdict shape differs accordingly (§3).

## 3. Per-Family Design

### 3.1 1.A — Agentic Reasoning Attacks (the flagship family; paper-grounded)

Everything here targets the pre-signature reasoning layer of the AP2 chain — the mandates that come
out remain cryptographically valid (`signature_valid=True`) regardless of outcome, exactly matching
the paper's Figure 1 (execution integrity preserved, decision integrity violated). Sub-attacks, in
priority order:

- **1.A.1 Branded Whisper (must-build baseline).** Attacker = malicious merchant, controls product
  descriptions only (no signatures/credentials/payment infra — model this as the adversary's
  capability set explicitly, matching the paper). `seed()`: reproduce the paper's exact setup — one
  fixed task ("basketball shoes for outdoor use" or equivalent), a hidden ranking directive embedded
  in one candidate product's metadata. Run it for the same shape of trial the paper used (10 trials)
  first as a literal reproduction, reporting our ASR next to their 100% in the README/notebook. Then
  `mutate()` takes over: LLM generates variant injection techniques/framings, scored via
  `IntentMatchScore` + a ranking-position check, and the trial count is pushed to 100+ per condition
  with Wilson CIs — this volume/automation delta *is* the novelty pitch (§3.4).
- **1.A.2 Vault Whisper (must-build baseline).** Attacker = the user themself, `injection_channel =
  "direct_user_prompt"`, `objective = "data_exposure"`. Reproduce the paper's setup: a crafted prompt
  attempts to override identity constraints during the Shopping Agent → Credentials Provider Agent
  handoff, targeting cross-account data. Reproduce the paper's ~10-trial baseline (they saw 20%
  full exposure + 30% partial), then auto-generate prompt variants the same way as 1.A.1. Ground
  truth is `cross_user_exposure: bool` + `exposed_data`, not a payment transaction — Blue's verdict
  here is "did this request get identity-re-verified before data left the Credentials Provider
  Agent," not a fraud score.
- **1.A.3 Intent drift via injection.** Same mechanic as Branded Whisper but scored purely on
  `IntentMatchScore` divergence rather than "ranked first" — the injected content changes *what* gets
  bought without necessarily making it the top-ranked/most-visible option. Cheap once 1.A.1 exists
  (shares its generator, different scoring target).
- **1.A.4 Context poisoning / 1.A.5 Cross-agent injection (stretch, build last within Phase 1).**
  Both are the same underlying mechanism — poisoned content introduced at one AP2 agent (`hop_index`
  0 = Merchant) that a *downstream* agent (`hop_index` 2/3 = Credentials Provider / Payment
  Processor) acts on without re-validation — formalized once `ap2_env.py`'s four-agent chain exists,
  since `hop_index` already encodes "which agent touched this." Build only if 1.A.1–1.A.3 and their
  Blue detectors are solid; these two are the first thing to cut if Phase 1 runs long.

Family-level note on the apparent overlap with 1.C: 1.A.3 is intent manipulation *caused by
adversarial injected content*; 1.C (below) is intent manipulation that Red produces *without* any
injection (ambiguous catalogs, unit/SKU confusion) — same `IntentMatchScore` machinery and same Blue
detector, distinguished only by `sub_attack`/`injection_channel` being populated or not. One codepath,
not two.

### 3.2 1.C — Intent Manipulation (non-injection trigger)

`seed()`/`mutate()`: LLM generates a plausible-but-partly-wrong product catalog (varying brand/category
while honoring the price ceiling) with no adversarial instruction embedded — this is a catalog/
reasoning-ambiguity attack, not a content-injection attack. `score()` reuses `IntentMatchScore`
(weighted match over category/brand/price/quantity/merchant-category/timing) — the exact function
1.A.3 also uses.

### 3.3 1.D — Delegation / Authorization Abuse (protocol-layer attacks live here)

Fully deterministic, no LLM. `mutate()` applies scope violations to an `AuthorizationGraph` (category
mismatch, wrong executing agent, expired window, amount exceeded). Blue is a pure policy verifier:
`ValidAuthorization = Identity AND Scope AND Purpose AND Time AND Amount AND DelegationChain`. This
family is explicitly where **Paper 1's RC-1..RC-5 structural taxonomy lives** (per the user's
decision not to make protocol exploitation its own top-level family) — `docs/identify/
04_delegation_authorization_abuse.md` should cite RC-2 (untrusted payment destination), RC-4
(TOCTOU/mandate-replay), and RC-5 (authorization scope not enforced) as the structural grounding, and
the deterministic verifier documents that it follows the PCAT P2/P5 *pattern* (no PCAT code exists to
copy — cite as prior art only).

### 3.4 Evaluation Philosophy — Evolved from the Paper's Own Method

The paper's evaluation loop is: `Baseline -> Attack condition -> Repeated trials -> ASR / exposure
rate`, on synthetic data in an isolated environment. `evaluation/adaptive_loop.py` extends this
exact shape rather than inventing a new one:

```
BASELINE -> RED GENERATION -> AP2 SIMULATION -> BLUE DETECTION -> ATTACK SUCCESS
         -> RED EVOLUTION -> BLUE RETRAINING -> UNSEEN-ATTACK EVALUATION
```

`RED EVOLUTION` is `red_team/evasion.py` (§4 Phase 1) run against the *real* Blue detector for that
family, not a heuristic proxy. `UNSEEN-ATTACK EVALUATION` is the step the paper doesn't have and we
should: hold out a slice of Red-generated variants from Blue's tuning, and report whether Blue
generalizes or memorizes — this is the single most defensible "novelty" claim in the submission and
belongs prominently in the flagship notebook and README, stated plainly as:

> The source paper hand-crafted 1-2 attacks per class and measured them over 10 trials. This system
> automates generation, mutation, adaptive evasion, and composition of those same attack classes, at
> 10x+ the trial volume, against a Blue detector that itself evolves across generations — directly
> answering the paper's own stated "automated adversarial generation and larger-scale testing" future
> work.

### 3.5 Paper Future-Work → Family Roadmap (documented in `docs/identify/00_taxonomy_overview.md`)

| Paper's named future threat        | Maps to               |
|-------------------------------------|------------------------|
| Cross-agent message tampering       | Multi-agent (Phase 4) |
| Compromised intermediary agent      | 1.D / Phase 4.A        |
| Mandate replay across trust domains | 1.D (protocol layer)   |
| Data poisoning of downstream reasoning | 1.A.4/1.A.5 / Phase 4.B |

This table is what makes the taxonomy read as "extending the AP2 paper into an adaptive framework"
rather than a disconnected list — include it even for rows whose family only ships as a docs-only
writeup (§4).

### 3.6 Remaining Families (unchanged from prior design pass)

- **Sequence Anomaly (merged 1.B + 3.A + 3.B, Phase 3)**: one Red state-machine generating
  transaction trajectories against a synthetic baseline profile, three named presets
  (`credential_ato`, `low_and_slow`, `sequence_shift`), one rolling z-score/velocity Blue detector.
  Preset parameter grids, not a search/RL optimizer.
- **Multi-Agent (Phase 4, stretch)**: extends `AuthorizationGraph.trust_weight` — mark a node
  compromised, measure `PropagationRate`. Collusion sub-item only if compromise lands with time to
  spare.
- **Synthetic Identity (Phase 5, stretch)**: needs a `Persona` schema extension, added only when this
  phase is reached.
- **Composer (Phase 6, stretch)**: chains traces via `composed_of` — e.g. Branded Whisper → 1.D scope
  violation → sequence-anomaly cash-out — mirroring Paper 1's CHAIN (V5→V4→V9). Document in
  `08_attack_composer.md` even if the generator script itself doesn't get written.

## 4. Build Sequence — 7 Days, User's Phase Order

Must-have vs. stretch: **Phases 1–2 are a hard commitment**. **Phases 3–6 are attempted strictly in
order**; whichever phase the clock runs out on, everything built to that point stays a coherent,
demoable, end-to-end system.

- **Day 1 — Setup + substrate.** `.gitignore` before first commit (`.env`, `traces/
  attack_traces.jsonl`, `evaluation/results/*`). Verify `.env` loads; confirm the LLM endpoint's
  served model name with one throwaway call. Build `schemas.py`, `ap2_env.py` (four agents + mandate
  chain), `llm_client.py`, `trace_io.py`, `red_team/base.py`, `blue_team/base.py`. Start
  `docs/identify/*.md` in parallel, including the §3.5 roadmap table.

- **Day 2 — Phase 1a: reproduce Branded Whisper + Vault Whisper as literal baselines.** Fixed-task,
  ~10-trial reproduction of both paper attacks on the real `ap2_env.py` chain, reporting measured ASR
  / exposure rate next to the paper's 100% / 20% figures. Build `reasoning_attack_detector.py`
  (dual-objective: payment-manipulation verdict + data-exposure verdict). Build `evaluation/
  metrics.py` with Wilson CIs against this family first. Notebook 01.

- **Day 2-3 — Phase 1b: automate variant generation + detector evasion.** `mutate()` takes over from
  the fixed seed for both Branded and Vault Whisper (LLM-generated technique/framing variants), push
  trial volume to 100+/condition. Build `red_team/evasion.py` wrapping the generator against the real
  Blue detector — the flagship feedback-loop story. 1.A.3 (intent drift) next if time allows; 1.A.4/
  1.A.5 (context poisoning, cross-agent injection) only if Phase 1 is otherwise done early — first
  things cut. Notebook 02.

- **Day 4 — Phase 2: 1.D Delegation Abuse + 1.C Intent Manipulation.** 1.D first (deterministic,
  fastest, cites Paper 1's RC-taxonomy). 1.C second (reuses `IntentMatchScore` from 1.A.3). Extend
  `unified_pipeline.py`. Notebook 03.

- **Day 5 — Phase 3: Sequence Anomaly (merged 1.B + 3.A + 3.B).** As designed in §3.6. Extend
  `unified_pipeline.py` to cover this family. Notebook 04.

- **Day 6 — Integration + eval, then Phase 4 if time remains.** Generate the full trace set
  (~200-400+ traces, weighted toward 1.A given it's the flagship), run `unified_pipeline.py` for the
  master results table (including the `evaluation/adaptive_loop.py` unseen-attack generalization
  check), build `notebooks/99_full_pipeline_eval.ipynb`, finish `README.md` with the paper-vs-us
  novelty framing (§3.4) front and center. Phase 4 (Multi-Agent) only if hours remain; Phases 5-6
  documented but not expected to ship as code.

- **Day 7 — Buffer + submission.** No new features. Confirm no secrets committed, notebooks run clean
  top-to-bottom on a fresh clone, dependencies pinned. Package and submit.

**Cut-list (unchanged, still governs when time runs short):**
1. Sequence-anomaly "optimize fraud value vs. detection" → preset parameter grid, not search/RL.
2. Sequence-anomaly Blue → rule-based/explainable by default; ML only as stretch.
3. 1.A.4/1.A.5 (context poisoning, cross-agent injection) → cut before 1.A.1/1.A.2/1.A.3.
4. PaySim/any external dataset → untouched unless a Tier-2 slot is reached with time to spare.
5. Streamlit/CLI/server demo → skip; flagship notebook only.
6. `pyproject.toml`/packaging → skip; `requirements.txt` + `PYTHONPATH=.`.
7. Four separate eval harnesses → rejected; one `AttackTrace` schema + one `evaluation/metrics.py`.
8. Phases 4-6 → attempted in order only after Phases 1-3 integrate cleanly; each gets a docs-only
   writeup regardless of whether code lands.

`requirements.txt`:
```
openai
pydantic>=2
pandas
numpy
scikit-learn
matplotlib
jupyter
python-dotenv
```

## 5. Verification

- Day-1 gate: `python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(bool(os.getenv('LLM_API_KEY')), bool(os.getenv('LLM_ENDPOINT')))"`, then one uncommitted connectivity
  call to confirm the served model name.
- Day 2 gate specifically: the Branded Whisper / Vault Whisper baseline reproduction must complete
  and produce a results row directly comparable to the paper's Table 1 (ASR% / exposure%) before
  moving on to automated variant generation — this is the credibility anchor for the whole 1.A family.
- After each phase: run that family's notebook top-to-bottom on a clean kernel; confirm
  `unified_pipeline.py` produces a non-degenerate Precision/Recall/F1/AUC row for every family built
  so far.
- Before first commit and before final submission: `git status` must never show `.env`; grep the
  diff for the literal API key string as a second check.
- Final check: clone the repo to a fresh directory and confirm `notebooks/99_full_pipeline_eval.ipynb`
  runs end-to-end with only `requirements.txt` installed and a valid `.env` present.
