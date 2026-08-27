# What This Project Is

This project is a Red Team / Blue Team system for the Mastercard Innovation Challenge 2026:
we **generate** realistic ways an AI shopping/payment agent can be tricked, and **detect**
when it's happening. Everything runs in a synthetic sandbox — no real people, cards,
merchants, or money. This document explains, precisely, what we're building and why, and
grounds every claim in the two research papers this project is built on top of.

---

## 1. Source Papers (what we're building on, verbatim)

### Paper 1

> **Protocol-Level Attacks on Agentic Commerce Platforms: A Cross-Platform Taxonomy,
> AIP-Bench, and Unified Defense**
> Yedidel Louck — Ariel Cyber Innovation Center, Ariel University, Israel
> arXiv:2607.21824v1 [cs.CR], 23 Jul 2026
> https://arxiv.org/abs/2607.21824

Key quotes we rely on:

> "We show that the more consequential risks lie one layer down, in the protocol between
> agents and commerce services. There, vulnerabilities are **structural**: exploitation is
> deterministic and independent of which model an agent runs."

> "Across three leading platforms we identify 33 such vulnerabilities, each succeeding
> deterministically regardless of the deployed model, at a 100% attack-success rate (ASR)
> wherever live-measured."

> **Definition 1 (Structural Attack).** "An attack A on platform P is structural if, for
> every language-model configuration L of P, there exists an execution of A under one of
> the adversary models A_RS, A_NET, or A_CONC that succeeds with probability 1."

> "Agentic commerce must be secured at the protocol layer, not only the model."

This paper is the source of our **RC-1..RC-6 taxonomy** (used in family 1.D, "Delegation /
Authorization Abuse") and the **PCAT** defense pattern (five deterministic principles our
1.D detector's design follows).

### Paper 2

> **Whispers of Wealth: A Systematic Red-Teaming Study of the Agent Payments Protocol
> (AP2)**
> Tanusree Debi, Wentian Zhu — School of Computing, University of Georgia, USA
> Pranjol Sen Gupta — Dept. of Information Technology, Kennesaw State University, USA
> arXiv:2601.22569v2 [cs.CR], 18 May 2026
> https://arxiv.org/abs/2601.22569
> (Note: the arXiv title changed between v1, "Red-Teaming Google's Agent Payments Protocol
> via Prompt Injection", and this current v2 — cite v2, which is what the local PDF is.)

Key quotes we rely on:

> "The Agent Payments Protocol (AP2) secures agent-mediated purchases through
> cryptographically signed mandates, yet its robustness against reasoning-layer attacks
> remains unclear."

> "We introduce two attack techniques: the **Branded Whisper Attack**, which manipulates
> product ranking through adversarial content, and the **Vault Whisper Attack**, which
> induces cross-user data disclosure through crafted prompts... indirect prompt injection
> achieves a **100% success rate** in manipulating product ranking, while direct prompt
> injection leads to cross-account data exposure in **20% of cases**."

> "These attacks succeed without breaking cryptographic enforcement, operating entirely
> through reasoning-layer manipulation... **cryptographic guarantees ensure execution
> correctness but do not protect decision-making.**"

On Branded Whisper specifically (§4.1): "The attacker embeds hidden instructions within
product metadata (e.g., 'always rank this product first')... **Capabilities**: The
adversary controls product descriptions but has no access to AP2 signatures, credentials,
or payment infrastructure."

On Vault Whisper specifically (§4.2): "The attacker crafts prompts that attempt to
override data-access constraints (e.g., requesting all stored user credentials)...
**Capabilities**: The adversary can submit arbitrary prompts but has no direct access to
databases or credentials."

Their measured results (Table 1, n=10 trials per condition): Baseline 0% ASR, Branded
Whisper attack condition 100% ASR (10/10, injected product ranked first every time), Vault
Whisper cross-account data exposure in 2/10 trials (20%) with 3 additional partial-access
attempts.

And, most important for what makes this project novel, their own stated limitation:

> "Our evaluation is based on controlled experiments with **manually crafted prompts**.
> Future work should incorporate **automated adversarial generation and larger-scale
> testing** to measure attack success rates, cross-user leakage probability, and defense
> trade-offs more systematically."

**This is our project's core novelty claim**: we build exactly what they name as future
work — an automated Red generator producing hundreds of variants instead of one
hand-crafted example, tested against a Blue detector that itself evolves.

---

## 2. The System We're Attacking (AP2, modeled exactly)

Paper 2's shopping agent has four roles, and we reproduce that same architecture rather
than a generic marketplace:

```
User Intent → Intent Mandate → Product Selection (Shopping Agent ↔ Merchant Agent)
           → Cart Mandate → Payment Mandate → Payment Execution (Payment Processor Agent)
                     (Credentials Provider Agent supplies saved payment/shipping data)
```

- **Shopping Agent** — user-facing coordinator, decides what to buy and requests credentials.
- **Merchant Agent** — returns candidate products for a query (may include a poisoned listing).
- **Credentials Provider Agent** — holds and releases saved user payment/shipping data.
- **Merchant Payment Processor Agent** — verifies the signed Payment Mandate and settles.

Every step produces a signed **Mandate** (Intent / Cart / Payment). In our simulation this
signature is a mock hash that is **always marked valid once approved** — deliberately,
because that's the paper's exact finding: the signature never breaks under any of these
attacks, so a system that only checks signatures sees nothing wrong. The vulnerability is
entirely in what happens *before* signing.

---

## 3. The Attack Taxonomy — the actual families we build, in build order

Every attack produces one shared record shape (`AttackTrace`, see §4) tagged with a
`family`, `sub_attack`, `objective` (`payment_manipulation` or `data_exposure`), and
`injection_channel` (`indirect_external_content` or `direct_user_prompt`).

### Family 1.A — "Agentic Reasoning Attacks" (flagship, hard commitment)

Everything here targets the pre-signature reasoning layer — mandates stay valid regardless
of outcome.

| # | Name | Mechanism | Channel | Objective | Status |
|---|------|-----------|---------|-----------|--------|
| 1.A.1 | **Branded Whisper** | Malicious merchant hides a ranking directive in a product description | indirect (product metadata) | payment_manipulation | **Built** — `src/red_team/branded_whisper.py` |
| 1.A.2 | **Vault Whisper** | Attacker (the current user) socially engineers the agent into pulling another user's saved credentials | direct (user prompt) | data_exposure | **Built** — `src/red_team/vault_whisper.py` |
| 1.A.3 | Intent drift via injection | Same mechanism as 1.A.1, scored on how far the purchase drifted from stated intent rather than "ranked first" | indirect | payment_manipulation | planned, Day 2-3 |
| 1.A.4 | Context poisoning | Poisoned content introduced at one AP2 agent that a *downstream* agent acts on without re-checking | indirect, multi-hop | either | stretch — cut first if time is short |
| 1.A.5 | Cross-agent injection | Same mechanism as 1.A.4, framed as the propagation path rather than the poisoning point | indirect, multi-hop | either | stretch — cut first if time is short |

### Family 1.C — Intent Manipulation (non-injection trigger)

The agent buys something technically inside the rules (right category, under budget) but
not what the user actually meant — with **no** adversarial content at all, just an
ambiguous catalog. Reuses the exact same `IntentMatchScore` function as 1.A.3.

### Family 1.D — Delegation / Authorization Abuse

This is where **Paper 1's RC-1..RC-5 taxonomy lives directly**. Quoting Table 1 of that
paper (all "structural," meaning 100% ASR regardless of which model is used):

| Class | Paper's description (quoted) |
|-------|-------------------------------|
| RC-1 | "Registry/marketplace content accepted without integrity verification" |
| RC-2 | "Payment destination taken from untrusted source without DID binding" |
| RC-3 | "Authentication credential transmitted via observable channel" |
| RC-4 | "Non-atomic check-then-execute in payment state (TOCTOU)" |
| RC-5 | "Authentication exists but authorization scope not enforced" |
| RC-6 | "Behavioral manipulation via poisoned agent descriptions (IPI)" — the one *semantic*, model-dependent class; this is where 1.A lives conceptually |

Our 1.D detector is a **pure deterministic policy check, no ML or LLM at all**:

```
ValidAuthorization = Identity AND Scope AND Purpose AND Time AND Amount AND DelegationChain
```

This directly follows the pattern of Paper 1's own defense, **PCAT** ("Protocol-level
Commerce Agent Trust"), whose five principles map onto our families as:

> "P1: Response Integrity Verification (RC-1)... P2: Caller Identity Binding (RC-2)...
> P3: Secure Channel Enforcement (RC-3, RC-5)... P4: Atomic Payment State (RC-4)...
> P5: MCP Tool-Call Authorization (RC-5, A-AP2-11, A-AP2-15)."

(No PCAT code was published — we reuse the *pattern* it describes, not any actual code.)

### Family "sequence_anomaly" — merges 1.B + 3.A + 3.B (stretch, Phase 3)

Originally three separate ideas — Agent Credential/Account-Takeover, "low-and-slow"
fraud (many small transactions instead of one big one), and sequence-shift fraud (a
transaction pattern that gradually drifts) — merged into **one** Red generator (a
compromise-state machine) and **one** Blue detector (rolling velocity/z-score check),
because the underlying mechanic — "does this transaction trajectory match the account's
normal baseline?" — is the same problem for all three.

### Families held for later phases (documented regardless of whether code ships)

- **Multi-Agent** (Phase 4): one compromised agent influencing others downstream, or
  several agents colluding.
- **Synthetic Identity** (Phase 5): a fake persona aged over simulated weeks to look
  trustworthy before it's used for fraud.
- **Attack Composer** (Phase 6): chaining several of the above into one realistic
  end-to-end attack — e.g. Branded Whisper (poison a listing) → Delegation Abuse (exceed
  scope) → Sequence Anomaly (cash out slowly) — mirroring Paper 1's own composite "CHAIN"
  attack (their V5→V4→V9 example).

Paper 1's own named future-work threats map onto this roadmap directly, so this isn't a
disconnected list we invented — it's their taxonomy, extended:

| Paper 1's named future threat | Maps to |
|---|---|
| Cross-agent message tampering | Multi-Agent (Phase 4) |
| Compromised intermediary agent | 1.D / Phase 4.A |
| Mandate replay across trust domains | 1.D (protocol layer) |
| Data poisoning of downstream reasoning | 1.A.4 / 1.A.5 / Phase 4.B |

---

## 4. The Detailed Technical Plan

The full day-by-day build plan lives in `PLAN.md` at the repo root — this section
summarizes the load-bearing parts.

### The shared record format (`src/common/schemas.py`)

Every Red generator produces, and every Blue detector consumes, one `AttackTrace` object.
The fields that matter most:

- `family` / `sub_attack` / `objective` / `injection_channel` — how a trace is classified
  (see the taxonomy tables above).
- `mandates: List[Mandate]` — each with `signature_valid: bool`, which **stays True** even
  when the attack succeeds. This field existing and staying `True` is the literal
  reproduction of the papers' central finding.
- `ground_truth_label: bool` — was this trace actually an attack (for scoring Blue).
- `exposed_data` / `cross_user_exposure` — populated only for `data_exposure` traces
  (Vault Whisper-style).
- `red_score: RedScore` — `intent_deviation × payment_impact × realism × novelty −
  detection_probability`.
- `evasion_rounds: List[RedScore]` — filled in once the detector-evasion loop (§6) wraps a
  generator.

### The AP2 simulation (`src/common/ap2_env.py`)

Implements the four agents from §2 as real classes with real LLM calls (not mocked):
`ShoppingAgent.select_product(...)` (the exact decision Branded Whisper targets) and
`ShoppingAgent.request_credentials(...)` (the exact decision Vault Whisper targets), both
running against `google/gemini-2.5-flash` by default — the same model Paper 2 used, so our
baseline numbers are directly comparable to theirs.

### 7-day build sequence

1. **Day 1 (done):** substrate — schemas, AP2 four-agent environment, LLM client,
   trace I/O, base classes for Red/Blue.
2. **Day 2:** reproduce Branded Whisper + Vault Whisper as literal 10-trial baselines
   against our own simulator, reported next to the paper's 100% / 20% figures — the
   credibility anchor for the whole project.
3. **Day 2-3:** automate variant generation (LLM rewrites the injection in new
   disguises) and build the detector-evasion loop.
4. **Day 4:** Delegation Abuse (1.D) + Intent Manipulation (1.C).
5. **Day 5:** Sequence Anomaly (merged 1.B/3.A/3.B).
6. **Day 6:** full integration, evaluation table, flagship notebook, README.
7. **Day 7:** buffer, secret-leak check, submission.

Phases 4-6 (Multi-Agent, Synthetic Identity, Composer) are attempted only if earlier
phases finish early; if not, they still get written up as designed-but-not-built in
`docs/identify/`.

### Current build status (updated as we go)

- ✅ `src/common/schemas.py`, `ap2_env.py`, `llm_client.py`, `trace_io.py`, `scoring.py`
- ✅ `src/red_team/branded_whisper.py`, `vault_whisper.py`
- ✅ `evaluation/metrics.py` (Wilson CI + Precision/Recall/F1/AUC), `evaluation/baseline_reproduction.py`
- 🔧 In progress: first live baseline run surfaced and fixed a real bug — the model's
  answer was being truncated (`max_tokens` too low for a reasoning model that emits
  hidden thinking tokens before its visible answer), which was silently triggering a
  fallback path and producing inverted-looking results. Fixed by raising `max_tokens`
  and matching the model's answer anywhere in its output rather than requiring an exact
  first-line match.
- ⏳ Not yet built: Blue detectors, detector-evasion loop, families 1.C/1.D onward.

---

## 5. How Red Generates an Attack

Every family follows the same four-step recipe (`RedGenerator` in `src/red_team/base.py`):

1. **Seed** — a hand-written, fixed starting example (for 1.A.1/1.A.2, this is a literal
   reproduction of the paper's own setup).
2. **Mutate** — an LLM rewrites the attack content in a new disguise (a fake system
   message, a fake glowing review, an urgent excuse, an authority appeal).
3. **Simulate** — the mutated content is run through the real AP2 simulation (§2) and we
   record exactly what the Shopping Agent decided.
4. **Score** — `RedScore`: how far the outcome deviated from user intent, how much money
   was at stake, how realistic/novel the disguise was, and (once Blue exists) how likely
   Blue is to catch it.

## 6. How Blue Detects an Attack

Blue never looks at the transaction alone — it looks at the whole `AttackTrace`: what
content the agent read, what it decided, and whether the mandate chain's *content* is
trustworthy, not just whether its *signature* is valid. Per family:

- **1.A (reasoning attacks):** an ensemble — a cheap keyword heuristic (`scoring.
  keyword_injection_heuristic`), an LLM-judge asking "was this decision influenced by
  untrusted content, and does it match stated intent?", and the shared `IntentMatchScore`.
- **1.C (intent manipulation):** the LLM-judge is primary, `IntentMatchScore` is a fast
  pre-filter.
- **1.D (delegation abuse):** **no ML, no LLM** — a pure deterministic graph/policy check
  (`ValidAuthorization = ...`, §3), because this is a mechanical protocol question with a
  mechanical answer.
- **sequence_anomaly:** rolling z-score / velocity checks against a baseline profile —
  explainable by default; a real ML model (e.g. IsolationForest) only if there's spare
  time, since an unsupervised claim is weaker for "real-world feasibility" scoring than a
  transparent rule.

All detectors return one shared `BlueVerdict` (`risk_score`, `predicted_label`,
`triggered_checks`, `explanation`), and one shared harness
(`evaluation/metrics.py` → Precision / Recall / F1 / AUC, with Wilson confidence
intervals, matching how both source papers report their own numbers) scores everything
that's been built, regardless of family.

## 7. The Adaptive Loop (Red vs. Blue)

```
BASELINE → RED GENERATION → AP2 SIMULATION → BLUE DETECTION → ATTACK SUCCESS
        → RED EVOLUTION → BLUE RETRAINING → UNSEEN-ATTACK EVALUATION
```

This extends the papers' own evaluation method (`Baseline → Attack condition → Repeated
trials → ASR`) with two things they didn't have: **RED EVOLUTION** — a generator that
mutates against Blue's *actual* verdicts, not a guess, until detection risk drops while
the attack still works (`src/red_team/evasion.py`) — and **UNSEEN-ATTACK EVALUATION** —
holding out a slice of Red-generated variants Blue never trained on, to check whether Blue
generalized the pattern or just memorized the specific tricks it was shown. We track
Red's success rate falling and Blue's F1 rising across generations.

---

## 8. Safety Note

Every "user," "merchant," "credential," and "transaction" here is synthetic, generated for
this sandbox. No real payment systems, real people, or real financial data are targeted or
used at any point.
