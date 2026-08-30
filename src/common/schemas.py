"""Shared data contracts for the agentic-commerce fraud red-team system.

One AttackTrace shape is produced by every Red generator and consumed by every
Blue detector and by evaluation/metrics.py. Family-specific fields that don't
fit the base schema go in `metadata`, not as new top-level fields, so the
unified pipeline (src/blue_team/unified_pipeline.py) never has to special-case
a family's shape.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# AP2 mandate chain
# ---------------------------------------------------------------------------

class Mandate(BaseModel):
    """A signed AP2 mandate. `signature_valid` stays True under successful
    attacks in families 1.A/1.C/1.D — that persistence is the entire point of
    Whispers of Wealth's finding: cryptographic execution integrity does not
    imply decision integrity."""

    mandate_type: Literal["intent", "cart", "payment"]
    content_hash: str
    signature_valid: bool
    approved_by_user: bool


# ---------------------------------------------------------------------------
# Intent / catalog
# ---------------------------------------------------------------------------

class IntentObject(BaseModel):
    category: str
    brand: Optional[str] = None
    max_amount: float
    quantity: int = 1
    geography: Optional[str] = None
    urgency: Optional[str] = None
    raw_user_statement: str


class Product(BaseModel):
    product_id: str
    title: str
    description: str
    price: float
    brand: Optional[str] = None
    category: str
    merchant_id: str
    is_malicious: bool = False


# ---------------------------------------------------------------------------
# Injected / adversarial content
# ---------------------------------------------------------------------------

class ExternalContentItem(BaseModel):
    """A piece of content an agent reads during reasoning. `hop_index` marks
    which AP2 agent surfaced it: 0=MerchantAgent, 1=ShoppingAgent,
    2=CredentialsProviderAgent, 3=MerchantPaymentProcessorAgent."""

    source_url: str
    text: str
    contains_injection: bool = False
    injection_technique: Optional[str] = None
    hop_index: int = 0


# ---------------------------------------------------------------------------
# Delegation / authorization graph (family 1.D, also used by Multi-Agent)
# ---------------------------------------------------------------------------

class DelegationEdge(BaseModel):
    edge_id: str
    from_agent: str
    to_agent: str
    allowed_categories: List[str]
    max_amount: float
    merchant_category_codes: List[str] = Field(default_factory=list)
    valid_from: datetime
    valid_until: datetime
    purpose: str
    trust_weight: float = 1.0  # < 1.0 marks a compromised/untrusted edge (Multi-Agent, Phase 4)


class AuthorizationGraph(BaseModel):
    nodes: List[str]
    edges: List[DelegationEdge]


# ---------------------------------------------------------------------------
# Transactions
# ---------------------------------------------------------------------------

class Transaction(BaseModel):
    txn_id: str
    agent_id: str
    merchant_id: str
    amount: float
    category: str
    timestamp: datetime
    executing_authorization_edge: Optional[str] = None


# ---------------------------------------------------------------------------
# Credential-compromise state machine (Sequence Anomaly family, Phase 3)
# ---------------------------------------------------------------------------

class CredentialState(str, Enum):
    LEGITIMATE = "legitimate"
    COMPROMISED_UNKNOWN = "compromised_unknown"
    COMPROMISED_MIMIC = "compromised_mimic"
    COMPROMISED_LOW_VALUE = "compromised_low_value"
    COMPROMISED_ESCALATING = "compromised_escalating"
    COMPROMISED_LEGIT_MERCHANT = "compromised_legit_merchant"


# ---------------------------------------------------------------------------
# Red scoring
# ---------------------------------------------------------------------------

class RedScore(BaseModel):
    intent_deviation: float = 0.0
    payment_impact: float = 0.0
    realism: float = 0.0
    novelty: float = 0.0
    detection_probability: float = 0.0
    r_red: float = 0.0

    @staticmethod
    def compute(
        intent_deviation: float,
        payment_impact: float,
        realism: float,
        novelty: float,
        detection_probability: float,
    ) -> "RedScore":
        r_red = intent_deviation * payment_impact * realism * novelty - detection_probability
        return RedScore(
            intent_deviation=intent_deviation,
            payment_impact=payment_impact,
            realism=realism,
            novelty=novelty,
            detection_probability=detection_probability,
            r_red=r_red,
        )


# ---------------------------------------------------------------------------
# BlueVerdict (produced by every Blue detector)
# ---------------------------------------------------------------------------

class BlueVerdict(BaseModel):
    trace_id: str
    risk_score: float  # 0.0-1.0
    predicted_label: bool  # True = flagged as attack
    triggered_checks: List[str] = Field(default_factory=list)
    explanation: str = ""


# ---------------------------------------------------------------------------
# The single trace format — every family produces this shape
# ---------------------------------------------------------------------------

FamilyLiteral = Literal[
    "reasoning_attack",
    "intent_manipulation",
    "delegation_abuse",
    "sequence_anomaly",
    "multi_agent",
    "synthetic_identity",
    "composite",
]

SubAttackLiteral = Optional[
    Literal[
        "branded_whisper",
        "vault_whisper",
        "intent_drift",
        "context_poisoning",
        "cross_agent_injection",
        "delegation_scope_violation",
        "ambiguous_catalog",
    ]
]


class AttackTrace(BaseModel):
    trace_id: str
    family: FamilyLiteral
    sub_attack: SubAttackLiteral = None
    objective: Literal["payment_manipulation", "data_exposure"] = "payment_manipulation"
    injection_channel: Optional[
        Literal["indirect_external_content", "direct_user_prompt"]
    ] = None

    ground_truth_label: bool  # True = this trace IS an attack

    user_intent: IntentObject
    external_content: List[ExternalContentItem] = Field(default_factory=list)
    mandates: List[Mandate] = Field(default_factory=list)
    authorization_graph: Optional[AuthorizationGraph] = None
    agent_reasoning_trace: List[Dict[str, Any]] = Field(default_factory=list)

    transactions: List[Transaction] = Field(default_factory=list)
    final_transaction: Optional[Transaction] = None

    exposed_data: Optional[Dict[str, Any]] = None
    cross_user_exposure: bool = False

    red_score: RedScore = Field(default_factory=RedScore)
    evasion_rounds: List[RedScore] = Field(default_factory=list)
    composed_of: List[str] = Field(default_factory=list)

    generation: int = 0  # adaptive co-evolution round this trace was produced in (evaluation/adaptive_loop.py)

    metadata: Dict[str, Any] = Field(default_factory=dict)

    generated_at: datetime = Field(default_factory=datetime.utcnow)
