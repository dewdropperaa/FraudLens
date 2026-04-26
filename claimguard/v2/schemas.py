from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


ComplexityLabel = Literal["simple", "complex", "high_risk"]
ModelRoute = Literal["mistral", "llama3", "deepseek-r1"]
ValidationStatus = Literal["VALID", "INVALID"]
DecisionEnum = Literal["APPROVED", "HUMAN_REVIEW", "REJECTED"]
DECISION_ENUM_VALUES: tuple[DecisionEnum, ...] = ("APPROVED", "HUMAN_REVIEW", "REJECTED")

# DocumentType is deliberately a free-form string. The coverage-score model
# (claimguard.v2.coverage_score) now drives pipeline decisions; the label is
# informational clustering only, so it must never hard-gate the pipeline.
DocumentType = str
CANONICAL_DOCUMENT_TYPES: tuple[str, ...] = (
    "medical_invoice",
    "medical_prescription",
    "lab_report",
    "medical_certificate",
    "insurance_attestation",
    "pharmacy_invoice",
    "hospital_bill",
    "medical_claim_bundle",
    "hybrid_bundle",
    "unknown_bundle",
    "irrelevant_document",
    "unknown",
)


class ClaimRequestV2(BaseModel):
    class IdentityPayload(BaseModel):
        cin: str | None = None
        ipp: str | None = None

    identity: IdentityPayload = Field(default_factory=IdentityPayload)
    documents: List[Dict[str, Any]] = Field(default_factory=list)
    documents_base64: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Optional inline files with keys {name, content_base64}; server extracts text/OCR.",
    )
    policy: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RoutingDecision(BaseModel):
    intent: str
    complexity: ComplexityLabel
    model: ModelRoute
    reason: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BlackboardEntry(BaseModel):
    score: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    explanation: str


class MemoryInsights(BaseModel):
    """Per-agent analysis of retrieved memory context."""

    similar_cases_found: int = Field(ge=0, default=0)
    fraud_matches: int = Field(ge=0, default=0)
    identity_reuse_detected: bool = False
    impact_on_score: str = ""
    notes: str = ""


class AgentOutput(BaseModel):
    class ClaimEvidence(BaseModel):
        statement: str
        evidence: str
        verified: bool

    class HallucinationDebug(BaseModel):
        claims_checked: int = Field(ge=0, default=0)
        verified_claims: int = Field(ge=0, default=0)
        hallucination_flags: List[str] = Field(default_factory=list)
        confidence_adjusted: float = Field(ge=0, le=1, default=0.0)

    agent: str
    score: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    claims: List[ClaimEvidence] = Field(default_factory=list)
    hallucination_flags: List[str] = Field(default_factory=list)
    explanation: str
    hallucination_penalty: float = Field(ge=0, le=1, default=0.0)
    debug_log: HallucinationDebug | None = None
    elapsed_ms: int = Field(ge=0)
    input_snapshot: Dict[str, Any] = Field(default_factory=dict)
    output_snapshot: Dict[str, Any] = Field(default_factory=dict)
    memory_insights: Optional[MemoryInsights] = None


class ValidationResult(BaseModel):
    """Result of claim validation performed by ClaimValidationAgent.

    ``document_type`` is a free-form string used as informational clustering;
    it is NEVER used as a gate. ``should_stop_pipeline`` is retained for
    backwards compatibility but is treated as an advisory soft-fail signal
    rather than a hard stop.
    """
    validation_status: ValidationStatus
    validation_score: int = Field(ge=0, le=100)
    document_type: str = "unknown"
    missing_fields: List[str] = Field(default_factory=list)
    found_fields: List[str] = Field(default_factory=list)
    reason: str
    should_stop_pipeline: bool = False
    details: Dict[str, Any] = Field(default_factory=dict)


class DecisionExplanationModel(BaseModel):
    """Structured, mandatory explanation attached to every pipeline exit."""

    summary: str = ""
    reasons: List[str] = Field(default_factory=list)
    signals: Dict[str, Any] = Field(default_factory=dict)
    tool_outputs: Dict[str, Any] = Field(default_factory=dict)


class PreValidationResult(BaseModel):
    score: int = Field(ge=0, le=100, default=0)
    confidence: int = Field(ge=0, le=100, default=100)
    status: str = Field(default="REJECTED")
    reason: str
    flags: List[str] = Field(default_factory=list)
    document_type: str = Field(default="UNKNOWN")
    injection_detected: bool = False
    passed: bool = False


class ClaimGuardV2Response(BaseModel):
    agent_outputs: List[AgentOutput]
    blackboard: Dict[str, Any]
    routing_decision: RoutingDecision
    goa_used: bool
    Ts: float = Field(ge=0, le=100)
    decision: DecisionEnum
    exit_reason: str
    retry_count: int = Field(ge=0, le=3)
    mahic_breakdown: Dict[str, float] = Field(default_factory=dict)
    contradictions: List[Dict[str, Any]] = Field(default_factory=list)
    trust_layer: Dict[str, Any] | None = None
    memory_context: List[Dict[str, Any]] = Field(default_factory=list)
    validation_result: Optional[ValidationResult] = None
    pre_validation_result: Optional[PreValidationResult] = None
    forensic_trace: Dict[str, Any] | None = None
    trace: Dict[str, Any] | None = None
    decision_trace: Dict[str, Any] | None = None
    explanation: Dict[str, Any] | None = None
    response_envelope: Dict[str, Any] | None = None
    system_flags: List[str] = Field(default_factory=list)
    claim_id: str | None = None
    score: float = Field(default=0.0)
    stage: str = Field(default="FINAL_DECISION")
    # PROD-FIX: structured flags by severity tier.
    flags: Dict[str, List[str]] = Field(
        default_factory=lambda: {"blocking": [], "warnings": [], "informational": []}
    )
    reason: str | None = None
    document_url: str | None = None
    extracted_data: Dict[str, Any] = Field(default_factory=dict)
    heatmap: List[Dict[str, Any]] = Field(default_factory=list)
    heatmap_fallback: List[Dict[str, Any]] = Field(default_factory=list)
    pipeline_version: str = "v2"
    pipeline_trace: List[str] = Field(
        default_factory=lambda: ["PRE_VALIDATION", "FIELD_VERIFICATION", "AGENTS", "CONSENSUS"]
    )
    explanation: DecisionExplanationModel = Field(default_factory=DecisionExplanationModel)

    @field_validator("explanation", mode="before")
    @classmethod
    def _coerce_explanation(cls, v: Any) -> Any:
        if v is None:
            return DecisionExplanationModel()
        if isinstance(v, dict):
            return DecisionExplanationModel(**{k: val for k, val in v.items() if k in DecisionExplanationModel.model_fields})
        return v
    coverage_score: Dict[str, Any] = Field(default_factory=dict)
    agent_results: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: str = "LOW"
    stage_reached: str = "FINAL_DECISION"
    agents: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    field_verification: Dict[str, Any] = Field(default_factory=dict)
    memory_status: str = "DISABLED"
    audit_trail: List[Dict[str, Any]] = Field(default_factory=list)
    processing_time_ms: int = 0
    routed_to: str | None = None
    blockchain_tx: str | None = None
    ipfs_document: str | None = None
    tx_hash: str | None = None
    ipfs_hash: str | None = None
