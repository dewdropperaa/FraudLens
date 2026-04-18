from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


ComplexityLabel = Literal["simple", "complex", "high_risk"]
ModelRoute = Literal["mistral", "llama3", "deepseek-r1"]


class ClaimRequestV2(BaseModel):
    identity: Dict[str, Any] = Field(default_factory=dict)
    documents: List[Dict[str, Any]] = Field(default_factory=list)
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
    agent: str
    score: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    explanation: str
    elapsed_ms: int = Field(ge=0)
    input_snapshot: Dict[str, Any] = Field(default_factory=dict)
    output_snapshot: Dict[str, Any] = Field(default_factory=dict)
    memory_insights: Optional[MemoryInsights] = None


class ClaimGuardV2Response(BaseModel):
    agent_outputs: List[AgentOutput]
    blackboard: Dict[str, Any]
    routing_decision: RoutingDecision
    goa_used: bool
    Ts: float = Field(ge=0, le=100)
    decision: str
    retry_count: int = Field(ge=0, le=3)
    mahic_breakdown: Dict[str, float] = Field(default_factory=dict)
    contradictions: List[Dict[str, Any]] = Field(default_factory=list)
    trust_layer: Dict[str, Any] | None = None
    memory_context: List[Dict[str, Any]] = Field(default_factory=list)
