from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


DecisionLabel = Literal["APPROVED", "REJECTED", "PENDING"]


class AgentDecisionOutput(BaseModel):
    """Structured per-agent output (audit-friendly)."""

    agent: str = Field(..., description="Human-readable agent name")
    decision: DecisionLabel
    score: float = Field(..., ge=0.0, le=100.0)
    reason: str = Field(..., description="Primary rationale shown to operators")
    explainability: str = Field(
        default="",
        description="Additional audit trail: signals, thresholds, and rule paths",
    )
    details: dict[str, Any] = Field(default_factory=dict)


class FinalConsensusPayload(BaseModel):
    """Post-crew aggregation (for APIs and audit logs)."""

    final_decision: DecisionLabel
    avg_score: float = Field(..., description="Unweighted arithmetic mean of agent scores")
    weighted_score: float = Field(
        ...,
        description="Weighted score used by ConsensusSystem (same as ClaimResult.score driver)",
    )
    consensus_threshold: float
    agents: list[AgentDecisionOutput]
    veto_applied: bool = Field(
        default=False,
        description="True when extreme-fraud veto forced REJECTED after weighted approval",
    )
