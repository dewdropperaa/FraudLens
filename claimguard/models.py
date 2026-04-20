from __future__ import annotations

import base64
import re
from datetime import datetime
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


InsuranceType = Literal["CNSS", "CNOPS"]
DecisionType = Literal["APPROVED", "REJECTED"]

_MAX_B64_DECODED_BYTES = 5 * 1024 * 1024


class DocumentBase64Part(BaseModel):
    """Single file sent inline as base64 (JSON ``/claim`` body)."""

    name: str = Field(..., min_length=1, max_length=512, description="Original filename (used for type detection)")
    content_base64: str = Field(..., min_length=4, description="Standard base64 of raw file bytes")

    @field_validator("content_base64", mode="before")
    @classmethod
    def strip_b64_payload(cls, v: Any) -> str:
        if v is None:
            raise ValueError("content_base64 is required")
        s = str(v).strip()
        if "base64," in s:
            s = s.split("base64,", 1)[-1]
        return re.sub(r"\s+", "", s)

    @field_validator("content_base64")
    @classmethod
    def validate_decoded_size(cls, v: str) -> str:
        try:
            raw = base64.b64decode(v, validate=True)
        except Exception as e:
            raise ValueError(f"invalid base64: {e}") from e
        if len(raw) > _MAX_B64_DECODED_BYTES:
            raise ValueError(
                f"decoded document exceeds {_MAX_B64_DECODED_BYTES} bytes ({len(raw)} bytes)"
            )
        if len(raw) == 0:
            raise ValueError("decoded document is empty")
        return v


class ClaimInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    patient_id: str = Field(..., min_length=1, max_length=128, description="Patient identifier")
    provider_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Healthcare provider / facility identifier (graph edges: patient–provider–claim)",
    )
    claim_id: str | None = Field(
        default=None,
        max_length=128,
        description="Optional client-supplied claim reference; server generates one if omitted",
    )
    amount: float = Field(..., gt=0, le=50_000_000, description="Claim amount")
    documents: List[str] = Field(
        default_factory=list,
        max_length=50,
        description="List of document IDs or filenames",
    )
    documents_base64: List[DocumentBase64Part] = Field(
        default_factory=list,
        max_length=20,
        description="Optional inline files (base64); server extracts text and merges into analysis",
    )
    # Each history item is free-form; our agents only look for `amount`, `date`, and `recent`.
    history: List[Dict[str, Any]] = Field(
        default_factory=list,
        max_length=200,
        description="Patient claim history",
    )
    insurance: InsuranceType = Field(..., description="Insurance provider")

    @field_validator("patient_id", "provider_id", mode="before")
    @classmethod
    def strip_required_ids(cls, v: Any) -> str:
        if v is None:
            raise ValueError("must not be empty")
        s = str(v).strip()
        if not s:
            raise ValueError("must not be empty")
        return s

    @field_validator("claim_id", mode="before")
    @classmethod
    def empty_claim_id_to_none(cls, v: Any) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    @field_validator("documents", mode="before")
    @classmethod
    def validate_document_entries(cls, v: Any) -> List[str]:
        if v is None:
            return []
        if not isinstance(v, list):
            raise ValueError("documents must be a list")
        out: List[str] = []
        for item in v:
            s = str(item).strip()
            if len(s) > 512:
                raise ValueError("each document id must be at most 512 characters")
            if s:
                out.append(s)
        return out

    @model_validator(mode="after")
    def limit_total_document_sources(self) -> ClaimInput:
        n = len(self.documents) + len(self.documents_base64)
        if n > 50:
            raise ValueError("documents + documents_base64 combined must be at most 50 entries")
        return self


class AgentResult(BaseModel):
    agent_name: str
    decision: bool
    # 0-100 (agent-specific). Consensus combines these with configurable weights.
    score: float
    reasoning: str
    details: Dict[str, Any] = Field(default_factory=dict)


class ClaimResult(BaseModel):
    claim_id: str
    decision: DecisionType
    score: float
    agent_results: List[AgentResult]
    consensus_decision: str | None = None
    Ts: float | None = None
    retry_count: int = 0
    mahic_breakdown: Dict[str, float] = Field(default_factory=dict)
    contradictions: List[Dict[str, Any]] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    tx_hash: str | None = None
    ipfs_hash: str | None = None
    ipfs_hashes: List[str] = Field(default_factory=list)
    claim_hash: str | None = None
    zk_proof_hash: str | None = None


class ClaimSubmitResponse(BaseModel):
    claim_id: str
    decision: DecisionType
    score: float
    agent_results: List[AgentResult]
    timestamp: datetime


class ClaimListResponse(BaseModel):
    items: List[ClaimResult]
    total: int
    page: int
    page_size: int
