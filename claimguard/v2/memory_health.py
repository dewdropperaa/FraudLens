from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class MemoryHealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class MemoryConfig:
    min_similarity: float = 0.7
    probe_claim_id: str = "memory-health-probe"
    degraded_memory_auto_approve_threshold: float = 95.0
    probe_timeout_ms: int = 300


@dataclass(frozen=True)
class MemoryHealthReport:
    status: MemoryHealthStatus
    latency_ms: int
    probe_result_count: int
    failure_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "latency_ms": int(self.latency_ms),
            "probe_result_count": int(self.probe_result_count),
            "failure_reason": self.failure_reason,
        }


def get_memory_health(config: MemoryConfig, memory_layer: Any) -> MemoryHealthReport:
    started = time.perf_counter()
    using_fake = bool(getattr(memory_layer, "_using_fake_embeddings", False))
    if using_fake:
        return MemoryHealthReport(
            status=MemoryHealthStatus.UNAVAILABLE,
            latency_ms=int((time.perf_counter() - started) * 1000),
            probe_result_count=0,
            failure_reason="embedding_model_unavailable_or_fake",
        )

    # Check 1: embedding model responsiveness.
    embedder = getattr(memory_layer, "_embedder", None)
    if embedder is None or not hasattr(embedder, "embed_query"):
        return MemoryHealthReport(
            status=MemoryHealthStatus.UNAVAILABLE,
            latency_ms=int((time.perf_counter() - started) * 1000),
            probe_result_count=0,
            failure_reason="embedding_model_missing",
        )
    try:
        probe_vec = embedder.embed_query(f"probe:{config.probe_claim_id}")
    except Exception as exc:
        return MemoryHealthReport(
            status=MemoryHealthStatus.UNAVAILABLE,
            latency_ms=int((time.perf_counter() - started) * 1000),
            probe_result_count=0,
            failure_reason=f"embedding_probe_failed:{exc}",
        )
    if not isinstance(probe_vec, list) or not probe_vec:
        return MemoryHealthReport(
            status=MemoryHealthStatus.DEGRADED,
            latency_ms=int((time.perf_counter() - started) * 1000),
            probe_result_count=0,
            failure_reason="embedding_probe_empty_vector",
        )

    # Check 2 + 3: retrieval success and minimum similarity quality.
    try:
        probe_claim = {"metadata": {"claim_id": config.probe_claim_id}}
        results = memory_layer.retrieve_similar_cases(probe_claim, k=5)
    except Exception as exc:
        return MemoryHealthReport(
            status=MemoryHealthStatus.UNAVAILABLE,
            latency_ms=int((time.perf_counter() - started) * 1000),
            probe_result_count=0,
            failure_reason=f"vector_store_query_failed:{exc}",
        )

    latency_ms = int((time.perf_counter() - started) * 1000)
    if latency_ms > config.probe_timeout_ms:
        return MemoryHealthReport(
            status=MemoryHealthStatus.DEGRADED,
            latency_ms=latency_ms,
            probe_result_count=len(results) if isinstance(results, list) else 0,
            failure_reason="probe_timeout_exceeded",
        )
    if not isinstance(results, list):
        return MemoryHealthReport(
            status=MemoryHealthStatus.DEGRADED,
            latency_ms=latency_ms,
            probe_result_count=0,
            failure_reason="vector_store_query_non_list",
        )
    if len(results) < 1:
        return MemoryHealthReport(
            status=MemoryHealthStatus.DEGRADED,
            latency_ms=latency_ms,
            probe_result_count=0,
            failure_reason="probe_result_count_below_1",
        )
    top_similarity = max(float(item.get("similarity", 0.0)) for item in results)
    if top_similarity <= float(config.min_similarity):
        return MemoryHealthReport(
            status=MemoryHealthStatus.DEGRADED,
            latency_ms=latency_ms,
            probe_result_count=len(results),
            failure_reason="probe_similarity_below_threshold",
        )
    return MemoryHealthReport(
        status=MemoryHealthStatus.HEALTHY,
        latency_ms=latency_ms,
        probe_result_count=len(results),
        failure_reason="",
    )

