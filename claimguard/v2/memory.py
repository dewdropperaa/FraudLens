"""Case Memory Layer for ClaimGuard v2.

Stores processed claims as dense vector embeddings and enables
similarity-based retrieval so agents can reason over past cases
for improved fraud detection.

Vector backend: FAISS IndexFlatIP (inner-product on L2-normalised vectors ==
cosine similarity).  Falls back to pure-Python cosine search when faiss-cpu is
not installed.  Embeddings: OllamaEmbeddings → FakeEmbeddings (64-d) fallback.
"""
from __future__ import annotations

import json
import logging
import math
import os
import pickle
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

LOGGER = logging.getLogger("claimguard.v2.memory")

# ── Runtime configuration (env-overridable) ────────────────────────────────
SIMILARITY_THRESHOLD: float = float(os.getenv("MEMORY_SIMILARITY_THRESHOLD", "0.7"))
MEMORY_DEFAULT_K: int = int(os.getenv("MEMORY_DEFAULT_K", "5"))
_OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
_OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
MEMORY_STORE_PATH: str = os.getenv("MEMORY_STORE_PATH", "claimguard_memory_store")


# ── Helpers ────────────────────────────────────────────────────────────────

def _build_embedding_text(
    *,
    cin: str,
    hospital: str,
    doctor: str,
    diagnosis: str,
    anomalies: str,
    fraud_label: str,
) -> str:
    """Concatenate all similarity axes into a single string for embedding.

    Only non-empty fields are included so the embedding is not polluted by
    placeholder noise.
    """
    fields = [
        ("CIN", cin),
        ("hospital", hospital),
        ("doctor", doctor),
        ("diagnosis", diagnosis),
        ("anomalies", anomalies),
        ("decision", fraud_label),
    ]
    return " | ".join(f"{k}: {v}" for k, v in fields if v and v.strip())


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def _normalize(vec: List[float]) -> List[float]:
    mag = math.sqrt(sum(x * x for x in vec))
    return [x / mag for x in vec] if mag > 0.0 else vec


# ── CaseMemoryEntry ────────────────────────────────────────────────────────

class CaseMemoryEntry:
    """Immutable snapshot of a fully processed claim persisted in memory."""

    __slots__ = (
        "claim_id", "cin", "hospital", "doctor", "diagnosis",
        "fraud_label", "ts_score", "agent_summary", "timestamp",
        "embedding_text",
    )

    def __init__(
        self,
        *,
        claim_id: str,
        cin: str,
        hospital: str,
        doctor: str,
        diagnosis: str,
        fraud_label: str,
        ts_score: float,
        agent_summary: str,
        timestamp: Optional[str] = None,
    ) -> None:
        self.claim_id = claim_id
        self.cin = cin
        self.hospital = hospital
        self.doctor = doctor
        self.diagnosis = diagnosis
        self.fraud_label = fraud_label          # "clean" | "suspicious" | "fraud"
        self.ts_score = float(ts_score)
        self.agent_summary = agent_summary
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        self.embedding_text = _build_embedding_text(
            cin=cin,
            hospital=hospital,
            doctor=doctor,
            diagnosis=diagnosis,
            anomalies=agent_summary,
            fraud_label=fraud_label,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "cin": self.cin,
            "hospital": self.hospital,
            "doctor": self.doctor,
            "diagnosis": self.diagnosis,
            "fraud_label": self.fraud_label,
            "ts_score": self.ts_score,
            "agent_summary": self.agent_summary,
            "timestamp": self.timestamp,
            "embedding_text": self.embedding_text,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CaseMemoryEntry":
        return cls(
            claim_id=d["claim_id"],
            cin=d["cin"],
            hospital=d["hospital"],
            doctor=d["doctor"],
            diagnosis=d["diagnosis"],
            fraud_label=d["fraud_label"],
            ts_score=d["ts_score"],
            agent_summary=d["agent_summary"],
            timestamp=d.get("timestamp"),
        )


# ── CaseMemoryLayer ────────────────────────────────────────────────────────

class CaseMemoryLayer:
    """
    Vector-based case memory for ClaimGuard v2.

    Architecture
    ────────────
    • Embeddings : OllamaEmbeddings (nomic-embed-text) → FakeEmbeddings(64-d)
    • Index      : FAISS IndexFlatIP (cosine on L2-normalised vecs) → pure-Python fallback
    • Persistence: <store_path>/entries.pkl, vectors.pkl, faiss.index

    Safety
    ──────
    • Memory is advisory: callers MUST filter results by similarity_threshold.
    • A failed store/retrieve never crashes the main pipeline (all errors are logged).
    """

    def __init__(
        self,
        *,
        ollama_base_url: str = _OLLAMA_BASE_URL,
        embed_model: str = _OLLAMA_EMBED_MODEL,
        store_path: str = MEMORY_STORE_PATH,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ) -> None:
        self._ollama_base_url = ollama_base_url
        self._embed_model = embed_model
        self._store_path = store_path
        self._similarity_threshold = similarity_threshold

        self._entries: List[CaseMemoryEntry] = []
        self._vectors: List[List[float]] = []
        self._faiss_index: Any = None
        self._use_faiss: bool = False
        self._faiss_checked: bool = False   # avoid repeated import attempts

        self._embedder = self._init_embedder()
        self._load_store()

    # ── Embedder setup ─────────────────────────────────────────────────────

    def _init_embedder(self) -> Any:
        try:
            # Prefer langchain-ollama (newer, avoids deprecation warning)
            try:
                from langchain_ollama import OllamaEmbeddings  # type: ignore
            except ImportError:
                from langchain_community.embeddings import OllamaEmbeddings  # type: ignore
            emb = OllamaEmbeddings(base_url=self._ollama_base_url, model=self._embed_model)
            emb.embed_query("probe")
            LOGGER.info("memory_embedder=ollama model=%s", self._embed_model)
            return emb
        except Exception as exc:
            LOGGER.warning("memory_embedder_ollama_unavailable reason=%s — using FakeEmbeddings", exc)
            from langchain_community.embeddings import FakeEmbeddings
            return FakeEmbeddings(size=64)

    def _embed(self, text: str) -> List[float]:
        try:
            return self._embedder.embed_query(text)
        except Exception as exc:
            LOGGER.warning("memory_embed_failed error=%s — falling back to FakeEmbeddings", exc)
            from langchain_community.embeddings import FakeEmbeddings
            return FakeEmbeddings(size=64).embed_query(text)

    # ── FAISS setup ────────────────────────────────────────────────────────

    def _try_init_faiss(self, dim: int) -> None:
        if self._faiss_index is not None or self._faiss_checked:
            return
        self._faiss_checked = True
        try:
            import faiss  # type: ignore
            self._faiss_index = faiss.IndexFlatIP(dim)
            self._use_faiss = True
            LOGGER.info("memory_vector_store=faiss dim=%d", dim)
        except Exception as exc:
            LOGGER.warning("faiss_unavailable — using cosine fallback. reason=%s", exc)
            self._use_faiss = False

    # ── Public API ─────────────────────────────────────────────────────────

    def store_case(self, entry: CaseMemoryEntry) -> None:
        """Embed and persist a processed claim into memory."""
        try:
            vec = _normalize(self._embed(entry.embedding_text))
            dim = len(vec)
            self._try_init_faiss(dim)

            self._entries.append(entry)
            self._vectors.append(vec)

            if self._use_faiss and self._faiss_index is not None:
                import numpy as np  # type: ignore
                self._faiss_index.add(np.array([vec], dtype="float32"))

            self._persist_store()
            LOGGER.info(
                "memory_stored claim_id=%s fraud_label=%s Ts=%.2f",
                entry.claim_id, entry.fraud_label, entry.ts_score,
            )
        except Exception as exc:
            LOGGER.error("memory_store_failed claim_id=%s error=%s", entry.claim_id, exc)

    def retrieve_similar_cases(
        self,
        current_claim: Dict[str, Any],
        k: int = MEMORY_DEFAULT_K,
    ) -> List[Dict[str, Any]]:
        """
        Return top-k past cases similar to `current_claim`.

        Similarity captures:
          • Same CIN                    (identity reuse detection)
          • Same hospital / doctor      (provider-level fraud patterns)
          • Similar anomaly patterns    (behavioural fingerprinting)
          • Similar diagnosis/billing   (clinical fraud patterns)

        Only results with similarity >= similarity_threshold are returned.
        Each result dict: claim_id, cin, fraud_label, similarity, summary,
                          hospital, doctor, diagnosis, ts_score, timestamp.
        """
        if not self._entries:
            return []

        query_text = _build_query_text(current_claim)
        try:
            query_vec = _normalize(self._embed(query_text))
        except Exception as exc:
            LOGGER.error("memory_retrieve_embed_failed error=%s", exc)
            return []

        if self._use_faiss and self._faiss_index is not None:
            raw_results = self._faiss_search(query_vec, k)
        else:
            raw_results = self._cosine_search(query_vec, k)

        filtered = [r for r in raw_results if r["similarity"] >= self._similarity_threshold]

        cin = (
            current_claim.get("identity", {}).get("cin")
            or current_claim.get("cin")
            or current_claim.get("patient_id", "?")
        )
        LOGGER.info(
            "memory_retrieved total_candidates=%d above_threshold=%d cin=%s",
            len(raw_results), len(filtered), cin,
        )
        return filtered

    # ── Search backends ────────────────────────────────────────────────────

    def _faiss_search(self, query_vec: List[float], k: int) -> List[Dict[str, Any]]:
        import numpy as np  # type: ignore
        arr = np.array([query_vec], dtype="float32")
        actual_k = min(k, len(self._entries))
        scores, indices = self._faiss_index.search(arr, actual_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._entries):
                continue
            results.append(_format_result(self._entries[idx], float(score)))
        return results

    def _cosine_search(self, query_vec: List[float], k: int) -> List[Dict[str, Any]]:
        scored: List[Tuple[float, int]] = [
            (_cosine(query_vec, v), i) for i, v in enumerate(self._vectors)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [_format_result(self._entries[idx], score) for score, idx in scored[:k]]

    # ── Persistence ────────────────────────────────────────────────────────

    def _persist_store(self) -> None:
        if not self._store_path:
            return
        try:
            os.makedirs(self._store_path, exist_ok=True)
            with open(os.path.join(self._store_path, "entries.pkl"), "wb") as f:
                pickle.dump([e.to_dict() for e in self._entries], f)
            with open(os.path.join(self._store_path, "vectors.pkl"), "wb") as f:
                pickle.dump(self._vectors, f)
            if self._use_faiss and self._faiss_index is not None:
                import faiss  # type: ignore
                faiss.write_index(
                    self._faiss_index,
                    os.path.join(self._store_path, "faiss.index"),
                )
        except Exception as exc:
            LOGGER.warning("memory_persist_failed error=%s", exc)

    def _load_store(self) -> None:
        if not self._store_path:
            return
        entries_path = os.path.join(self._store_path, "entries.pkl")
        vectors_path = os.path.join(self._store_path, "vectors.pkl")
        if not (os.path.exists(entries_path) and os.path.exists(vectors_path)):
            return
        try:
            with open(entries_path, "rb") as f:
                raw = pickle.load(f)
            with open(vectors_path, "rb") as f:
                self._vectors = pickle.load(f)
            self._entries = [CaseMemoryEntry.from_dict(d) for d in raw]

            faiss_path = os.path.join(self._store_path, "faiss.index")
            if os.path.exists(faiss_path):
                try:
                    import faiss  # type: ignore
                    self._faiss_index = faiss.read_index(faiss_path)
                    self._use_faiss = True
                    LOGGER.info("memory_loaded entries=%d store=faiss", len(self._entries))
                    return
                except Exception:
                    pass

            if self._vectors:
                self._try_init_faiss(len(self._vectors[0]))
            LOGGER.info("memory_loaded entries=%d store=%s", len(self._entries),
                        "faiss" if self._use_faiss else "cosine")
        except Exception as exc:
            LOGGER.warning("memory_load_failed error=%s — starting fresh", exc)
            self._entries = []
            self._vectors = []

    @property
    def entry_count(self) -> int:
        return len(self._entries)


# ── Module-level helpers ───────────────────────────────────────────────────

def _format_result(entry: CaseMemoryEntry, similarity: float) -> Dict[str, Any]:
    return {
        "claim_id": entry.claim_id,
        "cin": entry.cin,
        "fraud_label": entry.fraud_label,
        "similarity": round(similarity, 4),
        "summary": entry.agent_summary,
        "hospital": entry.hospital,
        "doctor": entry.doctor,
        "diagnosis": entry.diagnosis,
        "ts_score": entry.ts_score,
        "timestamp": entry.timestamp,
    }


def _build_query_text(claim: Dict[str, Any]) -> str:
    identity = claim.get("identity", {})
    policy = claim.get("policy", {})
    metadata = claim.get("metadata", {})

    cin = (
        identity.get("cin") or identity.get("CIN")
        or claim.get("cin") or claim.get("patient_id", "")
    )
    hospital = (
        identity.get("hospital") or policy.get("hospital")
        or metadata.get("hospital", "")
    )
    doctor = (
        identity.get("doctor") or policy.get("doctor")
        or metadata.get("doctor", "")
    )
    diagnosis = (
        policy.get("diagnosis") or metadata.get("diagnosis")
        or claim.get("diagnosis", "")
    )
    return _build_embedding_text(
        cin=str(cin),
        hospital=str(hospital),
        doctor=str(doctor),
        diagnosis=str(diagnosis),
        anomalies="",
        fraud_label="",
    )


def decision_to_fraud_label(decision: str, ts_score: float) -> str:
    """Convert an orchestrator decision + Ts score into a fraud label."""
    if decision == "AUTO_APPROVE":
        return "clean"
    if decision in ("REJECTED", "REFLEXIVE_TRIGGER"):
        return "fraud"
    # HUMAN_REVIEW — use Ts to distinguish borderline cases
    return "suspicious" if ts_score >= 60 else "fraud"


def build_agent_summary(agent_outputs: List[Dict[str, Any]]) -> str:
    """Build a concise summary from agent outputs for the embedding_text."""
    parts: List[str] = []
    for ao in agent_outputs:
        agent = ao.get("agent", "")
        explanation = ao.get("explanation", "")
        score = ao.get("score", 0.0)
        if explanation:
            parts.append(f"{agent}(score={score:.2f}): {explanation[:120]}")
    return " | ".join(parts) if parts else "no agent summary"


# ── Singleton ──────────────────────────────────────────────────────────────

_memory_singleton: CaseMemoryLayer | None = None


def get_memory_layer() -> CaseMemoryLayer:
    global _memory_singleton
    if _memory_singleton is None:
        _memory_singleton = CaseMemoryLayer()
    return _memory_singleton
