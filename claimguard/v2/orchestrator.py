from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List

from crewai import Agent, Crew, Process, Task
from langchain_community.embeddings import FakeEmbeddings, OllamaEmbeddings
from sklearn.cluster import KMeans

from claimguard.llm_factory import assert_ollama_connection, get_llm
from claimguard.v2.blackboard import AgentContract, BlackboardValidationError, SharedBlackboard
from claimguard.v2.concierge import build_routing_decision
from claimguard.v2.consensus import ConsensusEngine
from claimguard.v2.fraud_ring_graph import get_fraud_ring_graph
from claimguard.v2.flow_tracker import get_tracker
from claimguard.v2.memory import (
    CaseMemoryEntry,
    CaseMemoryLayer,
    build_agent_summary,
    decision_to_fraud_label,
    get_memory_layer,
)
from claimguard.v2.schemas import (
    AgentOutput,
    ClaimGuardV2Response,
    MemoryInsights,
    RoutingDecision,
)
from claimguard.v2.trust_layer import TrustLayerIPFSFailure, TrustLayerService

LOGGER = logging.getLogger("claimguard.v2")

SEQUENTIAL_AGENT_CONTRACTS: tuple[AgentContract, ...] = (
    AgentContract("IdentityAgent", ()),
    AgentContract("DocumentAgent", ("IdentityAgent",)),
    AgentContract("PolicyAgent", ("IdentityAgent", "DocumentAgent")),
    AgentContract("AnomalyAgent", ("IdentityAgent", "DocumentAgent", "PolicyAgent")),
    AgentContract("PatternAgent", ("AnomalyAgent",)),
    AgentContract("GraphRiskAgent", ("PatternAgent",)),
)

# Agent-role descriptions injected into CrewAI backstories so the LLM has
# context for its specialty and for how to interpret memory context.
_AGENT_BACKSTORIES: Dict[str, str] = {
    "IdentityAgent": (
        "You are an identity fraud specialist. "
        "You verify CIN validity, cross-reference documents, and detect identity reuse. "
        "When past cases show the same CIN was used fraudulently, you escalate risk."
    ),
    "DocumentAgent": (
        "You are a forensic document analyst. "
        "You assess completeness and authenticity of submitted documents. "
        "If similar past cases showed document forgery at the same provider, you flag it."
    ),
    "PolicyAgent": (
        "You are a compliance and policy risk analyst. "
        "You verify CNSS/CNOPS coverage rules and detect threshold gaming. "
        "If memory shows repeated near-limit claims from the same identity, you escalate."
    ),
    "AnomalyAgent": (
        "You are a behavioral anomaly detection expert. "
        "You surface abnormal amounts, history inconsistencies, and suspicious stability. "
        "Memory of past fraud cases with similar patterns must increase your risk assessment."
    ),
    "PatternAgent": (
        "You are a fraud pattern analyst. "
        "You detect repetition, timing patterns, and scripted billing signatures. "
        "Matching patterns from past fraud cases in memory must be explicitly flagged."
    ),
    "GraphRiskAgent": (
        "You are a network fraud analyst. "
        "You interpret probabilistic graph risk from provider-patient relationship clusters. "
        "If memory shows fraud involving the same hospital or doctor, you escalate risk."
    ),
}


def _safe_json_load(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[-1]
        if raw.startswith("json"):
            raw = raw[4:].lstrip()
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        loaded = {}
    if not isinstance(loaded, dict):
        return {}
    return loaded


def _parse_memory_insights(parsed: Dict[str, Any]) -> MemoryInsights | None:
    """Extract and validate memory_insights from an agent's JSON output."""
    raw = parsed.get("memory_insights")
    if not isinstance(raw, dict):
        return None
    try:
        return MemoryInsights(
            similar_cases_found=int(raw.get("similar_cases_found", 0)),
            fraud_matches=int(raw.get("fraud_matches", 0)),
            identity_reuse_detected=bool(raw.get("identity_reuse_detected", False)),
            impact_on_score=str(raw.get("impact_on_score", "")),
            notes=str(raw.get("notes", "")),
        )
    except Exception:
        return None


def _compute_fallback_memory_insights(
    memory_context: List[Dict[str, Any]],
    current_cin: str,
) -> MemoryInsights:
    """
    When the LLM does not return memory_insights (or returns malformed JSON),
    produce a deterministic fallback from the raw memory context.
    """
    if not memory_context:
        return MemoryInsights(similar_cases_found=0, fraud_matches=0)

    fraud_labels = {"fraud", "suspicious"}
    fraud_matches = sum(
        1 for c in memory_context if c.get("fraud_label", "").lower() in fraud_labels
    )
    identity_reuse = any(
        c.get("cin", "").strip().upper() == current_cin.strip().upper()
        and current_cin.strip()
        for c in memory_context
    )
    if fraud_matches > 0:
        impact = f"Memory contains {fraud_matches} fraud/suspicious case(s) — risk elevated"
    else:
        impact = "No fraud matches found in memory context"

    notes_parts: List[str] = []
    if identity_reuse:
        notes_parts.append("CIN reuse detected across memory cases")
    cinset = {c.get("cin", "") for c in memory_context if c.get("cin")}
    if len(cinset) == 1 and next(iter(cinset)) == current_cin:
        notes_parts.append("All memory cases share the same CIN")
    hospitals = {c.get("hospital", "") for c in memory_context if c.get("hospital")}
    if hospitals:
        notes_parts.append(f"Recurring hospital(s) in memory: {', '.join(sorted(hospitals))}")

    return MemoryInsights(
        similar_cases_found=len(memory_context),
        fraud_matches=fraud_matches,
        identity_reuse_detected=identity_reuse,
        impact_on_score=impact,
        notes="; ".join(notes_parts) if notes_parts else "Memory advisory only",
    )


class ClaimGuardV2Orchestrator:
    def __init__(
        self,
        *,
        trust_layer_service: TrustLayerService | None = None,
        memory_layer: CaseMemoryLayer | None = None,
    ) -> None:
        self._ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        self._embedding_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        self._consensus_engine = ConsensusEngine()
        self._trust_layer = trust_layer_service or TrustLayerService.build_default()
        self._memory = memory_layer or get_memory_layer()
        self._fraud_ring_graph = get_fraud_ring_graph()
        print("LLM Provider: OLLAMA")
        print("Available models:", ["mistral", "llama3", "deepseek-r1"])
        assert_ollama_connection()

    def _make_chat_llm(self, model_name: str):
        return get_llm(model_name)

    def _build_task_prompt(
        self,
        *,
        contract: AgentContract,
        claim_request: Dict[str, Any],
        blackboard: SharedBlackboard,
    ) -> str:
        memory_context = blackboard.memory_context
        memory_section = ""
        if memory_context:
            memory_section = (
                "\n\nMEMORY CONTEXT — PAST SIMILAR CASES (advisory, similarity >= threshold):\n"
                + json.dumps(memory_context, ensure_ascii=False, indent=2)
                + "\n\n"
                "MEMORY ANALYSIS RULES (MANDATORY):\n"
                "1. Examine every case in memory_context above.\n"
                "2. Answer internally: Have I seen similar cases? Do they indicate fraud?\n"
                "3. If past cases show fraud with the same CIN → flag identity reuse, increase risk.\n"
                "4. If past cases show fraud at the same hospital/doctor → increase risk.\n"
                "5. If similarity < 0.7 (already filtered) → ignore. Memory is ADVISORY, not absolute.\n"
                "6. If memory contradicts current data → note the contradiction, do NOT override score.\n"
                "7. Include 'memory_insights' in your JSON output (see schema below).\n"
            )
        else:
            memory_section = (
                "\n\nMEMORY CONTEXT: No similar past cases found above the similarity threshold.\n"
            )

        return (
            f"You are the {contract.name}. {_AGENT_BACKSTORIES.get(contract.name, '')}\n\n"
            "Evaluate the insurance claim context for your specialty.\n"
            "Return ONLY a single JSON object with these keys:\n"
            "  score       (float 0..1)   — your fraud risk assessment\n"
            "  confidence  (float 0..1)   — confidence in your score\n"
            "  explanation (string)       — concise reasoning\n"
            "  memory_insights (object)   — analysis of memory context:\n"
            "    { similar_cases_found: int, fraud_matches: int,\n"
            "      identity_reuse_detected: bool,\n"
            "      impact_on_score: string, notes: string }\n\n"
            # NOTE: "Claim: " / "Blackboard: " prefixes are intentionally stable:
            # tests and red-team tooling parse these exact lines to validate that
            # blackboard context was correctly passed to agents.
            f"Current agent: {contract.name}\n"
            f"Claim: {json.dumps(claim_request, ensure_ascii=False)}\n"
            f"Blackboard: {json.dumps(blackboard.to_dict(), ensure_ascii=False)}\n"
            + memory_section
        )

    @staticmethod
    def _goa_trigger(claim_request: Dict[str, Any], routing: RoutingDecision) -> bool:
        documents = claim_request.get("documents", [])
        return len(documents) > 1 or routing.complexity == "complex"

    def _get_embeddings(self):
        try:
            try:
                from langchain_ollama import OllamaEmbeddings as _OllamaEmb  # type: ignore
                return _OllamaEmb(base_url=self._ollama_base_url, model=self._embedding_model)
            except ImportError:
                pass
            return OllamaEmbeddings(base_url=self._ollama_base_url, model=self._embedding_model)
        except Exception:
            return FakeEmbeddings(size=64)

    def _cluster_documents(self, documents: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        if len(documents) <= 1:
            return [documents]
        embedder = self._get_embeddings()
        vectors = embedder.embed_documents(
            [json.dumps(d, ensure_ascii=False, sort_keys=True) for d in documents]
        )
        cluster_count = min(3, len(documents))
        labels = KMeans(n_clusters=cluster_count, random_state=42, n_init=10).fit_predict(vectors)
        grouped: Dict[int, List[Dict[str, Any]]] = {}
        for idx, label in enumerate(labels):
            grouped.setdefault(int(label), []).append(documents[idx])
        return list(grouped.values())

    def _run_goa(self, claim_request: Dict[str, Any], routing: RoutingDecision) -> Dict[str, Any]:
        docs = claim_request.get("documents", [])
        clusters = self._cluster_documents(docs)
        outputs: List[Dict[str, Any]] = []
        for i, cluster in enumerate(clusters):
            llm = self._make_chat_llm(routing.model)
            cluster_agent = Agent(
                role=f"ClusterRiskAgent-{i}",
                goal="Assess cluster-level risk and anomalies",
                backstory="Specialist in clustered claim document patterns.",
                llm=llm,
                verbose=False,
            )
            cluster_task = Task(
                description=(
                    "Assess risk for this document cluster and return JSON "
                    "with score, confidence, explanation.\n"
                    f"Cluster: {json.dumps(cluster, ensure_ascii=False)}"
                ),
                expected_output="JSON with score, confidence, explanation",
                agent=cluster_agent,
            )
            out = Crew(
                agents=[cluster_agent],
                tasks=[cluster_task],
                process=Process.sequential,
                verbose=False,
            ).kickoff()
            parsed = _safe_json_load(str(out))
            outputs.append(
                {
                    "cluster_id": i,
                    "size": len(cluster),
                    "result": {
                        "score": float(parsed.get("score", 0.5)),
                        "confidence": float(parsed.get("confidence", 0.5)),
                        "explanation": str(parsed.get("explanation", "No explanation provided")),
                    },
                }
            )
        return {"clusters": len(clusters), "cluster_outputs": outputs}

    def _resolve_current_cin(self, claim_request: Dict[str, Any]) -> str:
        identity = claim_request.get("identity", {})
        for key in ("cin", "CIN", "carte_nationale", "national_id"):
            val = identity.get(key)
            if val and str(val).strip():
                return str(val).strip().upper()
        for key in ("patient_id", "cin", "CIN"):
            val = claim_request.get(key)
            if val and str(val).strip():
                return str(val).strip().upper()
        return ""

    @staticmethod
    def _resolve_graph_fields(claim_request: Dict[str, Any]) -> Dict[str, Any]:
        identity = claim_request.get("identity", {})
        policy = claim_request.get("policy", {})
        metadata = claim_request.get("metadata", {})

        claim_id = str(
            metadata.get("claim_id")
            or claim_request.get("claim_id")
            or claim_request.get("id")
            or ""
        ).strip()
        cin = str(
            identity.get("cin")
            or identity.get("CIN")
            or claim_request.get("patient_id")
            or ""
        ).strip().upper()
        hospital = str(
            identity.get("hospital")
            or policy.get("hospital")
            or metadata.get("hospital")
            or ""
        ).strip()
        doctor = str(
            identity.get("doctor")
            or policy.get("doctor")
            or metadata.get("doctor")
            or ""
        ).strip()

        raw_anomaly = (
            metadata.get("anomaly_score")
            or claim_request.get("anomaly_score")
            or policy.get("anomaly_score")
            or 0.0
        )
        try:
            anomaly_score = float(raw_anomaly)
        except (TypeError, ValueError):
            anomaly_score = 0.0

        return {
            "claim_id": claim_id,
            "cin": cin,
            "hospital": hospital,
            "doctor": doctor,
            "anomaly_score": max(0.0, min(1.0, anomaly_score)),
        }

    def run(self, claim_request: Dict[str, Any]) -> ClaimGuardV2Response:
        routing = build_routing_decision(claim_request)
        blackboard = SharedBlackboard(claim_request, routing)

        # ── Step 1: Retrieve memory context BEFORE any agent runs ──────────
        claim_id = str(claim_request.get("metadata", {}).get("claim_id") or "")
        if not claim_id:
            claim_id = f"v2-{int(perf_counter() * 1000)}"
        
        tracker = get_tracker(claim_id)
        
        memory_context = self._memory.retrieve_similar_cases(claim_request)
        blackboard.inject_memory_context(memory_context)
        current_cin = self._resolve_current_cin(claim_request)

        if memory_context:
            fraud_cases = [c for c in memory_context if c.get("fraud_label") in ("fraud", "suspicious")]
            LOGGER.info(
                "memory_context_injected total=%d fraud_cases=%d cin=%s",
                len(memory_context), len(fraud_cases), current_cin,
            )
        else:
            LOGGER.info("memory_context_empty claim_cin=%s", current_cin)

        # ── Step 2: Run agents sequentially ────────────────────────────────
        agent_outputs: List[AgentOutput] = []
        fraud_ring_analysis: Dict[str, Any] = {"fraud_rings": []}

        for contract in SEQUENTIAL_AGENT_CONTRACTS:
            try:
                blackboard.require(contract.requires)
            except BlackboardValidationError:
                tracker.update(contract.name, "SKIPPED")
                for remaining in SEQUENTIAL_AGENT_CONTRACTS[SEQUENTIAL_AGENT_CONTRACTS.index(contract)+1:]:
                    tracker.update(remaining.name, "SKIPPED")
                raise
            
            tracker.update(contract.name, "RUNNING")
            started = perf_counter()
            llm = self._make_chat_llm(routing.model)
            agent = Agent(
                role=contract.name,
                goal="Evaluate claim risk for your scope and produce calibrated output.",
                backstory=_AGENT_BACKSTORIES.get(
                    contract.name,
                    "Insurance fraud specialist collaborating through a shared blackboard.",
                ),
                llm=llm,
                verbose=False,
            )
            prompt = self._build_task_prompt(
                contract=contract,
                claim_request=claim_request,
                blackboard=blackboard,
            )
            task = Task(
                description=prompt,
                expected_output=(
                    "JSON with keys: score, confidence, explanation, memory_insights"
                ),
                agent=agent,
            )
            crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=False,
            )
            try:
                result = crew.kickoff()
                parsed = _safe_json_load(str(result))
            except Exception as exc:
                tracker.update(contract.name, "FAILED")
                LOGGER.exception("v2_agent_failed agent=%s", contract.name)
                raise RuntimeError(
                    f"Ollama agent execution failed for {contract.name}: {exc}"
                ) from exc
            
            score = float(parsed.get("score", 0.0))
            confidence = float(parsed.get("confidence", 0.0))
            explanation = str(parsed.get("explanation", "No explanation provided"))

            # Extract or synthesise memory_insights
            memory_insights = _parse_memory_insights(parsed)
            if memory_insights is None and memory_context:
                memory_insights = _compute_fallback_memory_insights(memory_context, current_cin)

            if contract.name == "GraphRiskAgent":
                graph_fields = self._resolve_graph_fields(claim_request)
                if graph_fields["claim_id"] and graph_fields["cin"] and graph_fields["hospital"] and graph_fields["doctor"]:
                    graph_context = self._fraud_ring_graph.add_claim(**graph_fields)
                else:
                    graph_context = {
                        "cluster_membership": None,
                        "reuse_detection": {
                            "cin_reuse_detected": False,
                            "doctor_reuse_detected": False,
                            "cin_claim_count": 0,
                            "doctor_claim_count": 0,
                        },
                        "network_risk_score": 0.0,
                        "fraud_rings": {"fraud_rings": []},
                    }

                fraud_ring_analysis = graph_context.get("fraud_rings", {"fraud_rings": []})
                graph_risk = float(graph_context.get("network_risk_score", 0.0))
                parsed["score"] = max(float(parsed.get("score", 0.0)), graph_risk)
                parsed["confidence"] = max(float(parsed.get("confidence", 0.0)), min(1.0, 0.55 + (graph_risk * 0.4)))
                parsed["explanation"] = (
                    f"{parsed.get('explanation', '').strip()} "
                    f"[graph cluster={graph_context.get('cluster_membership')}; "
                    f"cin_reuse={graph_context['reuse_detection']['cin_reuse_detected']}; "
                    f"doctor_reuse={graph_context['reuse_detection']['doctor_reuse_detected']}; "
                    f"network_risk={graph_risk:.3f}]"
                ).strip()
                score = parsed["score"]
                confidence = parsed["confidence"]
                explanation = parsed["explanation"]

            is_fraud = score >= 0.7
            tracker.update(contract.name, "COMPLETED", score=score, confidence=confidence, explanation=explanation, is_fraud=is_fraud)
            
            elapsed_ms = int((perf_counter() - started) * 1000)

            blackboard.append(
                contract.name,
                score=score,
                confidence=confidence,
                explanation=explanation,
            )

            output = AgentOutput(
                agent=contract.name,
                score=score,
                confidence=confidence,
                explanation=explanation,
                elapsed_ms=elapsed_ms,
                input_snapshot={
                    "required_context": list(contract.requires),
                    "routing_model": routing.model,
                    "memory_cases_available": len(memory_context),
                    "graph_context": graph_context if contract.name == "GraphRiskAgent" else {},
                },
                output_snapshot=parsed,
                memory_insights=memory_insights,
            )
            agent_outputs.append(output)
            LOGGER.info(
                "v2_agent_complete agent=%s elapsed_ms=%s confidence=%.3f "
                "memory_fraud_matches=%s input=%s output=%s",
                contract.name,
                elapsed_ms,
                confidence,
                memory_insights.fraud_matches if memory_insights else 0,
                output.input_snapshot,
                output.output_snapshot,
            )

            # Early exit if identity is invalid
            if contract.name == "IdentityAgent" and score > 0.8:
                tracker.update("IdentityAgent", "FAILED")
                for remaining in SEQUENTIAL_AGENT_CONTRACTS[SEQUENTIAL_AGENT_CONTRACTS.index(contract)+1:]:
                    tracker.update(remaining.name, "SKIPPED")
                break

        # ── Step 3: GOA ────────────────────────────────────────────────────
        goa_used = self._goa_trigger(claim_request, routing)
        if goa_used:
            goa_payload = self._run_goa(claim_request, routing)
            blackboard.append(
                "GraphOfAgents",
                score=0.5,
                confidence=0.5,
                explanation=f"Generated {goa_payload['clusters']} dynamic clusters",
            )
            blackboard_state = blackboard.to_dict()
            blackboard_state["goa"] = goa_payload
        else:
            blackboard_state = blackboard.to_dict()
        blackboard_state["fraud_ring_analysis"] = fraud_ring_analysis

        # ── Step 4: Consensus ──────────────────────────────────────────────
        tracker.update("Consensus", "RUNNING")
        consensus_result = self._consensus_engine.evaluate(
            claim_request=claim_request,
            entries=blackboard_state.get("entries", {}),
        )
        tracker.update("Consensus", "COMPLETED")
        
        if consensus_result["decision"] == "HUMAN_REVIEW":
            tracker.update("HumanReview", "RUNNING")
        else:
            tracker.update("HumanReview", "SKIPPED")
        blackboard_state["reflexive_retry_logs"] = consensus_result["retry_logs"]
        blackboard_state["score_evolution"] = consensus_result["score_evolution"]
        blackboard_state["consensus_entries"] = consensus_result["entries"]
        LOGGER.info(
            "consensus_final Ts=%.2f decision=%s retries=%s score_evolution=%s",
            consensus_result["Ts"],
            consensus_result["decision"],
            consensus_result["retry_count"],
            consensus_result["score_evolution"],
        )

        # ── Step 5: Trust layer ────────────────────────────────────────────
        trust_layer_payload = None
        try:
            trust_layer_payload = self._trust_layer.process_if_applicable(
                claim_id=claim_id,
                decision=consensus_result["decision"],
                ts_score=consensus_result["Ts"],
                claim_request=claim_request,
                agent_outputs=[item.model_dump() for item in agent_outputs],
            )
        except TrustLayerIPFSFailure:
            raise
        except Exception as exc:
            LOGGER.exception("trust_layer_unexpected_error claim_id=%s error=%s", claim_id, exc)

        # ── Step 7: Store claim in memory AFTER consensus ──────────────────
        self._store_claim_in_memory(
            claim_id=claim_id,
            claim_request=claim_request,
            agent_outputs=agent_outputs,
            decision=consensus_result["decision"],
            ts_score=consensus_result["Ts"],
        )

        return ClaimGuardV2Response(
            agent_outputs=agent_outputs,
            blackboard=blackboard_state,
            routing_decision=routing,
            goa_used=goa_used,
            Ts=consensus_result["Ts"],
            decision=consensus_result["decision"],
            retry_count=consensus_result["retry_count"],
            mahic_breakdown=consensus_result["mahic_breakdown"],
            contradictions=consensus_result["contradictions"],
            trust_layer=trust_layer_payload.model_dump() if trust_layer_payload else None,
            memory_context=memory_context,
        )

    def _store_claim_in_memory(
        self,
        *,
        claim_id: str,
        claim_request: Dict[str, Any],
        agent_outputs: List[AgentOutput],
        decision: str,
        ts_score: float,
    ) -> None:
        """Build a CaseMemoryEntry from this claim's result and persist it."""
        try:
            identity = claim_request.get("identity", {})
            policy = claim_request.get("policy", {})
            metadata = claim_request.get("metadata", {})

            cin = (
                identity.get("cin") or identity.get("CIN")
                or claim_request.get("patient_id", "")
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
                or claim_request.get("diagnosis", "")
            )

            fraud_label = decision_to_fraud_label(decision, ts_score)
            agent_summary = build_agent_summary(
                [ao.model_dump() for ao in agent_outputs]
            )

            entry = CaseMemoryEntry(
                claim_id=claim_id,
                cin=str(cin),
                hospital=str(hospital),
                doctor=str(doctor),
                diagnosis=str(diagnosis),
                fraud_label=fraud_label,
                ts_score=ts_score,
                agent_summary=agent_summary,
            )
            self._memory.store_case(entry)
            LOGGER.info(
                "claim_stored_in_memory claim_id=%s fraud_label=%s Ts=%.2f",
                claim_id, fraud_label, ts_score,
            )
        except Exception as exc:
            LOGGER.error("memory_store_claim_failed claim_id=%s error=%s", claim_id, exc)

    def get_fraud_graph_debug(self, *, render_png: bool = False) -> Dict[str, Any]:
        rings = self._fraud_ring_graph.detect_fraud_rings()
        payload: Dict[str, Any] = {
            "fraud_rings": rings.get("fraud_rings", []),
            "node_count": int(self._fraud_ring_graph.graph.number_of_nodes()),
            "edge_count": int(self._fraud_ring_graph.graph.number_of_edges()),
            "png_path": None,
        }
        if render_png:
            artifact_dir = Path(__file__).resolve().parents[2] / "tests" / "artifacts"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            output_path = artifact_dir / "v2_fraud_graph.png"
            payload["png_path"] = self._fraud_ring_graph.visualize_graph(output_path)
        return payload


_singleton: ClaimGuardV2Orchestrator | None = None


def get_v2_orchestrator() -> ClaimGuardV2Orchestrator:
    global _singleton
    if _singleton is None:
        _singleton = ClaimGuardV2Orchestrator()
    return _singleton
