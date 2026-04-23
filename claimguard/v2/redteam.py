from __future__ import annotations

import copy
import json
import random
import re
import shutil
from contextlib import contextmanager
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Dict, List, Tuple

from claimguard.v2.memory import CaseMemoryLayer
from claimguard.v2.orchestrator import ClaimGuardV2Orchestrator

_CIN_TOKEN_RE = re.compile(r"\b[A-Z]{2,6}-?\d{4,}\b")
_SCENARIOS = ("clean", "suspicious", "fraud", "adversarial", "edge")


class _NoOpTrustLayer:
    def process_if_applicable(self, **_: Any) -> None:
        return None


class _NoMemoryLayer:
    def retrieve_similar_cases(self, current_claim: Dict[str, Any], k: int = 5) -> List[Dict[str, Any]]:
        _ = (current_claim, k)
        return []

    def store_case(self, entry: Any) -> None:
        _ = entry


@dataclass
class StrictModeConfig:
    critical_failures_threshold: int = 5
    hallucination_rate_max: float = 0.2
    fraud_detection_target: float = 0.7


@dataclass
class RedTeamConfig:
    claim_count: int = 100
    random_seed: int = 42
    strict_mode: StrictModeConfig = field(default_factory=StrictModeConfig)
    artifact_dir: Path = Path("tests/artifacts")
    generate_visualizations: bool = True
    use_simulated_agents: bool = False


class _SimAgent:
    def __init__(self, role: str, **_: Any) -> None:
        self.role = role


class _SimTask:
    def __init__(self, description: str, agent: _SimAgent, **_: Any) -> None:
        self.description = description
        self.agent = agent


def _extract_json(prefix: str, text: str) -> Dict[str, Any]:
    for line in text.splitlines():
        if line.startswith(prefix):
            try:
                return json.loads(line[len(prefix):].strip())
            except Exception:
                return {}
    return {}


def _simulated_payload(
    *,
    role: str,
    claim: Dict[str, Any],
    blackboard: Dict[str, Any],
) -> Dict[str, Any]:
    scenario = str(claim.get("metadata", {}).get("scenario", "clean"))
    docs = claim.get("documents", [])
    has_docs = bool(docs)
    identity_invalid = bool(claim.get("metadata", {}).get("identity_invalid", False))
    memory_context = blackboard.get("memory_context", [])
    memory_matches = sum(1 for item in memory_context if item.get("fraud_label") in {"fraud", "suspicious"})

    base = {
        "clean": {
            "IdentityAgent": (0.9, 0.9, "Identity verified from CIN and records."),
            "DocumentAgent": (0.88, 0.86, "Documents are coherent and complete."),
            "PolicyAgent": (0.86, 0.87, "Policy eligibility checks pass."),
            "AnomalyAgent": (0.2, 0.78, "No anomaly trend observed."),
            "PatternAgent": (0.25, 0.76, "No repeated suspicious pattern."),
            "GraphRiskAgent": (0.2, 0.74, "Network links do not indicate fraud ring."),
        },
        "suspicious": {
            "IdentityAgent": (0.68, 0.72, "Identity mostly valid but context is borderline."),
            "DocumentAgent": (0.55, 0.7, "Invoice fields show minor inconsistencies."),
            "PolicyAgent": (0.63, 0.71, "Coverage appears valid with policy edge conditions."),
            "AnomalyAgent": (0.72, 0.74, "Billing behavior is moderately abnormal."),
            "PatternAgent": (0.69, 0.72, "Partial repetition in provider patterns."),
            "GraphRiskAgent": (0.66, 0.73, "Moderate network risk with linked entities."),
        },
        "fraud": {
            "IdentityAgent": (0.15, 0.88, "Identity mismatch detected across submitted fields."),
            "DocumentAgent": (0.18, 0.84, "Document integrity strongly suggests alteration."),
            "PolicyAgent": (0.42, 0.79, "Policy context conflicts with claim data."),
            "AnomalyAgent": (0.93, 0.86, "Strong anomaly profile indicates probable fraud."),
            "PatternAgent": (0.95, 0.84, "Repeated fraud signatures detected."),
            "GraphRiskAgent": (0.9, 0.81, "Connected entities suggest coordinated fraud ring."),
        },
        "adversarial": {
            "IdentityAgent": (0.48, 0.69, "Identity plausibility is reduced by conflicting fields."),
            "DocumentAgent": (0.52, 0.82, "Noisy OCR shows conflicting totals; possible tampering."),
            "PolicyAgent": (0.58, 0.7, "Policy aligns partially but ambiguity remains."),
            "AnomalyAgent": (0.78, 0.76, "Irregular textual and billing signals detected."),
            "PatternAgent": (0.73, 0.74, "Pattern resembles stealthy fraud behavior."),
            "GraphRiskAgent": (0.71, 0.71, "Entity graph shows moderate risk propagation."),
        },
        "edge": {
            "IdentityAgent": (0.4, 0.61, "Sparse identity details reduce confidence."),
            "DocumentAgent": (0.35, 0.6, "Insufficient document evidence for robust verification."),
            "PolicyAgent": (0.47, 0.63, "Policy signals are incomplete in this edge case."),
            "AnomalyAgent": (0.55, 0.59, "Edge-value claim amount elevates uncertainty."),
            "PatternAgent": (0.5, 0.58, "Limited data prevents strong pattern conclusion."),
            "GraphRiskAgent": (0.46, 0.57, "Graph context is inconclusive with sparse signals."),
        },
    }
    score, confidence, explanation = base.get(scenario, base["clean"]).get(
        role,
        (0.5, 0.5, "No specific reasoning available."),
    )
    if not has_docs and role in {"DocumentAgent", "PolicyAgent"}:
        confidence = max(0.1, confidence - 0.15)
        explanation = "Missing documents materially reduce verification certainty."
    if identity_invalid and role == "IdentityAgent":
        score = 0.05
        explanation = "Identity CIN format is invalid and mismatched."
    if memory_matches and role in {"PatternAgent", "GraphRiskAgent"}:
        score = min(1.0, score + 0.08)
        explanation = f"{explanation} Memory indicates {memory_matches} related fraud match(es)."
    if scenario == "adversarial" and role == "DocumentAgent":
        explanation = f"{explanation} Referenced CIN-999999 due to OCR conflict."

    reasoning_steps: List[str] = [
        "Parsed claim payload and role-specific inputs.",
        "Checked required context from blackboard entries.",
        "Scored risk indicators and calibrated confidence.",
    ]
    if scenario == "edge" and role == "PatternAgent":
        reasoning_steps = []

    return {
        "score": round(max(0.0, min(1.0, score)), 4),
        "confidence": round(max(0.0, min(1.0, confidence)), 4),
        "explanation": explanation,
        "reasoning_steps": reasoning_steps,
        "memory_insights": {
            "similar_cases_found": len(memory_context),
            "fraud_matches": memory_matches,
            "identity_reuse_detected": memory_matches > 0,
            "impact_on_score": "elevated risk" if memory_matches else "neutral",
            "notes": "simulated red-team memory reasoning",
        },
    }


class _SimCrew:
    calls: List[Dict[str, Any]] = []

    def __init__(self, tasks: List[_SimTask], **_: Any) -> None:
        self._task = tasks[0]

    def kickoff(self) -> str:
        role = self._task.agent.role
        claim = _extract_json("Claim: ", self._task.description)
        blackboard = _extract_json("Blackboard: ", self._task.description)
        _SimCrew.calls.append(
            {
                "role": role,
                "claim_id": str(claim.get("metadata", {}).get("claim_id", "")),
                "scenario": str(claim.get("metadata", {}).get("scenario", "")),
                "blackboard_entries": sorted(list((blackboard.get("entries") or {}).keys())),
            }
        )
        return json.dumps(_simulated_payload(role=role, claim=claim, blackboard=blackboard))


class ClaimGuardRedTeamEngine:
    """Runs end-to-end adversarial simulations and builds analytical reports."""

    def __init__(self, config: RedTeamConfig | None = None) -> None:
        self.config = config or RedTeamConfig()
        self._rng = random.Random(self.config.random_seed)
        self._artifact_dir = Path(self.config.artifact_dir)
        self._artifact_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        claims = self._generate_claims(self.config.claim_count)
        with_memory = self._execute_batch(claims=claims, use_memory=True)
        without_memory = self._execute_batch(claims=claims, use_memory=False)

        report = self._build_report(
            claims=claims,
            with_memory_results=with_memory,
            without_memory_results=without_memory,
        )
        if self.config.generate_visualizations:
            report["visualizations"] = self._build_visualizations(with_memory)
        strict = self._evaluate_strict_mode(report)
        report["strict_mode"] = strict
        report["run_status"] = "FAILED" if strict["failed"] else "PASSED"
        return report

    def _generate_claims(self, total: int) -> List[Dict[str, Any]]:
        per_scenario = max(10, total // len(_SCENARIOS))
        cases: List[Dict[str, Any]] = []
        repeated_cins = [f"CIN-{self._rng.randint(100000, 999999)}" for _ in range(5)]
        repeated_doctors = [f"DOC-{self._rng.randint(100, 999)}" for _ in range(3)]
        repeated_hospitals = [f"HSP-{self._rng.randint(10, 99)}" for _ in range(3)]

        claim_index = 0
        for scenario in _SCENARIOS:
            for i in range(per_scenario):
                claim_index += 1
                claim = self._build_scenario_claim(
                    scenario=scenario,
                    claim_id=f"rt-{scenario}-{claim_index:04d}",
                    i=i,
                    repeated_cins=repeated_cins,
                    repeated_doctors=repeated_doctors,
                    repeated_hospitals=repeated_hospitals,
                )
                cases.append(claim)

        while len(cases) < total:
            claim_index += 1
            scenario = self._rng.choice(_SCENARIOS)
            cases.append(
                self._build_scenario_claim(
                    scenario=scenario,
                    claim_id=f"rt-{scenario}-{claim_index:04d}",
                    i=claim_index,
                    repeated_cins=repeated_cins,
                    repeated_doctors=repeated_doctors,
                    repeated_hospitals=repeated_hospitals,
                )
            )
        self._rng.shuffle(cases)
        return cases[:total]

    def _build_scenario_claim(
        self,
        *,
        scenario: str,
        claim_id: str,
        i: int,
        repeated_cins: List[str],
        repeated_doctors: List[str],
        repeated_hospitals: List[str],
    ) -> Dict[str, Any]:
        amount = float(self._rng.randint(200, 25000))
        cin = f"CIN-{self._rng.randint(100000, 999999)}"
        hospital = f"HSP-{self._rng.randint(100, 999)}"
        doctor = f"DOC-{self._rng.randint(100, 999)}"
        document_text = "Medical report confirms diagnosis and billing."
        expected_decision = "APPROVED"
        expected_fraud = False
        identity_invalid = False
        obvious_fraud = False
        expected_alerts: List[str] = []
        has_minimal_data = False

        if scenario == "clean":
            expected_decision = "APPROVED"
        elif scenario == "suspicious":
            expected_decision = "HUMAN_REVIEW"
            expected_fraud = True
            amount = float(self._rng.randint(18000, 45000))
            if i % 2 == 0:
                document_text = "Invoice has subtle mismatch in treatment dates."
            else:
                document_text = "Provider coding pattern appears unusual."
            expected_alerts = ["contradictions"]
        elif scenario == "fraud":
            expected_decision = "REJECTED"
            expected_fraud = True
            obvious_fraud = True
            identity_invalid = i % 3 == 0
            cin = repeated_cins[i % len(repeated_cins)] if not identity_invalid else "INVALID-CIN-XX"
            hospital = repeated_hospitals[i % len(repeated_hospitals)]
            doctor = repeated_doctors[i % len(repeated_doctors)]
            amount = float(self._rng.randint(35000, 95000))
            document_text = "Altered invoice with identity mismatch and forged stamp."
            expected_alerts = ["high_risk_score", "fraud_ring"]
        elif scenario == "adversarial":
            expected_decision = "HUMAN_REVIEW"
            expected_fraud = True
            cin = repeated_cins[i % len(repeated_cins)] if i % 2 == 0 else cin
            hospital = repeated_hospitals[i % len(repeated_hospitals)] if i % 2 == 0 else hospital
            doctor = repeated_doctors[i % len(repeated_doctors)] if i % 2 == 0 else doctor
            document_text = (
                "OCR:: P4TI3NT N4M3 ?? ; invoice_total=19999 but handwritten total=9999. "
                "Field conflict with plausible diagnostics."
            )
            expected_alerts = ["contradictions"]
        elif scenario == "edge":
            has_minimal_data = i % 2 == 0
            expected_decision = "HUMAN_REVIEW"
            if has_minimal_data:
                document_text = ""
                amount = 0.0
            else:
                amount = 9999999.0
                document_text = "Duplicated fields; duplicated fields; duplicated fields."
            expected_alerts = ["high_risk_score"] if has_minimal_data else []

        documents = (
            [{"id": f"doc-{claim_id}", "document_type": "invoice", "text": document_text}]
            if document_text
            else []
        )
        if scenario == "suspicious" and i % 3 == 0 and documents:
            documents = documents[:1]
        if scenario == "edge" and not has_minimal_data and documents:
            documents.append(copy.deepcopy(documents[0]))

        identity = {
            "cin": cin,
            "name": f"Claimant-{claim_id}",
            "hospital": hospital,
            "doctor": doctor,
            "country": "ma",
        }
        if identity_invalid:
            identity["cin_status"] = "INVALID"

        claim = {
            "identity": identity,
            "documents": documents,
            "policy": {
                "amount": amount,
                "country": "ma",
                "hospital": hospital,
                "doctor": doctor,
                "diagnosis": "general-check",
            },
            "metadata": {
                "claim_id": claim_id,
                "scenario": scenario,
                "expected_decision": expected_decision,
                "expected_fraud": expected_fraud,
                "obvious_fraud": obvious_fraud,
                "identity_invalid": identity_invalid,
                "has_minimal_data": has_minimal_data,
                "anomaly_score": 0.9 if scenario in {"fraud", "adversarial"} else 0.25,
                "expected_alerts": expected_alerts,
            },
        }
        if scenario in {"fraud", "adversarial"}:
            claim["metadata"]["manual_review"] = True
        return claim

    def _execute_batch(self, *, claims: List[Dict[str, Any]], use_memory: bool) -> List[Dict[str, Any]]:
        from claimguard.v2 import fraud_ring_graph as graph_module

        graph_module._graph_singleton = graph_module.FraudRingGraph()
        run_store = self._artifact_dir / ("redteam_memory_on" if use_memory else "redteam_memory_off")
        if run_store.exists():
            shutil.rmtree(run_store)

        memory_layer = (
            CaseMemoryLayer(store_path=str(run_store), similarity_threshold=0.7)
            if use_memory
            else _NoMemoryLayer()
        )
        orchestrator = ClaimGuardV2Orchestrator(
            trust_layer_service=_NoOpTrustLayer(),
            memory_layer=memory_layer,  # type: ignore[arg-type]
        )
        traces: List[Dict[str, Any]] = []
        with self._orchestrator_agent_mode(simulated=self.config.use_simulated_agents):
            if self.config.use_simulated_agents:
                _SimCrew.calls.clear()
            for claim in claims:
                started = perf_counter()
                error: str | None = None
                response_dict: Dict[str, Any] = {}
                try:
                    response = orchestrator.run(copy.deepcopy(claim))
                    response_dict = response.model_dump()
                except Exception as exc:  # pragma: no cover - runtime protection
                    error = str(exc)
                total_ms = int((perf_counter() - started) * 1000)

                if error:
                    traces.append(
                        {
                            "claim_id": claim["metadata"]["claim_id"],
                            "scenario": claim["metadata"]["scenario"],
                            "expected_decision": claim["metadata"].get("expected_decision", "UNKNOWN"),
                            "expected_fraud": bool(claim["metadata"].get("expected_fraud", False)),
                            "use_memory": use_memory,
                            "error": error,
                            "total_pipeline_ms": total_ms,
                            "agent_outputs": [],
                            "blackboard_snapshots": [],
                            "Ts": 0.0,
                            "decision": "EXECUTION_ERROR",
                            "retry_count": 0,
                            "alerts": [],
                            "validation_failures": ["pipeline_execution_error"],
                        }
                    )
                    continue

                alerts = self._alerts_for_trace(claim, response_dict)
                failures = self._behavior_failures(claim, response_dict, alerts)
                blackboard_snapshots = self._build_blackboard_snapshots(response_dict)
                traces.append(
                    {
                        "claim_id": claim["metadata"]["claim_id"],
                        "scenario": claim["metadata"]["scenario"],
                        "expected_decision": claim["metadata"].get("expected_decision", "UNKNOWN"),
                        "expected_fraud": bool(claim["metadata"].get("expected_fraud", False)),
                        "use_memory": use_memory,
                        "error": None,
                        "total_pipeline_ms": total_ms,
                        "agent_outputs": response_dict.get("agent_outputs", []),
                        "blackboard_snapshots": blackboard_snapshots,
                        "Ts": float(response_dict.get("Ts", 0.0)),
                        "decision": str(response_dict.get("decision", "UNKNOWN")),
                        "retry_count": int(response_dict.get("retry_count", 0)),
                        "alerts": alerts,
                        "validation_failures": failures,
                        "raw_response": response_dict,
                    }
                )
            if self.config.use_simulated_agents:
                # Validate "agents ignore blackboard context" deterministically by
                # inspecting what each simulated agent actually received.
                traces_by_id = {t["claim_id"]: t for t in traces}
                self._validate_blackboard_context_simulated(traces_by_id)
        return traces

    @contextmanager
    def _orchestrator_agent_mode(self, *, simulated: bool):
        if not simulated:
            yield
            return
        from claimguard.v2 import orchestrator as orch_module

        original_agent = orch_module.Agent
        original_task = orch_module.Task
        original_crew = orch_module.Crew
        orch_module.Agent = _SimAgent
        orch_module.Task = _SimTask
        orch_module.Crew = _SimCrew
        try:
            yield
        finally:
            orch_module.Agent = original_agent
            orch_module.Task = original_task
            orch_module.Crew = original_crew

    def _build_blackboard_snapshots(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        final_entries = response.get("blackboard", {}).get("entries", {})
        snapshots: List[Dict[str, Any]] = []
        cumulative: Dict[str, Any] = {}
        for output in response.get("agent_outputs", []):
            agent_name = output.get("agent", "")
            if agent_name in final_entries:
                cumulative[agent_name] = final_entries[agent_name]
            snapshots.append(
                {
                    "stage": agent_name,
                    "entries": copy.deepcopy(cumulative),
                    "memory_context_count": len(response.get("memory_context", [])),
                }
            )
        snapshots.append(
            {
                "stage": "Consensus",
                "entries": copy.deepcopy(response.get("blackboard", {}).get("consensus_entries", {})),
                "Ts": float(response.get("Ts", 0.0)),
                "decision": response.get("decision", "UNKNOWN"),
                "retry_count": int(response.get("retry_count", 0)),
            }
        )
        return snapshots

    def _alerts_for_trace(self, claim: Dict[str, Any], response: Dict[str, Any]) -> List[Dict[str, Any]]:
        alerts: List[Dict[str, Any]] = []
        ts = float(response.get("Ts", 0.0))
        contradictions = response.get("contradictions", [])
        retry_count = int(response.get("retry_count", 0))
        fraud_rings = (
            response.get("blackboard", {})
            .get("fraud_ring_analysis", {})
            .get("fraud_rings", [])
        )
        if ts < 60.0:
            alerts.append({"type": "high_risk_score", "severity": "HIGH"})
        if contradictions:
            sev = "HIGH" if len(contradictions) > 2 else "MEDIUM"
            alerts.append({"type": "contradictions", "severity": sev})
        if retry_count >= 2:
            alerts.append({"type": "retry_loop", "severity": "MEDIUM"})
        if fraud_rings:
            alerts.append({"type": "fraud_ring", "severity": "HIGH"})
        if claim.get("metadata", {}).get("identity_invalid") and response.get("decision") != "REJECTED":
            alerts.append({"type": "identity_override", "severity": "HIGH"})
        return alerts

    def _behavior_failures(
        self,
        claim: Dict[str, Any],
        response: Dict[str, Any],
        alerts: List[Dict[str, Any]],
    ) -> List[str]:
        failures: List[str] = []
        ts = float(response.get("Ts", 0.0))
        contradictions = response.get("contradictions", [])
        decision = response.get("decision", "")
        retry_count = int(response.get("retry_count", 0))
        expected_alert_types = set(claim.get("metadata", {}).get("expected_alerts", []))
        observed_alert_types = {item["type"] for item in alerts}

        if claim.get("metadata", {}).get("has_minimal_data") and ts >= 99.99:
            failures.append("score_100_with_missing_data")

        if contradictions and any(float(o.get("confidence", 0.0)) > 0.8 for o in response.get("agent_outputs", [])):
            failures.append("high_confidence_with_contradictions")

        entries = response.get("blackboard", {}).get("entries", {})
        for output in response.get("agent_outputs", []):
            required_context = output.get("input_snapshot", {}).get("required_context", [])
            missing = [ctx for ctx in required_context if ctx not in entries]
            if missing:
                failures.append("agent_ignored_blackboard_context")
                break

        memory_context = response.get("memory_context", [])
        if any(float(item.get("similarity", 0.0)) < 0.7 for item in memory_context):
            failures.append("memory_similarity_threshold_violation")

        if claim.get("metadata", {}).get("identity_invalid") and decision != "REJECTED":
            failures.append("identity_invalid_pipeline_continued")

        if claim.get("metadata", {}).get("obvious_fraud") and decision == "APPROVED":
            failures.append("obvious_fraud_not_detected")

        if retry_count > 1:
            score_evolution = response.get("blackboard", {}).get("score_evolution", [])
            if score_evolution and max(score_evolution) <= score_evolution[0]:
                failures.append("reflexive_loop_no_improvement")

        if expected_alert_types and not expected_alert_types.issubset(observed_alert_types):
            failures.append("alerts_not_triggered")

        # Severity correctness (internal policy validation)
        severity_by_type = {a.get("type"): str(a.get("severity", "")).upper() for a in alerts}
        if ts < 60.0:
            if severity_by_type.get("high_risk_score") != "HIGH":
                failures.append("alert_severity_incorrect")
        if contradictions:
            expected_sev = "HIGH" if len(contradictions) > 2 else "MEDIUM"
            if severity_by_type.get("contradictions") != expected_sev:
                failures.append("alert_severity_incorrect")
        if retry_count >= 2:
            if severity_by_type.get("retry_loop") not in {"MEDIUM", "HIGH"}:
                failures.append("alert_severity_incorrect")
        if (
            response.get("blackboard", {})
            .get("fraud_ring_analysis", {})
            .get("fraud_rings", [])
        ):
            if severity_by_type.get("fraud_ring") != "HIGH":
                failures.append("alert_severity_incorrect")

        fraud_rings = (
            response.get("blackboard", {})
            .get("fraud_ring_analysis", {})
            .get("fraud_rings", [])
        )
        if claim.get("metadata", {}).get("obvious_fraud") and not fraud_rings:
            failures.append("fraud_cluster_not_flagged")
        return failures

    def _validate_blackboard_context_simulated(self, traces_by_id: Dict[str, Dict[str, Any]]) -> None:
        """
        In simulated-agent mode we can deterministically validate that each agent
        received (and thus could use) prior blackboard entries it depends on.
        """
        try:
            from claimguard.v2.orchestrator import SEQUENTIAL_AGENT_CONTRACTS
        except Exception:  # pragma: no cover
            return

        requires_by_agent = {c.name: set(c.requires) for c in SEQUENTIAL_AGENT_CONTRACTS}
        for call in _SimCrew.calls:
            claim_id = str(call.get("claim_id", ""))
            role = str(call.get("role", ""))
            required = requires_by_agent.get(role, set())
            provided = set(call.get("blackboard_entries") or [])
            if required and not required.issubset(provided):
                trace = traces_by_id.get(claim_id)
                if trace is not None:
                    trace["validation_failures"] = sorted(
                        set(trace.get("validation_failures", [])) | {"agent_ignored_blackboard_context"}
                    )

    def _build_report(
        self,
        *,
        claims: List[Dict[str, Any]],
        with_memory_results: List[Dict[str, Any]],
        without_memory_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        claim_index = {claim["metadata"]["claim_id"]: claim for claim in claims}
        summary = self._summary_block(with_memory_results)
        accuracy_metrics = self._accuracy_metrics(with_memory_results, claim_index)
        reasoning_metrics, agent_reasoning = self._reasoning_metrics(with_memory_results, claim_index)
        consistency = self._consistency_metrics(with_memory_results)
        alerts = self._alert_metrics(with_memory_results, claim_index)
        performance = self._performance_metrics(with_memory_results)
        graph_detection = self._graph_metrics(claims, with_memory_results)
        memory_impact = self._memory_impact(with_memory_results, without_memory_results, claim_index)
        weaknesses = self._system_weaknesses(with_memory_results)

        report = {
            "summary": summary,
            "accuracy_metrics": accuracy_metrics,
            "reasoning_metrics": reasoning_metrics,
            "consistency_analysis": consistency,
            "system_weaknesses": weaknesses,
            "agent_performance": agent_reasoning,
            "memory_impact": memory_impact,
            "graph_detection": graph_detection,
            "alerts": alerts,
            "performance_metrics": performance,
            "recommendations": self._recommendations(weaknesses, memory_impact, graph_detection, alerts),
            "run_metadata": {
                "claims_generated": len(claims),
                "random_seed": self.config.random_seed,
                "with_memory_runs": len(with_memory_results),
                "without_memory_runs": len(without_memory_results),
            },
            "sample_traces": with_memory_results[:3],
        }
        return report

    @staticmethod
    def _summary_block(results: List[Dict[str, Any]]) -> Dict[str, int]:
        total = len(results)
        failed = sum(1 for r in results if r.get("validation_failures"))
        critical_failures = sum(
            1
            for r in results
            if {
                "identity_invalid_pipeline_continued",
                "obvious_fraud_not_detected",
                "fraud_cluster_not_flagged",
                "pipeline_execution_error",
            }
            & set(r.get("validation_failures", []))
        )
        return {
            "total_tests": total,
            "passed": total - failed,
            "failed": failed,
            "critical_failures": critical_failures,
        }

    @staticmethod
    def _is_fraud_decision(decision: str) -> bool:
        return decision in {"HUMAN_REVIEW", "REJECTED", "REFLEXIVE_TRIGGER"}

    def _accuracy_metrics(self, results: List[Dict[str, Any]], claim_index: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        tp = fp = tn = fn = 0
        for row in results:
            claim = claim_index.get(row["claim_id"], {})
            actual_fraud = bool(claim.get("metadata", {}).get("expected_fraud", False))
            predicted_fraud = self._is_fraud_decision(str(row.get("decision", "")))
            if actual_fraud and predicted_fraud:
                tp += 1
            elif actual_fraud and not predicted_fraud:
                fn += 1
            elif not actual_fraud and predicted_fraud:
                fp += 1
            else:
                tn += 1
        fraud_detection_rate = (tp / (tp + fn)) if (tp + fn) else 0.0
        false_positive_rate = (fp / (fp + tn)) if (fp + tn) else 0.0
        false_negative_rate = (fn / (fn + tp)) if (fn + tp) else 0.0
        return {
            "fraud_detection_rate": round(fraud_detection_rate, 4),
            "false_positive_rate": round(false_positive_rate, 4),
            "false_negative_rate": round(false_negative_rate, 4),
        }

    def _reasoning_metrics(
        self,
        results: List[Dict[str, Any]],
        claim_index: Dict[str, Dict[str, Any]],
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
        hallucination_hits = 0
        confidence_values: List[float] = []
        false_confidence_hits = 0
        complete_reasoning_hits = 0
        total_outputs = 0
        per_agent: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "runs": 0,
            "avg_latency_ms": 0.0,
            "avg_confidence": 0.0,
            "hallucination_rate": 0.0,
            "reasoning_completeness_score": 0.0,
        })
        agent_latencies: Dict[str, List[int]] = defaultdict(list)
        agent_confidences: Dict[str, List[float]] = defaultdict(list)
        agent_hallucinations: Dict[str, int] = defaultdict(int)
        agent_complete: Dict[str, int] = defaultdict(int)

        for row in results:
            claim = claim_index.get(row["claim_id"], {})
            for output in row.get("agent_outputs", []):
                total_outputs += 1
                agent = str(output.get("agent", "unknown"))
                confidence = float(output.get("confidence", 0.0))
                explanation = str(output.get("explanation", ""))
                snapshot = output.get("output_snapshot", {})

                confidence_values.append(confidence)
                agent_latencies[agent].append(int(output.get("elapsed_ms", 0)))
                agent_confidences[agent].append(confidence)

                reasoning_steps = snapshot.get("reasoning_steps")
                has_reasoning_steps = isinstance(reasoning_steps, list) and len(reasoning_steps) > 0
                if has_reasoning_steps and explanation.strip():
                    complete_reasoning_hits += 1
                    agent_complete[agent] += 1

                hallucinated = self._is_hallucinated(claim, row, explanation)
                mismatch = self._reasoning_mismatch(claim, explanation)
                if hallucinated:
                    hallucination_hits += 1
                    agent_hallucinations[agent] += 1
                if mismatch and confidence > 0.8:
                    false_confidence_hits += 1

        for agent in sorted(set(agent_latencies) | set(agent_confidences)):
            runs = max(1, len(agent_confidences.get(agent, [])))
            per_agent[agent] = {
                "runs": runs,
                "avg_latency_ms": round(mean(agent_latencies.get(agent, [0])), 2),
                "avg_confidence": round(mean(agent_confidences.get(agent, [0.0])), 4),
                "hallucination_rate": round(agent_hallucinations.get(agent, 0) / runs, 4),
                "reasoning_completeness_score": round(agent_complete.get(agent, 0) / runs, 4),
            }

        total_outputs_nonzero = max(1, total_outputs)
        return (
            {
                "hallucination_rate": round(hallucination_hits / total_outputs_nonzero, 4),
                "avg_confidence": round(mean(confidence_values), 4) if confidence_values else 0.0,
                "false_confidence_rate": round(false_confidence_hits / total_outputs_nonzero, 4),
                "reasoning_completeness_score": round(complete_reasoning_hits / total_outputs_nonzero, 4),
            },
            per_agent,
        )

    def _is_hallucinated(self, claim: Dict[str, Any], trace: Dict[str, Any], explanation: str) -> bool:
        known_tokens = set()
        cin = str(claim.get("identity", {}).get("cin", "")).upper()
        if cin:
            known_tokens.add(cin)
        for mem in trace.get("raw_response", {}).get("memory_context", []):
            value = str(mem.get("cin", "")).upper()
            if value:
                known_tokens.add(value)
        tokens = {token.upper() for token in _CIN_TOKEN_RE.findall(explanation)}
        if tokens and not tokens.issubset(known_tokens):
            return True
        if "not present in records" in explanation.lower() and claim.get("documents"):
            return True
        return False

    @staticmethod
    def _reasoning_mismatch(claim: Dict[str, Any], explanation: str) -> bool:
        text = explanation.lower()
        docs_missing = not bool(claim.get("documents"))
        if "missing document" in text and not docs_missing:
            return True
        if "identity mismatch" in text and not claim.get("metadata", {}).get("identity_invalid"):
            return True
        return False

    @staticmethod
    def _consistency_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
        score_std: List[float] = []
        contradiction_count = 0
        ts_with_contradictions: List[float] = []
        ts_without_contradictions: List[float] = []
        for row in results:
            scores = [float(o.get("score", 0.0)) for o in row.get("agent_outputs", [])]
            if scores:
                avg = sum(scores) / len(scores)
                variance = sum((s - avg) ** 2 for s in scores) / len(scores)
                score_std.append(variance ** 0.5)
            contradictions = row.get("raw_response", {}).get("contradictions", [])
            contradiction_count += len(contradictions)
            if contradictions:
                ts_with_contradictions.append(float(row.get("Ts", 0.0)))
            else:
                ts_without_contradictions.append(float(row.get("Ts", 0.0)))
        inter_agent_agreement = 1.0 - min(1.0, (mean(score_std) if score_std else 0.0))
        contradiction_frequency = contradiction_count / max(1, len(results))
        impact = 0.0
        if ts_with_contradictions and ts_without_contradictions:
            impact = mean(ts_without_contradictions) - mean(ts_with_contradictions)
        return {
            "inter_agent_agreement": round(inter_agent_agreement, 4),
            "contradiction_frequency": round(contradiction_frequency, 4),
            "contradiction_impact_on_ts": round(impact, 4),
        }

    @staticmethod
    def _alert_metrics(results: List[Dict[str, Any]], claim_index: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
        triggered = missed = false_alerts = 0
        for row in results:
            claim = claim_index.get(row["claim_id"], {})
            expected = set(claim.get("metadata", {}).get("expected_alerts", []))
            observed = {item["type"] for item in row.get("alerts", [])}
            triggered += len(observed)
            missed += len(expected - observed)
            false_alerts += len(observed - expected)
        return {
            "triggered": triggered,
            "missed": missed,
            "false_alerts": false_alerts,
        }

    @staticmethod
    def _performance_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        per_agent: Dict[str, List[int]] = defaultdict(list)
        total_pipeline = [int(row.get("total_pipeline_ms", 0)) for row in results]
        retry_latency = defaultdict(list)
        for row in results:
            retry = int(row.get("retry_count", 0))
            retry_latency[retry].append(int(row.get("total_pipeline_ms", 0)))
            for output in row.get("agent_outputs", []):
                per_agent[str(output.get("agent", "unknown"))].append(int(output.get("elapsed_ms", 0)))

        retry_0 = mean(retry_latency.get(0, [0]))
        retry_1_plus = mean([v for r, vals in retry_latency.items() if r >= 1 for v in vals] or [0])
        retry_overhead = max(0.0, retry_1_plus - retry_0)
        return {
            "latency_per_agent_ms": {k: round(mean(v), 2) for k, v in per_agent.items()},
            "avg_total_pipeline_ms": round(mean(total_pipeline), 2) if total_pipeline else 0.0,
            "max_total_pipeline_ms": max(total_pipeline) if total_pipeline else 0,
            "retry_overhead_ms": round(retry_overhead, 2),
        }

    @staticmethod
    def _graph_metrics(claims: List[Dict[str, Any]], results: List[Dict[str, Any]]) -> Dict[str, int]:
        repeated_cin_counts = Counter(
            claim.get("identity", {}).get("cin", "")
            for claim in claims
            if claim.get("identity", {}).get("cin")
        )
        expected_cluster_cins = {cin for cin, count in repeated_cin_counts.items() if count >= 3}
        detected_claims = set()
        for row in results:
            rings = (
                row.get("raw_response", {})
                .get("blackboard", {})
                .get("fraud_ring_analysis", {})
                .get("fraud_rings", [])
            )
            for ring in rings:
                for claim_id in ring.get("claims", []):
                    detected_claims.add(claim_id)

        expected_cluster_claims = {
            claim["metadata"]["claim_id"]
            for claim in claims
            if claim.get("identity", {}).get("cin", "") in expected_cluster_cins
        }
        missed = len(expected_cluster_claims - detected_claims)
        return {
            "clusters_detected": len(detected_claims),
            "missed_clusters": missed,
        }

    def _memory_impact(
        self,
        with_memory_results: List[Dict[str, Any]],
        without_memory_results: List[Dict[str, Any]],
        claim_index: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        with_metrics = self._accuracy_metrics(with_memory_results, claim_index)
        without_metrics = self._accuracy_metrics(without_memory_results, claim_index)
        improvement = with_metrics["fraud_detection_rate"] - without_metrics["fraud_detection_rate"]
        fp_delta = with_metrics["false_positive_rate"] - without_metrics["false_positive_rate"]
        risks = []
        if fp_delta > 0:
            risks.append("Memory increased false-positive rate.")
        if improvement < 0:
            risks.append("Memory reduced fraud detection performance.")
        if not risks:
            risks.append("No major memory risk observed in this run.")
        return {
            "improvement": round(improvement, 4),
            "fraud_detection_with_memory": with_metrics["fraud_detection_rate"],
            "fraud_detection_without_memory": without_metrics["fraud_detection_rate"],
            "false_positive_delta": round(fp_delta, 4),
            "risks": risks,
        }

    @staticmethod
    def _system_weaknesses(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        counter = Counter()
        for row in results:
            counter.update(row.get("validation_failures", []))
        weaknesses = []
        for issue, freq in counter.most_common(8):
            if freq >= 10:
                impact = "HIGH"
            elif freq >= 4:
                impact = "MEDIUM"
            else:
                impact = "LOW"
            weaknesses.append({"issue": issue, "frequency": freq, "impact": impact})
        return weaknesses

    @staticmethod
    def _recommendations(
        weaknesses: List[Dict[str, Any]],
        memory_impact: Dict[str, Any],
        graph_detection: Dict[str, Any],
        alert_metrics: Dict[str, int],
    ) -> List[str]:
        recs: List[str] = []
        weakness_names = {w["issue"] for w in weaknesses}
        if "high_confidence_with_contradictions" in weakness_names:
            recs.append("Add confidence calibration penalties when contradiction count > 0.")
        if "identity_invalid_pipeline_continued" in weakness_names:
            recs.append("Hard-stop pipeline when identity validity checks fail early.")
        if "alerts_not_triggered" in weakness_names:
            recs.append("Introduce central alert policy to enforce mandatory threshold alerts.")
        if graph_detection.get("missed_clusters", 0) > 0:
            recs.append("Lower graph ring anomaly threshold or increase ring sensitivity for repeated CIN clusters.")
        if memory_impact.get("false_positive_delta", 0.0) > 0.02:
            recs.append("Tune memory similarity threshold and add case-quality weighting to reduce memory false positives.")
        if alert_metrics.get("false_alerts", 0) > alert_metrics.get("triggered", 0) * 0.3:
            recs.append("Refine alert severity mapping to reduce noisy alerts.")
        if not recs:
            recs.append("No urgent systemic weakness detected; keep monitoring with larger adversarial batches.")
        return recs

    def _evaluate_strict_mode(self, report: Dict[str, Any]) -> Dict[str, Any]:
        failures: List[str] = []
        strict = self.config.strict_mode
        critical_failures = int(report["summary"]["critical_failures"])
        hallucination_rate = float(report["reasoning_metrics"]["hallucination_rate"])
        fraud_detection = float(report["accuracy_metrics"]["fraud_detection_rate"])
        if critical_failures > strict.critical_failures_threshold:
            failures.append("critical_failures_exceeded")
        if hallucination_rate > strict.hallucination_rate_max:
            failures.append("hallucination_rate_exceeded")
        if fraud_detection < strict.fraud_detection_target:
            failures.append("fraud_detection_below_target")
        return {
            "failed": bool(failures),
            "failure_reasons": failures,
            "thresholds": {
                "critical_failures_threshold": strict.critical_failures_threshold,
                "hallucination_rate_max": strict.hallucination_rate_max,
                "fraud_detection_target": strict.fraud_detection_target,
            },
        }

    def _build_visualizations(self, results: List[Dict[str, Any]]) -> Dict[str, str]:
        paths: Dict[str, str] = {}
        try:
            import matplotlib.pyplot as plt

            ts_values = [float(row.get("Ts", 0.0)) for row in results]
            decisions = [row.get("decision", "UNKNOWN") for row in results]
            confidence_pairs: List[Tuple[float, bool]] = []
            disagreements: List[int] = []
            for row in results:
                outputs = row.get("agent_outputs", [])
                if outputs:
                    avg_conf = mean(float(o.get("confidence", 0.0)) for o in outputs)
                    confidence_pairs.append((avg_conf, row.get("decision") == row.get("expected_decision")))
                contradictions = row.get("raw_response", {}).get("contradictions", [])
                disagreements.append(len(contradictions))

            ts_path = self._artifact_dir / "redteam_ts_distribution.png"
            plt.figure(figsize=(8, 5))
            plt.hist(ts_values, bins=15, color="#2672ec", alpha=0.85)
            plt.title("Ts Score Distribution")
            plt.xlabel("Ts Score")
            plt.ylabel("Claims")
            plt.tight_layout()
            plt.savefig(ts_path)
            plt.close()
            paths["ts_score_distribution"] = str(ts_path)

            conf_path = self._artifact_dir / "redteam_confidence_vs_correctness.png"
            plt.figure(figsize=(8, 5))
            xs = [pair[0] for pair in confidence_pairs]
            ys = [1 if pair[1] else 0 for pair in confidence_pairs]
            plt.scatter(xs, ys, alpha=0.7, c=ys, cmap="coolwarm")
            plt.title("Confidence vs Correctness")
            plt.xlabel("Average Agent Confidence")
            plt.ylabel("Correct (1=yes, 0=no)")
            plt.tight_layout()
            plt.savefig(conf_path)
            plt.close()
            paths["confidence_vs_correctness"] = str(conf_path)

            disagree_path = self._artifact_dir / "redteam_agent_disagreements.png"
            plt.figure(figsize=(8, 5))
            plt.bar(range(len(disagreements)), disagreements, color="#f39c12")
            plt.title("Agent Disagreements per Claim")
            plt.xlabel("Claim Index")
            plt.ylabel("Contradiction Count")
            plt.tight_layout()
            plt.savefig(disagree_path)
            plt.close()
            paths["agent_disagreements"] = str(disagree_path)

            decision_counter = Counter(decisions)
            decision_path = self._artifact_dir / "redteam_decision_breakdown.png"
            plt.figure(figsize=(7, 5))
            labels = list(decision_counter.keys())
            values = [decision_counter[k] for k in labels]
            plt.bar(labels, values, color="#27ae60")
            plt.title("Decision Breakdown")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(decision_path)
            plt.close()
            paths["decision_breakdown"] = str(decision_path)
        except Exception:  # pragma: no cover - plotting is optional
            return {}
        return paths


def run_red_teaming(
    *,
    claim_count: int = 100,
    random_seed: int = 42,
    artifact_dir: str = "tests/artifacts",
    generate_visualizations: bool = True,
    use_simulated_agents: bool = False,
    strict_mode: StrictModeConfig | None = None,
) -> Dict[str, Any]:
    config = RedTeamConfig(
        claim_count=claim_count,
        random_seed=random_seed,
        strict_mode=strict_mode or StrictModeConfig(),
        artifact_dir=Path(artifact_dir),
        generate_visualizations=generate_visualizations,
        use_simulated_agents=use_simulated_agents,
    )
    return ClaimGuardRedTeamEngine(config).run()


def write_redteam_report(report: Dict[str, Any], artifact_dir: str = "tests/artifacts") -> str:
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "redteam_report.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return str(output_path)
