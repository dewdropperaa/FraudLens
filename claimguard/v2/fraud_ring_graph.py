from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import networkx as nx


@dataclass
class ClaimNodeContext:
    claim_node: str
    cin_node: str
    hospital_node: str
    doctor_node: str
    anomaly_score: float


class FraudRingGraph:
    """In-memory graph for coordinated fraud-ring detection in v2."""

    def __init__(self) -> None:
        self.graph = nx.Graph()

    @staticmethod
    def _safe_str(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @staticmethod
    def _safe_score(value: Any) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            score = 0.0
        return max(0.0, min(1.0, score))

    def _make_context(
        self,
        *,
        claim_id: str,
        cin: str,
        hospital: str,
        doctor: str,
        anomaly_score: float,
    ) -> ClaimNodeContext:
        return ClaimNodeContext(
            claim_node=f"claim::{claim_id}",
            cin_node=f"cin::{cin}",
            hospital_node=f"hospital::{hospital}",
            doctor_node=f"doctor::{doctor}",
            anomaly_score=self._safe_score(anomaly_score),
        )

    def add_claim(
        self,
        *,
        claim_id: Any,
        cin: Any,
        hospital: Any,
        doctor: Any,
        anomaly_score: Any,
    ) -> Dict[str, Any]:
        claim_id_s = self._safe_str(claim_id)
        cin_s = self._safe_str(cin)
        hospital_s = self._safe_str(hospital)
        doctor_s = self._safe_str(doctor)
        if not all([claim_id_s, cin_s, hospital_s, doctor_s]):
            raise ValueError("Fraud ring graph requires non-empty claim_id, cin, hospital, and doctor.")

        ctx = self._make_context(
            claim_id=claim_id_s,
            cin=cin_s,
            hospital=hospital_s,
            doctor=doctor_s,
            anomaly_score=anomaly_score,
        )

        self.graph.add_node(ctx.claim_node, node_type="claim", entity_id=claim_id_s, anomaly_score=ctx.anomaly_score)
        self.graph.add_node(ctx.cin_node, node_type="cin", entity_id=cin_s)
        self.graph.add_node(ctx.hospital_node, node_type="hospital", entity_id=hospital_s)
        self.graph.add_node(ctx.doctor_node, node_type="doctor", entity_id=doctor_s)

        self.graph.add_edge(ctx.cin_node, ctx.claim_node, edge_type="cin_claim")
        self.graph.add_edge(ctx.claim_node, ctx.hospital_node, edge_type="claim_hospital")
        self.graph.add_edge(ctx.claim_node, ctx.doctor_node, edge_type="claim_doctor")
        return self.analyze_claim(claim_id_s)

    def _connected_components(self) -> List[Set[str]]:
        return list(nx.connected_components(self.graph))

    def _claims_in_component(self, component: Set[str]) -> List[str]:
        out: List[str] = []
        for node in component:
            attrs = self.graph.nodes[node]
            if attrs.get("node_type") == "claim":
                out.append(str(attrs.get("entity_id", "")))
        return sorted([c for c in out if c])

    def _shared_entity(
        self,
        component: Set[str],
        *,
        entity_type: str,
    ) -> bool:
        ids = {
            str(self.graph.nodes[node].get("entity_id", ""))
            for node in component
            if self.graph.nodes[node].get("node_type") == entity_type
        }
        ids = {x for x in ids if x}
        return len(ids) >= 1

    def _avg_claim_anomaly(self, component: Set[str]) -> float:
        scores: List[float] = []
        for node in component:
            attrs = self.graph.nodes[node]
            if attrs.get("node_type") == "claim":
                scores.append(self._safe_score(attrs.get("anomaly_score", 0.0)))
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def _node_is_high_degree(self, node: str) -> bool:
        typed_nodes = [n for n, a in self.graph.nodes(data=True) if a.get("node_type") != "claim"]
        if len(typed_nodes) < 4 or node not in self.graph:
            return False
        degree_values = [self.graph.degree(n) for n in typed_nodes]
        degree_values_sorted = sorted(degree_values)
        threshold_idx = max(0, int(len(degree_values_sorted) * 0.9) - 1)
        threshold = degree_values_sorted[threshold_idx]
        return self.graph.degree(node) >= threshold and self.graph.degree(node) >= 2

    def _reuse_detection(self, claim_id: str) -> Dict[str, Any]:
        claim_node = f"claim::{claim_id}"
        if claim_node not in self.graph:
            return {
                "cin_reuse_detected": False,
                "doctor_reuse_detected": False,
                "cin_claim_count": 0,
                "doctor_claim_count": 0,
            }

        cin_neighbors = [
            nb for nb in self.graph.neighbors(claim_node) if self.graph.nodes[nb].get("node_type") == "cin"
        ]
        doctor_neighbors = [
            nb for nb in self.graph.neighbors(claim_node) if self.graph.nodes[nb].get("node_type") == "doctor"
        ]
        cin_count = 0
        doctor_count = 0
        cin_reuse = False
        doctor_reuse = False

        if cin_neighbors:
            cin_node = cin_neighbors[0]
            cin_count = sum(
                1 for nb in self.graph.neighbors(cin_node) if self.graph.nodes[nb].get("node_type") == "claim"
            )
            cin_reuse = cin_count > 1
        if doctor_neighbors:
            doctor_node = doctor_neighbors[0]
            suspicious_claim_neighbors = [
                nb
                for nb in self.graph.neighbors(doctor_node)
                if self.graph.nodes[nb].get("node_type") == "claim"
                and self._safe_score(self.graph.nodes[nb].get("anomaly_score", 0.0)) >= 0.6
            ]
            doctor_count = len(suspicious_claim_neighbors)
            doctor_reuse = doctor_count > 1

        return {
            "cin_reuse_detected": cin_reuse,
            "doctor_reuse_detected": doctor_reuse,
            "cin_claim_count": cin_count,
            "doctor_claim_count": doctor_count,
        }

    def detect_fraud_rings(self, anomaly_threshold: float = 0.6, min_claims: int = 3) -> Dict[str, Any]:
        fraud_rings: List[Dict[str, Any]] = []
        for idx, component in enumerate(self._connected_components()):
            claims = self._claims_in_component(component)
            if len(claims) < min_claims:
                continue

            avg_anomaly = self._avg_claim_anomaly(component)
            has_shared_hospital = self._shared_entity(component, entity_type="hospital")
            has_shared_doctor = self._shared_entity(component, entity_type="doctor")
            if not (has_shared_hospital or has_shared_doctor):
                continue
            if avg_anomaly <= anomaly_threshold:
                continue

            high_degree_nodes = [
                node for node in component if self.graph.nodes[node].get("node_type") != "claim" and self._node_is_high_degree(node)
            ]
            reuse_hits = 0
            for claim_id in claims:
                reuse = self._reuse_detection(claim_id)
                if reuse["cin_reuse_detected"] or reuse["doctor_reuse_detected"]:
                    reuse_hits += 1

            normalized_claim_size = min(1.0, len(claims) / 10.0)
            high_degree_factor = min(1.0, len(high_degree_nodes) / 4.0)
            reuse_factor = min(1.0, reuse_hits / max(1, len(claims)))
            risk_score = round(
                (avg_anomaly * 0.55) + (normalized_claim_size * 0.2) + (high_degree_factor * 0.15) + (reuse_factor * 0.1),
                4,
            )
            reason_parts = [
                f"cluster has {len(claims)} claims",
                f"average anomaly score {avg_anomaly:.3f}",
                "shared hospital" if has_shared_hospital else "shared doctor",
            ]
            if high_degree_nodes:
                reason_parts.append("high-degree entities detected")
            if reuse_hits:
                reason_parts.append("CIN/doctor reuse detected")

            fraud_rings.append(
                {
                    "cluster_id": f"cluster-{idx}",
                    "nodes": sorted(component),
                    "claims": claims,
                    "risk_score": risk_score,
                    "reason": "; ".join(reason_parts),
                }
            )

        fraud_rings.sort(key=lambda item: item["risk_score"], reverse=True)
        return {"fraud_rings": fraud_rings}

    def analyze_claim(self, claim_id: str) -> Dict[str, Any]:
        claim_node = f"claim::{claim_id}"
        if claim_node not in self.graph:
            return {
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

        rings = self.detect_fraud_rings()
        matching_ring: Optional[Dict[str, Any]] = None
        for ring in rings["fraud_rings"]:
            if claim_id in ring["claims"]:
                matching_ring = ring
                break

        reuse = self._reuse_detection(claim_id)
        if matching_ring is not None:
            cluster_membership = matching_ring["cluster_id"]
            network_risk_score = float(matching_ring["risk_score"])
        else:
            cluster_membership = "none"
            anomaly_score = self._safe_score(self.graph.nodes[claim_node].get("anomaly_score", 0.0))
            reuse_boost = 0.15 if (reuse["cin_reuse_detected"] or reuse["doctor_reuse_detected"]) else 0.0
            network_risk_score = round(min(1.0, anomaly_score + reuse_boost), 4)

        return {
            "cluster_membership": cluster_membership,
            "reuse_detection": reuse,
            "network_risk_score": network_risk_score,
            "fraud_rings": rings,
        }

    def visualize_graph(self, output_path: str | Path) -> str:
        # Optional plotting path for demo usage only.
        import matplotlib.pyplot as plt

        rings = self.detect_fraud_rings()
        suspicious_nodes: Set[str] = set()
        for ring in rings["fraud_rings"]:
            suspicious_nodes.update(ring["nodes"])

        node_colors = ["red" if node in suspicious_nodes else "blue" for node in self.graph.nodes()]
        pos = nx.spring_layout(self.graph, seed=42)
        plt.figure(figsize=(10, 8))
        nx.draw(
            self.graph,
            pos,
            node_color=node_colors,
            node_size=450,
            font_size=7,
            with_labels=True,
            alpha=0.9,
        )
        plt.title("ClaimGuard v2 Fraud Ring Graph (red=suspicious, blue=normal)")
        out = str(output_path)
        plt.savefig(out)
        plt.close()
        return out


_graph_singleton: Optional[FraudRingGraph] = None


def get_fraud_ring_graph() -> FraudRingGraph:
    global _graph_singleton
    if _graph_singleton is None:
        _graph_singleton = FraudRingGraph()
    return _graph_singleton
