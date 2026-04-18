import sys
from pathlib import Path

from fastapi.testclient import TestClient

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from claimguard.main import create_app


def test_v2_debug_fraud_graph_endpoint_returns_shape(monkeypatch) -> None:
    from claimguard.v2 import orchestrator as orchestrator_module

    class FakeOrchestrator:
        def get_fraud_graph_debug(self, *, render_png: bool = False):
            return {
                "fraud_rings": [
                    {
                        "cluster_id": "cluster-1",
                        "nodes": ["claim::1", "cin::A", "hospital::H"],
                        "claims": ["1", "2", "3"],
                        "risk_score": 0.91,
                        "reason": "cluster has 3 claims; average anomaly score 0.91; shared hospital",
                    }
                ],
                "node_count": 12,
                "edge_count": 18,
                "png_path": "tests/artifacts/v2_fraud_graph.png" if render_png else None,
            }

    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("DOCUMENT_ENCRYPTION_KEY", "0" * 32)
    monkeypatch.setenv("CLAIMAGUARD_API_KEYS", "test-api-key-for-ci")
    monkeypatch.setattr(orchestrator_module, "_singleton", FakeOrchestrator())

    client = TestClient(create_app())
    response = client.get(
        "/v2/debug/fraud-graph",
        headers={"x-api-key": "test-api-key-for-ci"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert "fraud_rings" in payload
    assert "node_count" in payload
    assert "edge_count" in payload
    assert "png_path" in payload
    assert isinstance(payload["fraud_rings"], list)
    assert payload["png_path"] is None

    response_png = client.get(
        "/v2/debug/fraud-graph?render_png=true",
        headers={"x-api-key": "test-api-key-for-ci"},
    )
    assert response_png.status_code == 200
    payload_png = response_png.json()
    assert payload_png["png_path"] is not None
