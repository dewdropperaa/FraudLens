import base64
import json

import pytest
from fastapi.testclient import TestClient

from claimguard.main import app

API_HEADERS = {"X-API-Key": "test-api-key-for-ci"}

client = TestClient(app)

EXPECTED_AGENT_NAMES = (
    "Anomaly Agent",
    "Pattern Agent",
    "Identity Agent",
    "Document Agent",
    "Policy Agent",
    "Graph Agent",
)


def _assert_common_response_shape(data: dict) -> None:
    assert "claim_id" in data
    assert data["decision"] in ["APPROVED", "REJECTED"]
    assert isinstance(data["score"], (int, float))
    assert 0 <= float(data["score"]) <= 100
    assert isinstance(data["agent_results"], list)
    assert len(data["agent_results"]) == 6
    names = [ar["agent_name"] for ar in data["agent_results"]]
    assert names == list(EXPECTED_AGENT_NAMES)
    assert "tx_hash" in data
    assert "ipfs_hash" in data
    assert "ipfs_hashes" in data

    # Ensure each agent result is structured JSON.
    for ar in data["agent_results"]:
        assert "agent_name" in ar
        assert ar["decision"] in [True, False]
        assert isinstance(ar["score"], (int, float))
        assert 0 <= float(ar["score"]) <= 100
        assert isinstance(ar["reasoning"], str)
        assert isinstance(ar.get("details", {}), dict)


def _post_claim(claim_data: dict) -> dict:
    response = client.post("/claim", json=claim_data, headers=API_HEADERS)
    assert response.status_code == 200
    data = response.json()
    _assert_common_response_shape(data)
    return data


def test_1_valid_claim_cnss_approved_and_get_works():
    claim_data = {
        "patient_id": "12345678",
        "provider_id": "3037",
        "amount": 5000.0,
        "documents": ["medical_report", "invoice", "prescription"],
        "history": [
            {"amount": 2000.0, "date": "2024-01-15", "recent": False},
            {"amount": 3000.0, "date": "2024-02-20", "recent": False},
        ],
        "insurance": "CNSS",
    }

    posted = _post_claim(claim_data)
    assert posted["decision"] == "APPROVED"
    assert float(posted["score"]) >= 75
    assert isinstance(posted["ipfs_hashes"], list)
    assert len(posted["ipfs_hashes"]) >= 1
    # Blockchain tx is optional unless Sepolia env is configured.
    assert posted["tx_hash"] is None or isinstance(posted["tx_hash"], str)

    get_response = client.get(f"/claim/{posted['claim_id']}", headers=API_HEADERS)
    assert get_response.status_code == 200
    fetched = get_response.json()
    _assert_common_response_shape(fetched)
    assert fetched["claim_id"] == posted["claim_id"]
    assert fetched["decision"] == posted["decision"]


def test_2_valid_claim_cnops_approved():
    claim_data = {
        "patient_id": "87654321",
        "provider_id": "1543",
        "amount": 10000.0,
        "documents": ["medical_report", "invoice", "prescription", "lab_results"],
        "history": [
            {"amount": 5000.0, "date": "2024-01-10", "recent": False},
            {"amount": 4000.0, "date": "2024-03-05", "recent": False},
        ],
        "insurance": "CNOPS",
    }

    posted = _post_claim(claim_data)
    assert posted["decision"] == "APPROVED"
    assert float(posted["score"]) >= 75
    assert len(posted["ipfs_hashes"]) >= 1


def test_3_fraud_high_amount_rejected():
    claim_data = {
        "patient_id": "12345678",
        "provider_id": "3847",
        "amount": 50000.0,
        "documents": ["medical_report", "invoice"],
        "history": [
            {"amount": 2000.0, "date": "2024-01-15", "recent": False},
            {"amount": 2500.0, "date": "2024-02-20", "recent": False},
        ],
        "insurance": "CNSS",
    }

    posted = _post_claim(claim_data)
    assert posted["decision"] == "REJECTED"
    assert float(posted["score"]) < 75


def test_4_fraud_invalid_patient_id_rejected():
    claim_data = {
        "patient_id": "ABC123",
        "provider_id": "3911",
        "amount": 5000.0,
        "documents": ["medical_report", "invoice"],
        "history": [
            {"amount": 2000.0, "date": "2024-01-15", "recent": False},
        ],
        "insurance": "CNSS",
    }

    posted = _post_claim(claim_data)
    assert posted["decision"] == "REJECTED"


def test_5_fraud_insufficient_docs_rejected():
    claim_data = {
        "patient_id": "12345678",
        "provider_id": "3825",
        "amount": 8000.0,
        "documents": [],
        "history": [
            {"amount": 2000.0, "date": "2024-01-15", "recent": False},
        ],
        "insurance": "CNOPS",
    }

    posted = _post_claim(claim_data)
    assert posted["decision"] == "REJECTED"


def test_6_fraud_multiple_recent_claims_rejected():
    claim_data = {
        "patient_id": "12345678",
        "provider_id": "2462",
        "amount": 6000.0,
        "documents": ["medical_report", "invoice", "prescription"],
        "history": [
            {"amount": 2000.0, "date": "2024-01-15", "recent": True},
            {"amount": 3000.0, "date": "2024-01-20", "recent": True},
            {"amount": 4000.0, "date": "2024-01-25", "recent": True},
            {"amount": 5000.0, "date": "2024-02-01", "recent": True},
        ],
        "insurance": "CNSS",
    }

    posted = _post_claim(claim_data)
    assert posted["decision"] == "REJECTED"


def test_7_get_nonexistent_claim_404():
    response = client.get("/claim/nonexistent-claim-id", headers=API_HEADERS)
    assert response.status_code == 404


def test_8_invalid_amount_validation_422():
    # `amount` must be > 0 (Pydantic validation).
    claim_data = {
        "patient_id": "12345678",
        "provider_id": "111",
        "amount": 0.0,
        "documents": ["medical_report"],
        "history": [],
        "insurance": "CNSS",
    }

    response = client.post("/claim", json=claim_data, headers=API_HEADERS)
    assert response.status_code == 422


def test_9_missing_provider_id_validation_422():
    claim_data = {
        "patient_id": "12345678",
        "amount": 1000.0,
        "documents": [],
        "history": [],
        "insurance": "CNSS",
    }
    response = client.post("/claim", json=claim_data, headers=API_HEADERS)
    assert response.status_code == 422


def test_10_graph_agent_uses_real_claim_and_provider_nodes():
    """Graph details must reference resolved claim_id and provided provider_id (no placeholders)."""
    claim_data = {
        "patient_id": "33680",
        "provider_id": "3847",
        "claim_id": "e2e-graph-claim-001",
        "amount": 12000.0,
        "documents": ["medical_report", "invoice", "prescription"],
        "history": [{"amount": 2000.0, "date": "2024-01-15", "recent": False}],
        "insurance": "CNSS",
    }
    posted = _post_claim(claim_data)
    assert posted["claim_id"] == "e2e-graph-claim-001"
    graph = next(ar for ar in posted["agent_results"] if ar["agent_name"] == "Graph Agent")
    details = graph["details"]
    risk = details["risk_nodes"]
    assert f"claim::{posted['claim_id']}" in risk
    assert "provider::3847" in risk
    assert "patient::33680" in risk
    assert "unknown" not in str(details).lower()
    assert details["pattern_detected"] in (
        "no_graph_pattern",
        "short_time_burst",
        "patient_cluster_same_provider",
        "high_degree_provider",
    )


def test_11_server_generates_claim_id_when_omitted():
    claim_data = {
        "patient_id": "21811",
        "provider_id": "4979",
        "amount": 2500.0,
        "documents": ["invoice"],
        "history": [],
        "insurance": "CNOPS",
    }
    posted = _post_claim(claim_data)
    assert posted["claim_id"]
    graph = next(ar for ar in posted["agent_results"] if ar["agent_name"] == "Graph Agent")
    assert f"claim::{posted['claim_id']}" in graph["details"]["risk_nodes"]


def test_12_sequential_claim_requests_stay_fast():
    """Parallel agents + async pipeline: p50-style budget <2s (max <3s on slow CI)."""
    import time

    claim_data = {
        "patient_id": "12345678",
        "provider_id": "3037",
        "amount": 5000.0,
        "documents": ["medical_report", "invoice", "prescription"],
        "history": [
            {"amount": 2000.0, "date": "2024-01-15", "recent": False},
        ],
        "insurance": "CNSS",
    }
    durations = []
    for _ in range(5):
        t0 = time.perf_counter()
        response = client.post("/claim", json=claim_data, headers=API_HEADERS)
        durations.append(time.perf_counter() - t0)
        assert response.status_code == 200
        _assert_common_response_shape(response.json())

    mean_s = sum(durations) / len(durations)
    max_s = max(durations)
    assert mean_s < 2.0, f"mean latency {mean_s:.2f}s expected <2s"
    assert max_s < 3.0, f"max latency {max_s:.2f}s expected <3s (CI/dev variance)"


def test_13_multipart_upload_extracts_text_and_approves():
    content = b"medical report invoice prescription ordonnance facture"
    response = client.post(
        "/claim/upload",
        data={
            "patient_id": "12345678",
            "provider_id": "3037",
            "amount": "5000",
            "insurance": "CNSS",
            "history_json": json.dumps(
                [{"amount": 2000.0, "date": "2024-01-15", "recent": False}]
            ),
        },
        files=[("files", ("evidence.txt", content, "text/plain"))],
        headers=API_HEADERS,
    )
    assert response.status_code == 200
    data = response.json()
    _assert_common_response_shape(data)
    assert data["decision"] == "APPROVED"


def test_14_json_body_documents_base64_merges_extractions():
    raw = b"medical report invoice prescription ordonnance facture"
    b64 = base64.b64encode(raw).decode("ascii")
    claim_data = {
        "patient_id": "12345678",
        "provider_id": "3037",
        "amount": 5000.0,
        "insurance": "CNSS",
        "documents": [],
        "documents_base64": [{"name": "pack.txt", "content_base64": b64}],
        "history": [{"amount": 2000.0, "date": "2024-01-15", "recent": False}],
    }
    posted = _post_claim(claim_data)
    assert posted["decision"] == "APPROVED"
    doc = next(ar for ar in posted["agent_results"] if ar["agent_name"] == "Document Agent")
    assert doc["decision"] is True
