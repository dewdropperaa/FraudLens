import sys
from pathlib import Path

from fastapi.testclient import TestClient

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from claimguard.main import create_app
from claimguard.security import create_access_token


def test_human_review_context_allows_authenticated_non_investigator(monkeypatch) -> None:
    from claimguard.v2 import orchestrator as orchestrator_module

    class FakeOrchestrator:
        def get_human_review_context(self, claim_id: str):
            return {
                "claim_id": claim_id,
                "decision": "HUMAN_REVIEW",
                "agents": [{"agent": "IdentityAgent", "status": "DONE"}],
            }

    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("CLAIMAGUARD_API_KEYS", "test-api-key-for-ci")
    monkeypatch.setenv("CLAIMAGUARD_JWT_SECRET", "test-secret")
    monkeypatch.setattr(orchestrator_module, "_singleton", FakeOrchestrator())

    token = create_access_token(user_id="user-1", email="user@test.local", role="patient")
    client = TestClient(create_app())
    response = client.get(
        "/v2/claim/claim-123/human-review-context",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["claim_id"] == "claim-123"
    assert payload["decision"] == "HUMAN_REVIEW"
    assert "agents" in payload
