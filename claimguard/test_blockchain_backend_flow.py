from fastapi.testclient import TestClient

from claimguard.main import app
import claimguard.services.consensus as consensus_module

API_HEADERS = {"X-API-Key": "test-api-key-for-ci"}


class _FakeBlockchainService:
    def validate_claim_on_chain(self, claim_id, score, decision, ipfs_hashes, patient_id):
        return {
            "tx_hash": "0xabc123fake",
            "block_number": 123,
            "status": "success",
            "claim_id_hash": "0xclaimhash",
            "document_hash": "0xdochash",
            "zk_proof_hash": "0xzkhash",
        }


class _FakeIPFSService:
    async def upload_claim_documents(self, claim_id, documents):
        return (["QmFakeHashOne", "QmFakeHashTwo"], {"doc_0.json": "QmFakeHashOne"})


def test_approved_claim_returns_tx_and_ipfs_hash(monkeypatch):
    # Force blockchain integration path without real Sepolia transaction.
    monkeypatch.setattr(consensus_module, "get_blockchain_service", lambda: _FakeBlockchainService())
    monkeypatch.setattr(consensus_module, "get_ipfs_service", lambda: _FakeIPFSService())

    # Rebuild singleton with blockchain enabled.
    consensus_module._consensus_singleton = None
    try:
        service = consensus_module.get_consensus_system()
        service._blockchain_enabled = True

        client = TestClient(app)
        payload = {
            "patient_id": "12345678",
            "provider_id": "4052",
            "amount": 3500.0,
            "documents": ["medical_report", "invoice", "prescription"],
            "history": [{"amount": 2500.0, "date": "2025-01-10", "recent": False}],
            "insurance": "CNSS",
        }
        response = client.post("/claim", json=payload, headers=API_HEADERS)
        assert response.status_code == 200
        body = response.json()

        assert body["decision"] == "APPROVED"
        assert body["tx_hash"] == "0xabc123fake"
        assert body["ipfs_hash"] == "QmFakeHashOne"
        assert body["claim_hash"] == "0xclaimhash"
        assert body["zk_proof_hash"] == "0xzkhash"
    finally:
        # Avoid leaking state to other tests.
        consensus_module._consensus_singleton = None
