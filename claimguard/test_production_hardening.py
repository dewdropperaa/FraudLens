"""
Production hardening checks: configuration, encryption, auth, persistence, and abuse controls.
"""
import os

import pytest
from fastapi.testclient import TestClient

from claimguard.config import parse_document_encryption_key
from claimguard.main import app
from claimguard.services.ipfs import IPFSService

API_HEADERS = {"X-API-Key": "test-api-key-for-ci"}

client = TestClient(app)


def test_dotenv_does_not_override_existing_env(monkeypatch, tmp_path):
    """python-dotenv must not clobber variables already set (e.g. in CI)."""
    env_file = tmp_path / ".env"
    env_file.write_text("CLAIMAGUARD_API_KEYS=wrong-key\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLAIMAGUARD_API_KEYS", "test-api-key-for-ci")
    from claimguard.config import load_environment

    load_environment()
    assert os.environ["CLAIMAGUARD_API_KEYS"] == "test-api-key-for-ci"


def test_document_encryption_key_parses_to_32_bytes():
    key = "0" * 32
    assert len(parse_document_encryption_key(key)) == 32


def test_aes_gcm_roundtrip():
    svc = IPFSService.__new__(IPFSService)
    svc._aes_key = parse_document_encryption_key("k" * 32)
    raw = b"phi must stay confidential"
    enc = svc._encrypt_content(raw)
    assert enc != raw
    dec = svc._decrypt_content(enc)
    assert dec == raw


def test_post_claim_without_credentials_returns_401():
    payload = {
        "patient_id": "1",
        "provider_id": "2",
        "amount": 100.0,
        "documents": [],
        "history": [],
        "insurance": "CNSS",
    }
    r = client.post("/claim", json=payload)
    assert r.status_code == 401


def test_post_claim_with_valid_api_key_succeeds():
    payload = {
        "patient_id": "991",
        "provider_id": "992",
        "amount": 2500.0,
        "documents": ["invoice"],
        "history": [],
        "insurance": "CNOPS",
    }
    r = client.post("/claim", json=payload, headers=API_HEADERS)
    assert r.status_code == 200
    data = r.json()
    cid = data["claim_id"]
    g = client.get(f"/claim/{cid}", headers=API_HEADERS)
    assert g.status_code == 200
    assert g.json()["claim_id"] == cid


def test_claims_list_pagination_shape():
    r = client.get("/claims?page=1&page_size=5", headers=API_HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) >= {"items", "total", "page", "page_size"}
    assert body["page"] == 1
    assert body["page_size"] == 5
    assert isinstance(body["items"], list)


def test_oversized_raw_body_rejected():
    r = client.post(
        "/claim",
        content=b"x" * 300_000,
        headers={**API_HEADERS, "content-type": "application/json"},
    )
    assert r.status_code == 413


def test_jwt_bearer_accepted_when_no_api_keys(monkeypatch):
    import jwt

    monkeypatch.setenv("CLAIMAGUARD_API_KEYS", "")
    monkeypatch.setenv("CLAIMAGUARD_JWT_SECRET", "jwt-test-secret-key-32bytes-min!!")
    token = jwt.encode(
        {"sub": "tester", "aud": "claimguard-api"},
        "jwt-test-secret-key-32bytes-min!!",
        algorithm="HS256",
    )
    payload = {
        "patient_id": "1",
        "provider_id": "2",
        "amount": 500.0,
        "documents": [],
        "history": [],
        "insurance": "CNSS",
    }
    r = client.post(
        "/claim",
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200
