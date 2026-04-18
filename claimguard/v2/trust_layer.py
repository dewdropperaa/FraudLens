from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol

import requests
from pydantic import BaseModel, ConfigDict, Field

LOGGER = logging.getLogger("claimguard.v2.trust_layer")

_TRUST_DOC_TYPES = ("medical", "invoice", "prescription")


class TrustLayerIPFSFailure(RuntimeError):
    pass


class SanitizedTrustDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    document_type: str
    content: str


class OnChainTrustPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cid_hash: str
    validator: str
    timestamp: int
    ts_score: float = Field(ge=0, le=100)
    final_decision: str


class FirebaseTrustRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    cid: str
    Ts: float = Field(ge=0, le=100)
    decision: str
    agent_summary: str
    timestamp: str


class TrustLayerResult(BaseModel):
    cid: str
    tx_hash: Optional[str] = None
    firebase_id: str
    status: str


class IPFSClient(Protocol):
    def upload_documents(self, claim_id: str, documents: List[SanitizedTrustDocument]) -> str: ...


class BlockchainClient(Protocol):
    def store_record(self, payload: OnChainTrustPayload) -> str: ...


class FirebaseClient(Protocol):
    def store_record(self, payload: FirebaseTrustRecord) -> str: ...


class FallbackLogger(Protocol):
    def log_blockchain_failure(self, context: Dict[str, Any]) -> None: ...


class DefaultFallbackLogger:
    def log_blockchain_failure(self, context: Dict[str, Any]) -> None:
        LOGGER.warning("trust_layer_blockchain_fallback context=%s", context)


class PinataIPFSClient:
    def __init__(
        self,
        *,
        pinata_jwt: Optional[str] = None,
        pinata_api_key: Optional[str] = None,
        pinata_api_secret: Optional[str] = None,
    ) -> None:
        self._jwt = pinata_jwt or os.getenv("PINATA_JWT")
        self._api_key = pinata_api_key or os.getenv("PINATA_API_KEY")
        self._api_secret = pinata_api_secret or os.getenv("PINATA_API_SECRET")
        self._endpoint = os.getenv("PINATA_PIN_JSON_URL", "https://api.pinata.cloud/pinning/pinJSONToIPFS")

    def upload_documents(self, claim_id: str, documents: List[SanitizedTrustDocument]) -> str:
        if not documents:
            raise TrustLayerIPFSFailure("No trust-eligible documents found for IPFS upload.")
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self._jwt:
            headers["Authorization"] = f"Bearer {self._jwt}"
        elif self._api_key and self._api_secret:
            headers["pinata_api_key"] = self._api_key
            headers["pinata_secret_api_key"] = self._api_secret
        else:
            raise TrustLayerIPFSFailure("Missing Pinata credentials.")

        payload = {
            "pinataMetadata": {"name": f"claim-{claim_id}-trust-layer"},
            "pinataContent": {
                "claim_id": claim_id,
                "documents": [doc.model_dump() for doc in documents],
            },
        }
        response = requests.post(self._endpoint, headers=headers, data=json.dumps(payload), timeout=25)
        if response.status_code >= 400:
            raise TrustLayerIPFSFailure(f"Pinata upload failed ({response.status_code}): {response.text}")
        body = response.json()
        cid = body.get("IpfsHash")
        if not cid:
            raise TrustLayerIPFSFailure("Pinata response missing IpfsHash.")
        return str(cid)


class EthereumTrustClient:
    def __init__(self) -> None:
        self._rpc_url = os.getenv("SEPOLIA_RPC_URL", "")
        self._contract_address = os.getenv("TRUST_CONTRACT_ADDRESS", "")
        self._private_key = os.getenv("SEPOLIA_PRIVATE_KEY", "")
        self._validator_id = os.getenv("TRUST_VALIDATOR_ID", "claimguard-v2")
        self._abi_json = os.getenv("TRUST_CONTRACT_ABI_JSON", "")
        self._function_name = os.getenv("TRUST_CONTRACT_FUNCTION", "storeTrustRecord")

    def store_record(self, payload: OnChainTrustPayload) -> str:
        from web3 import Web3

        if not (self._rpc_url and self._contract_address and self._private_key and self._abi_json):
            raise RuntimeError("Blockchain trust configuration is incomplete.")
        w3 = Web3(Web3.HTTPProvider(self._rpc_url))
        if not w3.is_connected():
            raise RuntimeError("Unable to connect to Sepolia RPC.")
        account = w3.eth.account.from_key(self._private_key)
        contract = w3.eth.contract(
            address=Web3.to_checksum_address(self._contract_address),
            abi=json.loads(self._abi_json),
        )
        fn = getattr(contract.functions, self._function_name)
        txn = fn(
            payload.cid_hash,
            payload.validator,
            payload.timestamp,
            int(round(payload.ts_score * 100)),
            payload.final_decision,
        ).build_transaction(
            {
                "from": account.address,
                "nonce": w3.eth.get_transaction_count(account.address),
                "chainId": 11155111,
                "gasPrice": w3.eth.gas_price,
            }
        )
        signed = account.sign_transaction(txn)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        return str(tx_hash.hex())

    @property
    def validator_id(self) -> str:
        return self._validator_id


class FirestoreTrustClient:
    def __init__(self) -> None:
        self._collection = os.getenv("TRUST_FIRESTORE_COLLECTION", "claim_trust_records")
        self._credential_json = os.getenv("FIREBASE_CREDENTIALS_JSON", "").strip()
        self._credential_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "").strip()

    def _ensure_app(self):
        import firebase_admin
        from firebase_admin import credentials

        app_name = "claimguard-v2-trust-layer"
        try:
            return firebase_admin.get_app(app_name)
        except ValueError:
            if self._credential_json:
                cred_info = json.loads(self._credential_json)
                cred = credentials.Certificate(cred_info)
            elif self._credential_path:
                cred = credentials.Certificate(self._credential_path)
            else:
                cred = credentials.ApplicationDefault()
            return firebase_admin.initialize_app(cred, name=app_name)

    def store_record(self, payload: FirebaseTrustRecord) -> str:
        from firebase_admin import firestore

        app = self._ensure_app()
        client = firestore.client(app=app)
        ref = client.collection(self._collection).document()
        ref.set(payload.model_dump())
        return str(ref.id)


@dataclass
class TrustLayerService:
    ipfs_client: IPFSClient
    blockchain_client: BlockchainClient
    firebase_client: FirebaseClient
    fallback_logger: FallbackLogger
    validator_id: str = "claimguard-v2"

    @classmethod
    def build_default(cls) -> "TrustLayerService":
        chain_client = EthereumTrustClient()
        validator = chain_client.validator_id
        return cls(
            ipfs_client=PinataIPFSClient(),
            blockchain_client=chain_client,
            firebase_client=FirestoreTrustClient(),
            fallback_logger=DefaultFallbackLogger(),
            validator_id=validator,
        )

    @staticmethod
    def _document_type(value: Dict[str, Any]) -> str:
        for key in ("document_type", "doc_type", "type", "category"):
            raw = str(value.get(key, "")).strip().lower()
            if raw:
                return raw
        return ""

    @staticmethod
    def _extract_content(value: Dict[str, Any]) -> str:
        for key in ("text", "content", "extracted_text", "summary"):
            raw = value.get(key)
            if raw is not None:
                return str(raw)
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    def _sanitize_documents(self, documents: List[Dict[str, Any]]) -> List[SanitizedTrustDocument]:
        clean_documents: List[SanitizedTrustDocument] = []
        for idx, row in enumerate(documents):
            doc_type = self._document_type(row)
            if not any(token in doc_type for token in _TRUST_DOC_TYPES):
                continue
            payload = {
                "document_id": str(row.get("id") or row.get("document_id") or f"doc-{idx}"),
                "document_type": doc_type,
                "content": self._extract_content(row),
            }
            clean_documents.append(SanitizedTrustDocument.model_validate(payload))
        return clean_documents

    @staticmethod
    def _agent_summary(agent_outputs: List[Dict[str, Any]]) -> str:
        items: List[str] = []
        for output in agent_outputs[:3]:
            agent = str(output.get("agent", "agent"))
            explanation = str(output.get("explanation", "")).strip()
            items.append(f"{agent}: {explanation[:120]}")
        return " | ".join(items)

    @staticmethod
    def _hash_cid(cid: str) -> str:
        digest = hashlib.sha256(cid.encode("utf-8")).hexdigest()
        return f"0x{digest}"

    def process_if_applicable(
        self,
        *,
        claim_id: str,
        decision: str,
        ts_score: float,
        claim_request: Dict[str, Any],
        agent_outputs: List[Dict[str, Any]],
    ) -> Optional[TrustLayerResult]:
        if decision != "AUTO_APPROVE":
            return None

        documents = claim_request.get("documents", [])
        trusted_docs = self._sanitize_documents(documents)
        cid = self.ipfs_client.upload_documents(claim_id, trusted_docs)

        now = datetime.now(timezone.utc)
        onchain_payload = OnChainTrustPayload.model_validate(
            {
                "cid_hash": self._hash_cid(cid),
                "validator": self.validator_id,
                "timestamp": int(now.timestamp()),
                "ts_score": ts_score,
                "final_decision": decision,
            }
        )

        tx_hash: Optional[str] = None
        try:
            tx_hash = self.blockchain_client.store_record(onchain_payload)
        except Exception as exc:
            self.fallback_logger.log_blockchain_failure(
                {
                    "claim_id": claim_id,
                    "cid": cid,
                    "validator": self.validator_id,
                    "error": str(exc),
                }
            )

        firebase_payload = FirebaseTrustRecord.model_validate(
            {
                "claim_id": claim_id,
                "cid": cid,
                "Ts": ts_score,
                "decision": decision,
                "agent_summary": self._agent_summary(agent_outputs),
                "timestamp": now.isoformat(),
            }
        )
        firebase_id = self.firebase_client.store_record(firebase_payload)
        return TrustLayerResult(cid=cid, tx_hash=tx_hash, firebase_id=firebase_id, status="stored")
