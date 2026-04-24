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


def is_trust_eligible(blackboard: Dict[str, Any]) -> bool:
    identity = blackboard.get("identity", {}) if isinstance(blackboard.get("identity"), dict) else {}
    amount = blackboard.get("amount")
    if amount is None and isinstance(blackboard.get("verified_structured_data"), dict):
        amount = blackboard["verified_structured_data"].get("amount")
    document_type = blackboard.get("document_type")
    if document_type is None and isinstance(blackboard.get("document_classification"), dict):
        document_type = blackboard["document_classification"].get("document_type")
    return (
        identity.get("cin") is not None
        and amount is not None
        and str(document_type or "").strip().lower() == "medical_claim_bundle"
    )


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
    trust_hash: str = ""
    dispute_risk: bool = False
    evidence_cid: str | None = None


class TrustLayerResult(BaseModel):
    cid: str | None = None
    evidence_cid: str | None = None
    tx_hash: Optional[str] = None
    firebase_id: str
    status: str
    trust_hash: str


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


class HallucinationGuard:
    # PROD-FIX: isolate true hallucination detection from degraded-mode signals.
    def should_flag_hallucination(self, context: dict) -> bool:
        if not isinstance(context, dict):
            return False
        if bool(context.get("layer2_disabled")) and not bool(context.get("layer1_triggered")):
            return False
        contradictory_to_ocr = bool(context.get("contradicts_ocr_text"))
        fabricated_field = bool(context.get("field_not_present_in_document"))
        return contradictory_to_ocr or fabricated_field


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

    def healthcheck(self) -> Dict[str, Any]:
        # BLOCKCHAIN-FIX: live Pinata connectivity/auth check.
        headers: Dict[str, str] = {}
        if self._jwt:
            headers["Authorization"] = f"Bearer {self._jwt}"
        elif self._api_key and self._api_secret:
            headers["pinata_api_key"] = self._api_key
            headers["pinata_secret_api_key"] = self._api_secret
        else:
            return {
                "status": "FAIL",
                "configured": False,
                "endpoint": "https://api.pinata.cloud/data/testAuthentication",
                "message": "Missing Pinata credentials.",
            }
        try:
            response = requests.get(
                "https://api.pinata.cloud/data/testAuthentication",
                headers=headers,
                timeout=15,
            )
            ok = response.status_code == 200
            payload: Dict[str, Any] = {"raw": response.text[:400]}
            try:
                payload = response.json()
            except Exception:
                pass
            return {
                "status": "OK" if ok else "FAIL",
                "configured": True,
                "endpoint": "https://api.pinata.cloud/data/testAuthentication",
                "http_status": response.status_code,
                "response": payload,
            }
        except Exception as exc:
            return {
                "status": "FAIL",
                "configured": True,
                "endpoint": "https://api.pinata.cloud/data/testAuthentication",
                "message": str(exc),
            }


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
        tx_hash_hex = str(tx_hash.hex())
        # BLOCKCHAIN-FIX: require mined confirmation to avoid "pretend" anchoring.
        timeout_s_raw = os.getenv("TRUST_CHAIN_RECEIPT_TIMEOUT_S", "120").strip()
        try:
            timeout_s = max(10, int(float(timeout_s_raw)))
        except ValueError:
            timeout_s = 120
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=timeout_s)
        if int(getattr(receipt, "status", 0)) != 1:
            raise RuntimeError(f"Blockchain transaction reverted: {tx_hash_hex}")
        return tx_hash_hex

    @staticmethod
    def _dummy_value_for_type(sol_type: str) -> Any:
        kind = str(sol_type or "").lower()
        if kind.startswith("uint") or kind.startswith("int"):
            return 1
        if kind == "address":
            return "0x0000000000000000000000000000000000000001"
        if kind.startswith("bytes32"):
            return "0x" + ("11" * 32)
        if kind.startswith("bytes"):
            return "0x11"
        if kind == "bool":
            return True
        if kind == "string":
            return "healthcheck"
        if kind.endswith("[]"):
            return []
        return 0

    def healthcheck(self) -> Dict[str, Any]:
        # BLOCKCHAIN-FIX: live RPC/contract and dry-run call check.
        from web3 import Web3

        configured = bool(self._rpc_url and self._contract_address and self._private_key and self._abi_json)
        if not configured:
            return {
                "status": "FAIL",
                "configured": False,
                "message": "Missing blockchain config (rpc/contract/private_key/abi).",
            }
        report: Dict[str, Any] = {"configured": True}
        try:
            w3 = Web3(Web3.HTTPProvider(self._rpc_url))
            if not w3.is_connected():
                return {"status": "FAIL", "configured": True, "message": "Unable to connect to Sepolia RPC."}
            report["rpc_connected"] = True
            report["chain_id"] = int(w3.eth.chain_id)
            checksum_address = Web3.to_checksum_address(self._contract_address)
            report["contract_address"] = checksum_address
            code = w3.eth.get_code(checksum_address)
            report["contract_deployed"] = bool(code and code != b"\x00")
            if not report["contract_deployed"]:
                return {**report, "status": "FAIL", "message": "No bytecode at configured contract address."}

            abi = json.loads(self._abi_json)
            contract = w3.eth.contract(address=checksum_address, abi=abi)
            fn_name = str(self._function_name or "").strip()
            fn_abi = next(
                (
                    item for item in abi
                    if isinstance(item, dict)
                    and item.get("type") == "function"
                    and item.get("name") == fn_name
                ),
                None,
            )
            if fn_abi is None:
                return {
                    **report,
                    "status": "FAIL",
                    "message": f"Function '{fn_name}' not found in contract ABI.",
                }
            inputs = fn_abi.get("inputs", []) if isinstance(fn_abi, dict) else []
            args = [self._dummy_value_for_type(str(inp.get("type", ""))) for inp in inputs]
            account = w3.eth.account.from_key(self._private_key)
            fn = getattr(contract.functions, fn_name)(*args)
            tx = fn.build_transaction({"from": account.address})
            _ = w3.eth.call(
                {
                    "to": checksum_address,
                    "from": account.address,
                    "data": tx.get("data", "0x"),
                }
            )
            report["dry_run"] = {
                "status": "OK",
                "function": fn_name,
                "args_types": [str(inp.get("type", "")) for inp in inputs],
            }
            return {**report, "status": "OK"}
        except Exception as exc:
            report["dry_run"] = {"status": "FAIL", "message": str(exc)}
            return {**report, "status": "FAIL", "message": str(exc)}

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
    def _hash_cid(cid: str | None) -> str:
        normalized = (cid or "").strip()
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return f"0x{digest}"

    @staticmethod
    def _stable_hash_payload(payload: Dict[str, Any]) -> str:
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _build_tier1_hash(
        self,
        *,
        claim_id: str,
        decision: str,
        ts_score: float,
        timestamp: str,
        agent_output_summary: str,
        flags: List[str],
    ) -> str:
        return self._stable_hash_payload(
            {
                "claim_id": claim_id,
                "decision": decision,
                "Ts": round(float(ts_score), 4),
                "timestamp": timestamp,
                "agent_output_summary": agent_output_summary,
                "flags": sorted(str(flag) for flag in flags),
            }
        )

    def process_approved_claim(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        claim_id = str(payload.get("claim_id") or "")
        raw_decision = str(payload.get("decision") or "").strip()
        if not raw_decision:
            return {"status": "SKIPPED"}
        decision = raw_decision.upper()
        ts_score = float(payload.get("ts_score", 0.0))
        claim_request = payload.get("claim_request") or {}
        blackboard = payload.get("blackboard") or {}
        agent_outputs = list(payload.get("agent_outputs") or [])
        flags = list(payload.get("flags") or [])
        dispute_risk = bool(payload.get("dispute_risk"))

        now = datetime.now(timezone.utc)
        timestamp_iso = now.isoformat()
        agent_summary = self._agent_summary(agent_outputs)
        trust_hash = self._build_tier1_hash(
            claim_id=claim_id,
            decision=decision,
            ts_score=ts_score,
            timestamp=timestamp_iso,
            agent_output_summary=agent_summary,
            flags=flags,
        )

        if decision != "APPROVED":
            # Tier-1 audit hash + Firebase; optional IPFS evidence bundle when disputed.
            evidence_cid: str | None = None
            if dispute_risk:
                bb = blackboard if isinstance(blackboard, dict) else {}
                dispute_docs = list(self._sanitize_documents(claim_request.get("documents", [])))
                if not dispute_docs and is_trust_eligible(bb):
                    raw_text = bb.get("ocr_text") or bb.get("text") or ""
                    dispute_docs.append(
                        SanitizedTrustDocument(
                            document_id=claim_id,
                            document_type="medical_claim_bundle",
                            content=raw_text[:5000],
                        )
                    )
                if not dispute_docs:
                    dispute_docs.append(
                        SanitizedTrustDocument(
                            document_id=f"{claim_id}-dispute-bundle",
                            document_type="medical_claim_bundle",
                            content=json.dumps(
                                {"agent_outputs": agent_outputs, "flags": flags},
                                ensure_ascii=False,
                                sort_keys=True,
                            )[:5000],
                        )
                    )
                try:
                    evidence_cid = self.ipfs_client.upload_documents(claim_id, dispute_docs)
                except Exception as e:
                    LOGGER.error(f"IPFS FAILED: {e}")
                    evidence_cid = None
            firebase_payload = FirebaseTrustRecord.model_validate(
                {
                    "claim_id": claim_id,
                    "cid": "",
                    "Ts": ts_score,
                    "decision": decision,
                    "agent_summary": agent_summary,
                    "timestamp": timestamp_iso,
                    "trust_hash": trust_hash,
                    "dispute_risk": dispute_risk,
                    "evidence_cid": evidence_cid,
                }
            )
            firebase_id = self.firebase_client.store_record(firebase_payload)
            return {
                "cid": None,
                "evidence_cid": evidence_cid,
                "tx_hash": None,
                "firebase_id": firebase_id,
                "status": "stored",
                "trust_hash": trust_hash,
            }

        assert decision == "APPROVED", "Approved-only side effects enforced"

        cid: str | None = None
        tx_hash: Optional[str] = None

        bb = blackboard if isinstance(blackboard, dict) else {}
        documents = list(self._sanitize_documents(claim_request.get("documents", [])))
        if not documents and is_trust_eligible(bb):
            raw_text = bb.get("ocr_text") or bb.get("text") or ""
            documents.append(
                SanitizedTrustDocument(
                    document_id=claim_id,
                    document_type="medical_claim_bundle",
                    content=raw_text[:5000],
                )
            )
        LOGGER.info(
            f"[TRUST LAYER] documents_count={len(documents)} eligible={is_trust_eligible(bb)}"
        )
        try:
            cid = self.ipfs_client.upload_documents(claim_id, documents)
        except Exception as e:
            LOGGER.error(f"IPFS FAILED: {e}")
            cid = None

        onchain_payload = OnChainTrustPayload.model_validate(
            {
                "cid_hash": self._hash_cid(cid),
                "validator": self.validator_id,
                "timestamp": int(now.timestamp()),
                "ts_score": ts_score,
                "final_decision": decision,
            }
        )
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
                "cid": cid or "",
                "Ts": ts_score,
                "decision": decision,
                "agent_summary": agent_summary,
                "timestamp": timestamp_iso,
                "trust_hash": trust_hash,
                "dispute_risk": False,
                "evidence_cid": None,
            }
        )
        firebase_id = self.firebase_client.store_record(firebase_payload)
        return {
            "cid": cid,
            "evidence_cid": None,
            "tx_hash": tx_hash,
            "firebase_id": firebase_id,
            "status": "stored",
            "trust_hash": trust_hash,
        }

    def process_if_applicable(
        self,
        *,
        claim_id: str,
        decision: str,
        ts_score: float,
        claim_request: Dict[str, Any],
        agent_outputs: List[Dict[str, Any]],
        flags: List[str] | None = None,
        score_evolution: List[float] | None = None,
        dispute_risk: bool = False,
    ) -> Optional[TrustLayerResult]:
        result = self.process_approved_claim(
            {
                "claim_id": claim_id,
                "decision": decision,
                "ts_score": ts_score,
                "claim_request": claim_request,
                "agent_outputs": agent_outputs,
                "flags": list(flags or []),
                "dispute_risk": dispute_risk,
            }
        )
        if result.get("status") == "SKIPPED":
            return None
        return TrustLayerResult.model_validate(result)

    def healthcheck(self) -> Dict[str, Any]:
        # BLOCKCHAIN-FIX: combined production readiness checks.
        ipfs_check = (
            self.ipfs_client.healthcheck()
            if hasattr(self.ipfs_client, "healthcheck")
            else {"status": "FAIL", "message": "IPFS client has no healthcheck()."}
        )
        chain_check = (
            self.blockchain_client.healthcheck()
            if hasattr(self.blockchain_client, "healthcheck")
            else {"status": "FAIL", "message": "Blockchain client has no healthcheck()."}
        )
        contract_ok = bool(chain_check.get("contract_deployed", False))
        rpc_ok = bool(chain_check.get("rpc_connected", False))
        dry_run_ok = str((chain_check.get("dry_run", {}) or {}).get("status", "")).upper() == "OK"
        pinata_ok = str(ipfs_check.get("status", "")).upper() == "OK"
        overall_ok = pinata_ok and rpc_ok and contract_ok and dry_run_ok
        return {
            "status": "OK" if overall_ok else "FAIL",
            "checks": {
                "pinata": ipfs_check,
                "sepolia_rpc": {
                    "status": "OK" if rpc_ok else "FAIL",
                    "chain_id": chain_check.get("chain_id"),
                },
                "contract": {
                    "status": "OK" if contract_ok else "FAIL",
                    "address": chain_check.get("contract_address"),
                },
                "tx_dry_run": chain_check.get("dry_run", {"status": "FAIL"}),
            },
        }
