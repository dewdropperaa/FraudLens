"""Blockchain service for ClaimGuard (Sepolia)."""
from __future__ import annotations

import hashlib
import hmac
import os
import time
from functools import wraps
from typing import Any, Dict, Optional

from eth_account import Account
from web3 import Web3
from web3.exceptions import ContractLogicError, TransactionNotFound

from claimguard.config import (
    get_sepolia_private_key,
    get_sepolia_rpc_url,
    load_environment,
    parse_document_encryption_key,
)


def retry_on_failure(max_retries: int = 3, delay: float = 2.0):
    """Decorator to retry function on failure"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                        time.sleep(delay)
            raise last_error
        return wrapper
    return decorator


class BlockchainService:
    """
    Service for interacting with ClaimValidator smart contract on Ethereum.
    Stores hash-only proofs on chain (never raw personal data).
    """
    
    # Contract ABI - matches ClaimValidator.sol
    CONTRACT_ABI = [
        {
            "inputs": [],
            "stateMutability": "nonpayable",
            "type": "constructor"
        },
        {
            "anonymous": False,
            "inputs": [
                {"indexed": True, "name": "claimIdHash", "type": "bytes32"},
                {"indexed": False, "name": "score", "type": "uint256"},
                {"indexed": False, "name": "approved", "type": "bool"},
                {"indexed": False, "name": "timestamp", "type": "uint256"},
                {"indexed": False, "name": "validator", "type": "address"}
            ],
            "name": "ClaimValidated",
            "type": "event"
        },
        {
            "inputs": [
                {"name": "claimIdHash", "type": "bytes32"},
                {"name": "score", "type": "uint256"},
                {"name": "approved", "type": "bool"}
            ],
            "name": "validateClaim",
            "outputs": [{"name": "", "type": "bool"}],
            "stateMutability": "nonpayable",
            "type": "function"
        },
        {
            "inputs": [{"name": "claimIdHash", "type": "bytes32"}],
            "name": "getClaimProof",
            "outputs": [
                {"name": "score", "type": "uint256"},
                {"name": "approved", "type": "bool"},
                {"name": "documentHash", "type": "bytes32"},
                {"name": "zkProofHash", "type": "bytes32"},
                {"name": "timestamp", "type": "uint256"},
                {"name": "validator", "type": "address"}
            ],
            "stateMutability": "view",
            "type": "function"
        },
        {
            "inputs": [{"name": "claimIdHash", "type": "bytes32"}],
            "name": "isClaimValidated",
            "outputs": [{"name": "", "type": "bool"}],
            "stateMutability": "view",
            "type": "function"
        },
        {
            "inputs": [{"name": "newValidator", "type": "address"}],
            "name": "setAuthorizedValidator",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function"
        },
        {
            "inputs": [],
            "name": "owner",
            "outputs": [{"name": "", "type": "address"}],
            "stateMutability": "view",
            "type": "function"
        },
        {
            "inputs": [],
            "name": "authorizedValidator",
            "outputs": [{"name": "", "type": "address"}],
            "stateMutability": "view",
            "type": "function"
        },
        {
            "inputs": [],
            "name": "totalValidatedClaims",
            "outputs": [{"name": "", "type": "uint256"}],
            "stateMutability": "view",
            "type": "function"
        },
        {
            "inputs": [{"name": "index", "type": "uint256"}],
            "name": "getClaimIdByIndex",
            "outputs": [{"name": "", "type": "bytes32"}],
            "stateMutability": "view",
            "type": "function"
        }
    ]
    
    def __init__(self):
        load_environment()
        self.rpc_url = get_sepolia_rpc_url()
        self.private_key = get_sepolia_private_key() or None
        self.contract_address = os.getenv("CONTRACT_ADDRESS")
        
        # Initialize Web3 connection
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        
        if not self.w3.is_connected():
            raise ConnectionError(f"Failed to connect to Ethereum network at {self.rpc_url}")
        
        # Setup account if private key provided
        if self.private_key:
            self.account = Account.from_key(self.private_key)
            self.address = self.account.address
        else:
            self.account = None
            self.address = None
        
        # Load contract if address provided
        self.contract = None
        if self.contract_address:
            self._load_contract()
    
    def _load_contract(self) -> None:
        """Load the smart contract instance"""
        if not self.w3.is_address(self.contract_address):
            raise ValueError(f"Invalid contract address: {self.contract_address}")
        
        self.contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(self.contract_address),
            abi=self.CONTRACT_ABI
        )
    
    def set_contract_address(self, address: str) -> None:
        """Set or update the contract address"""
        self.contract_address = address
        self._load_contract()
    
    @staticmethod
    def hash_claim_id(claim_id: str) -> bytes:
        """Hash a claim ID using keccak256 (never store raw IDs on chain)"""
        return Web3.keccak(text=claim_id)
    
    @staticmethod
    def hash_documents(ipfs_hashes: list[str]) -> bytes:
        """Hash combined IPFS references off-chain (for API proof)."""
        combined = "|".join(sorted(ipfs_hashes))
        return Web3.keccak(text=combined)
    
    @staticmethod
    def generate_zk_proof_hash(claim_id: str, patient_id: str, score: float) -> bytes:
        """
        Build a bytes32 **commitment** for on-chain anchoring.

        This is **not** a zero-knowledge proof: there is no SNARK/STARK verifier on-chain.
        The hash is a deterministic, privacy-aware binding: the raw ``patient_id`` never
        appears in plaintext in the commitment input (SHA-256 blind). Prefer
        ``ZK_PROOF_COMMITMENT_SECRET`` for HMAC; if unset, ``DOCUMENT_ENCRYPTION_KEY`` is
        used when present; otherwise SHA-256 of the structured body is used (dev/test).

        For production ZK, replace with Circom + snarkjs / rapidsnark or a hosted prover
        and store only the proof hash on-chain.
        """
        load_environment()
        blind = hashlib.sha256(patient_id.encode("utf-8")).digest()
        score_str = f"{float(score):.6f}".encode("ascii")
        body = b"v4|" + claim_id.encode("utf-8") + b"|" + blind + b"|" + score_str

        raw_secret = (os.getenv("ZK_PROOF_COMMITMENT_SECRET") or "").strip()
        if not raw_secret:
            raw_secret = (os.getenv("DOCUMENT_ENCRYPTION_KEY") or "").strip()

        if raw_secret:
            try:
                key = parse_document_encryption_key(raw_secret)
            except ValueError:
                key = hashlib.sha256(raw_secret.encode("utf-8")).digest()
            digest = hmac.new(key, body, hashlib.sha256).digest()
        else:
            digest = hashlib.sha256(body).digest()

        return Web3.keccak(digest)
    
    def get_transaction_count(self) -> int:
        """Get the nonce for the account"""
        if not self.address:
            raise ValueError("No account configured")
        return self.w3.eth.get_transaction_count(self.address, "pending")
    
    def get_gas_price(self) -> int:
        """Get current gas price with 10% buffer"""
        base_price = self.w3.eth.gas_price
        return int(base_price * 1.1)
    
    def estimate_gas(self, tx: Dict) -> int:
        """Estimate gas for transaction with 20% buffer"""
        try:
            estimated = self.w3.eth.estimate_gas(tx)
            return int(estimated * 1.2)
        except Exception:
            return 300000  # Default gas limit
    
    @retry_on_failure(max_retries=3, delay=2.0)
    def validate_claim_on_chain(
        self,
        claim_id: str,
        score: float,
        decision: str,
        ipfs_hashes: list[str],
        patient_id: str
    ) -> Dict[str, Any]:
        """
        Submit a claim validation to the blockchain
        
        Args:
            claim_id: Unique claim identifier
            score: Validation score (0-100)
            decision: "APPROVED" or "REJECTED"
            ipfs_hashes: List of IPFS document hashes
            patient_id: Patient identifier (used only for ZK proof)
            
        Returns:
            Dict with tx_hash, block_number, and status
        """
        if not self.contract:
            raise ValueError("Contract not loaded. Set CONTRACT_ADDRESS environment variable.")
        
        if not self.account:
            raise ValueError(
                "No account configured. Set SEPOLIA_PRIVATE_KEY or PRIVATE_KEY environment variable."
            )
        
        # Hash sensitive data - never store raw data on chain
        claim_id_hash = self.hash_claim_id(claim_id)
        document_hash = self.hash_documents(ipfs_hashes)
        zk_proof_hash = self.generate_zk_proof_hash(claim_id, patient_id, score)
        
        # Convert decision to boolean
        approved = decision.upper() == "APPROVED"
        
        # Build transaction
        tx = {
            'from': self.address,
            'nonce': self.get_transaction_count(),
            'gasPrice': self.get_gas_price(),
        }
        
        # Estimate gas
        gas_estimate = self.estimate_gas({
            **tx,
            'to': self.contract_address,
            'data': self.contract.encode_abi(
                fn_name="validateClaim",
                args=[claim_id_hash, int(score), approved]
            )
        })
        tx['gas'] = gas_estimate
        
        # Build and sign transaction
        transaction = self.contract.functions.validateClaim(
            claim_id_hash,
            int(score),
            approved
        ).build_transaction(tx)
        
        signed_tx = self.w3.eth.account.sign_transaction(transaction, self.private_key)
        
        # Send transaction
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        tx_hash_hex = self.w3.to_hex(tx_hash)
        
        print(f"Transaction sent: {tx_hash_hex}")
        
        # Wait for confirmation
        receipt = self._wait_for_transaction(tx_hash_hex)
        
        return {
            "tx_hash": tx_hash_hex,
            "block_number": receipt['blockNumber'],
            "gas_used": receipt['gasUsed'],
            "status": "success" if receipt['status'] == 1 else "failed",
            "claim_id_hash": self.w3.to_hex(claim_id_hash),
            "document_hash": self.w3.to_hex(document_hash),
            "zk_proof_hash": self.w3.to_hex(zk_proof_hash)
        }
    
    def _wait_for_transaction(self, tx_hash: str, timeout: int = 120) -> Dict:
        """Wait for transaction to be mined"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                receipt = self.w3.eth.get_transaction_receipt(tx_hash)
                if receipt is not None:
                    return dict(receipt)
            except TransactionNotFound:
                pass
            
            time.sleep(2)
        
        raise TimeoutError(f"Transaction {tx_hash} not mined within {timeout} seconds")
    
    def get_claim_proof(self, claim_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve claim validation proof from blockchain
        
        Args:
            claim_id: The claim identifier
            
        Returns:
            Dict with proof data or None if not found
        """
        if not self.contract:
            raise ValueError("Contract not loaded")
        
        claim_id_hash = self.hash_claim_id(claim_id)
        
        try:
            result = self.contract.functions.getClaimProof(claim_id_hash).call()
            
            return {
                "score": result[0],
                "approved": result[1],
                "document_hash": self.w3.to_hex(result[2]),
                "zk_proof_hash": self.w3.to_hex(result[3]),
                "timestamp": result[4],
                "validator": result[5],
                "claim_id_hash": self.w3.to_hex(claim_id_hash)
            }
        except ContractLogicError:
            return None
    
    def is_claim_validated(self, claim_id: str) -> bool:
        """Check if a claim has been validated on chain"""
        if not self.contract:
            return False
        
        claim_id_hash = self.hash_claim_id(claim_id)
        return self.contract.functions.isClaimValidated(claim_id_hash).call()
    
    def get_total_validated_claims(self) -> int:
        """Get total number of validated claims"""
        if not self.contract:
            return 0
        return self.contract.functions.totalValidatedClaims().call()
    
    def get_network_info(self) -> Dict[str, Any]:
        """Get current network information"""
        return {
            "connected": self.w3.is_connected(),
            "chain_id": self.w3.eth.chain_id,
            "latest_block": self.w3.eth.block_number,
            "gas_price": self.w3.eth.gas_price,
            "contract_address": self.contract_address,
            "account_address": self.address
        }


# Singleton instance
_blockchain_service = None


def get_blockchain_service() -> BlockchainService:
    """Get or create blockchain service instance"""
    global _blockchain_service
    if _blockchain_service is None:
        _blockchain_service = BlockchainService()
    return _blockchain_service
