"""
Test file for blockchain and IPFS integration
Run: pytest test_blockchain_flow.py -v
"""
import pytest

# Blockchain/IPFS tests are integration-heavy (RPC connectivity, contract deployment, Pinata)
# and are out of scope for the ClaimGuard backend API end-to-end contract tests.
pytest.skip("Skip blockchain/IPFS integration tests.", allow_module_level=True)
import os
import sys
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import ClaimInput, ClaimResult, BlockchainProof, IPFSStorage
from services.blockchain import BlockchainService, get_blockchain_service
from services.ipfs import IPFSService, get_ipfs_service


class TestBlockchainService:
    """Test blockchain service functionality"""
    
    def test_hash_claim_id(self):
        """Test claim ID hashing"""
        claim_id = "test-claim-123"
        hash1 = BlockchainService.hash_claim_id(claim_id)
        hash2 = BlockchainService.hash_claim_id(claim_id)
        
        # Same input should produce same hash
        assert hash1 == hash2
        # Hash should be 32 bytes
        assert len(hash1) == 32
    
    def test_hash_documents(self):
        """Test document hashing"""
        ipfs_hashes = ["QmHash1", "QmHash2", "QmHash3"]
        hash1 = BlockchainService.hash_documents(ipfs_hashes)
        hash2 = BlockchainService.hash_documents(["QmHash3", "QmHash2", "QmHash1"])
        
        # Hash should be same regardless of order
        assert hash1 == hash2
        assert len(hash1) == 32
    
    def test_generate_zk_proof_hash(self):
        """Test ZK proof hash generation"""
        zk_hash = BlockchainService.generate_zk_proof_hash(
            claim_id="claim-123",
            patient_id="patient-456",
            score=85.5
        )
        
        assert len(zk_hash) == 32
        assert isinstance(zk_hash, bytes)
    
    def test_network_info_without_config(self):
        """Test network info when contract not deployed"""
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(BlockchainService, '__init__', lambda self: None):
                service = BlockchainService.__new__(BlockchainService)
                service.w3 = MagicMock()
                service.w3.is_connected.return_value = True
                service.w3.eth.chain_id = 11155111
                service.w3.eth.block_number = 12345
                service.w3.eth.gas_price = 1000000000
                service.contract_address = None
                service.address = None
                
                info = service.get_network_info()
                
                assert info['connected'] == True
                assert info['chain_id'] == 11155111


class TestIPFSService:
    """Test IPFS service functionality"""
    
    def test_compute_hash(self):
        """Test content hashing"""
        content = b"test document content"
        hash1 = IPFSService.compute_hash(content)
        hash2 = IPFSService.compute_hash(content)
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex string
    
    def test_encrypt_decrypt(self):
        """Test content encryption/decryption"""
        with patch.dict(os.environ, {"DOCUMENT_ENCRYPTION_KEY": "test-key-123"}):
            service = IPFSService.__new__(IPFSService)
            service.encryption_key = "test-key-123"
            
            original = b"sensitive medical data"
            encrypted = service._encrypt_content(original)
            decrypted = service._decrypt_content(encrypted)
            
            # Decrypted should match original
            assert decrypted == original
            # Encrypted should be different
            assert encrypted != original
    
    @pytest.mark.asyncio
    async def test_simulated_upload(self):
        """Test simulated IPFS upload (no Pinata credentials)"""
        service = IPFSService.__new__(IPFSService)
        service.pinata_jwt = None
        service.pinata_api_key = None
        service.pinata_api_secret = None
        service.encryption_key = "test-key"
        
        result = await service._simulate_upload(
            content=b"test content",
            file_name="test.json",
            original_hash="abc123"
        )
        
        assert result.ipfs_hash.startswith("Qm")
        assert result.file_name == "test.json"
        assert result.content_hash == "abc123"
    
    @pytest.mark.asyncio
    async def test_upload_json(self):
        """Test JSON upload"""
        service = IPFSService.__new__(IPFSService)
        service.pinata_jwt = None
        service.pinata_api_key = None
        service.pinata_api_secret = None
        service.encryption_key = "test-key"
        
        result = await service.upload_json(
            data={"test": "data"},
            file_name="test.json",
            encrypt=False
        )
        
        assert result.ipfs_hash.startswith("Qm")


class TestModels:
    """Test Pydantic models"""
    
    def test_blockchain_proof_model(self):
        """Test BlockchainProof model"""
        proof = BlockchainProof(
            tx_hash="0x123abc",
            block_number=12345,
            claim_id_hash="0xabc123",
            document_hash="0xdef456",
            zk_proof_hash="0xghi789",
            gas_used=150000,
            status="success"
        )
        
        assert proof.tx_hash == "0x123abc"
        assert proof.block_number == 12345
        assert proof.status == "success"
    
    def test_ipfs_storage_model(self):
        """Test IPFSStorage model"""
        storage = IPFSStorage(
            ipfs_hashes=["QmHash1", "QmHash2"],
            document_map={"doc1": "QmHash1", "doc2": "QmHash2"},
            total_files=2,
            encrypted=True
        )
        
        assert len(storage.ipfs_hashes) == 2
        assert storage.total_files == 2
        assert storage.encrypted == True
    
    def test_claim_result_with_blockchain(self):
        """Test ClaimResult with blockchain proof"""
        proof = BlockchainProof(
            tx_hash="0x123",
            block_number=100,
            claim_id_hash="0xabc",
            document_hash="0xdef",
            zk_proof_hash="0xghi"
        )
        
        storage = IPFSStorage(
            ipfs_hashes=["QmTest"],
            document_map={"test": "QmTest"},
            total_files=1
        )
        
        result = ClaimResult(
            claim_id="test-claim",
            decision="APPROVED",
            score=85.0,
            agent_results=[],
            blockchain_proof=proof,
            ipfs_storage=storage
        )
        
        assert result.blockchain_proof.tx_hash == "0x123"
        assert result.ipfs_storage.ipfs_hashes == ["QmTest"]


class TestIntegration:
    """Integration tests for full flow"""
    
    def test_full_claim_flow_simulated(self):
        """Test full claim flow with simulated blockchain/IPFS"""
        import asyncio

        from services.consensus import ConsensusSystem
        from models import AgentResult

        # Mock Firebase
        with patch('services.consensus.db') as mock_db:
            mock_db.collection.return_value.document.return_value.set = Mock()

            # Create consensus system without blockchain
            with patch.dict(os.environ, {}, clear=True):
                consensus = ConsensusSystem()

                # Process a claim
                claim_data = {
                    "patient_id": "patient-001",
                    "provider_id": "prov-001",
                    "amount": 5000.0,
                    "documents": ["doc1.pdf", "doc2.pdf"],
                    "history": [],
                    "insurance": "CNSS"
                }

                result = asyncio.run(consensus.process_claim(claim_data))
                
                assert result.claim_id is not None
                assert result.decision in ["APPROVED", "REJECTED"]
                assert 0 <= result.score <= 100
    
    def test_claim_input_validation(self):
        """Test claim input validation"""
        # Valid input
        claim = ClaimInput(
            patient_id="patient-001",
            provider_id="prov-001",
            amount=1000.0,
            documents=["doc1.pdf"],
            history=[],
            insurance="CNSS"
        )
        
        assert claim.patient_id == "patient-001"
        assert claim.amount == 1000.0
        
        # Invalid amount should raise error
        with pytest.raises(Exception):
            ClaimInput(
                patient_id="patient-001",
                provider_id="prov-001",
                amount=-100,  # Invalid negative amount
                documents=[],
                history=[],
                insurance="CNSS"
            )


class TestContractInteraction:
    """Test smart contract interaction (requires deployed contract)"""
    
    @pytest.mark.skipif(
        not os.getenv("CONTRACT_ADDRESS"),
        reason="Contract not deployed"
    )
    def test_validate_claim_on_chain(self):
        """Test actual blockchain transaction"""
        service = get_blockchain_service()
        
        result = service.validate_claim_on_chain(
            claim_id="pytest-test-claim",
            score=90.0,
            decision="APPROVED",
            ipfs_hashes=["QmTestHash"],
            patient_id="pytest-patient"
        )
        
        assert result['tx_hash'] is not None
        assert result['status'] == 'success'
        assert result['block_number'] > 0
    
    @pytest.mark.skipif(
        not os.getenv("CONTRACT_ADDRESS"),
        reason="Contract not deployed"
    )
    def test_get_claim_proof(self):
        """Test retrieving claim proof from blockchain"""
        service = get_blockchain_service()
        
        # First validate a claim
        service.validate_claim_on_chain(
            claim_id="pytest-retrieve-test",
            score=85.0,
            decision="APPROVED",
            ipfs_hashes=["QmTest"],
            patient_id="pytest-patient"
        )
        
        # Then retrieve it
        proof = service.get_claim_proof("pytest-retrieve-test")
        
        assert proof is not None
        assert proof['score'] == 85
        assert proof['approved'] == True


def test_health_check():
    """Simple health check test"""
    assert True


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
