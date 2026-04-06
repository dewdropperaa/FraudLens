"""
IPFS service for ClaimGuard - handles document storage on IPFS via Pinata
"""
import os
import json
import hashlib
import secrets
import aiohttp
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import time
from functools import wraps

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from claimguard.config import load_environment, parse_document_encryption_key

_ENCRYPTION_VERSION = 1
_NONCE_LEN = 12


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry async function on failure"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                        await asyncio.sleep(delay)
            raise last_error
        return wrapper
    return decorator


@dataclass
class IPFSUploadResult:
    """Result of IPFS upload operation"""
    ipfs_hash: str
    file_name: str
    file_size: int
    content_hash: str
    pin_size: int
    timestamp: float


class IPFSService:
    """
    Service for uploading and managing documents on IPFS via Pinata
    Ensures no sensitive data is exposed - documents are encrypted before upload
    """
    
    PINATA_API_URL = "https://api.pinata.cloud"
    
    def __init__(self):
        load_environment()
        # Pinata API credentials
        self.pinata_api_key = os.getenv("PINATA_API_KEY")
        self.pinata_api_secret = os.getenv("PINATA_API_SECRET")
        self.pinata_jwt = os.getenv("PINATA_JWT")

        raw_key = os.getenv("DOCUMENT_ENCRYPTION_KEY", "").strip()
        if not raw_key:
            raise RuntimeError(
                "DOCUMENT_ENCRYPTION_KEY is required (32-byte key: 64 hex chars, base64, or 32 UTF-8 chars)"
            )
        self._aes_key: bytes = parse_document_encryption_key(raw_key)

        # Validate credentials
        self._validate_credentials()
    
    def _validate_credentials(self) -> None:
        """Validate that Pinata credentials are available"""
        if not self.pinata_jwt and not (self.pinata_api_key and self.pinata_api_secret):
            print("Warning: Pinata credentials not configured. IPFS uploads will be simulated.")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Pinata API requests"""
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.pinata_jwt:
            headers["Authorization"] = f"Bearer {self.pinata_jwt}"
        elif self.pinata_api_key and self.pinata_api_secret:
            headers["pinata_api_key"] = self.pinata_api_key
            headers["pinata_api_secret"] = self.pinata_api_secret
        
        return headers
    
    def _encrypt_content(self, content: bytes) -> bytes:
        """AES-256-GCM: version byte | 12-byte nonce | ciphertext + tag."""
        aesgcm = AESGCM(self._aes_key)
        nonce = secrets.token_bytes(_NONCE_LEN)
        ciphertext = aesgcm.encrypt(nonce, content, None)
        return bytes([_ENCRYPTION_VERSION]) + nonce + ciphertext

    def _decrypt_content(self, encrypted_content: bytes) -> bytes:
        if len(encrypted_content) < 1 + _NONCE_LEN + 16:
            raise ValueError("encrypted payload too short or corrupt")
        if encrypted_content[0] != _ENCRYPTION_VERSION:
            raise ValueError(
                "unsupported document encryption version; re-upload with current DOCUMENT_ENCRYPTION_KEY"
            )
        nonce = encrypted_content[1 : 1 + _NONCE_LEN]
        ciphertext = encrypted_content[1 + _NONCE_LEN :]
        aesgcm = AESGCM(self._aes_key)
        return aesgcm.decrypt(nonce, ciphertext, None)
    
    @staticmethod
    def compute_hash(content: bytes) -> str:
        """Compute SHA-256 hash of content"""
        return hashlib.sha256(content).hexdigest()
    
    @retry_on_failure(max_retries=3, delay=2.0)
    async def upload_document(
        self,
        content: bytes,
        file_name: str,
        encrypt: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> IPFSUploadResult:
        """
        Upload a document to IPFS via Pinata
        
        Args:
            content: Raw document content
            file_name: Name of the file
            encrypt: Whether to encrypt the content before upload
            metadata: Optional metadata to attach
            
        Returns:
            IPFSUploadResult with hash and metadata
        """
        # Compute original hash for verification
        original_hash = self.compute_hash(content)
        
        # Encrypt if requested
        if encrypt:
            content = self._encrypt_content(content)
            file_name = f"encrypted_{file_name}"
        
        # Check if credentials are available
        if not self.pinata_jwt and not (self.pinata_api_key and self.pinata_api_secret):
            # Simulate upload for testing
            return await self._simulate_upload(content, file_name, original_hash)
        
        # Prepare multipart form data
        form_data = aiohttp.FormData()
        form_data.add_field(
            'file',
            content,
            filename=file_name,
            content_type='application/octet-stream'
        )
        
        # Add metadata if provided
        if metadata:
            metadata_json = json.dumps({
                "name": file_name,
                "keyvalues": {k: str(v) for k, v in metadata.items()}
            })
            form_data.add_field('pinataMetadata', metadata_json)
        
        # Upload to Pinata
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.PINATA_API_URL}/pinning/pinFileToIPFS",
                headers=self._get_headers(),
                data=form_data
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Pinata upload failed: {error_text}")
                
                result = await response.json()
        
        return IPFSUploadResult(
            ipfs_hash=result['IpfsHash'],
            file_name=file_name,
            file_size=len(content),
            content_hash=original_hash,
            pin_size=result.get('PinSize', len(content)),
            timestamp=time.time()
        )
    
    async def _simulate_upload(
        self,
        content: bytes,
        file_name: str,
        original_hash: str
    ) -> IPFSUploadResult:
        """Simulate IPFS upload for testing without Pinata credentials"""
        # Generate a simulated IPFS hash (Qm prefix is typical for IPFS)
        import hashlib
        hash_input = f"{file_name}:{original_hash}:{time.time()}".encode()
        simulated_hash = "Qm" + hashlib.sha256(hash_input).hexdigest()[:44]
        
        print(f"[SIMULATED] IPFS upload: {file_name} -> {simulated_hash}")
        
        return IPFSUploadResult(
            ipfs_hash=simulated_hash,
            file_name=file_name,
            file_size=len(content),
            content_hash=original_hash,
            pin_size=len(content),
            timestamp=time.time()
        )
    
    async def upload_json(
        self,
        data: Dict[str, Any],
        file_name: str,
        encrypt: bool = True
    ) -> IPFSUploadResult:
        """
        Upload JSON data to IPFS
        
        Args:
            data: Dictionary to upload as JSON
            file_name: Name for the file
            encrypt: Whether to encrypt the content
            
        Returns:
            IPFSUploadResult with hash and metadata
        """
        content = json.dumps(data, indent=2).encode('utf-8')
        return await self.upload_document(content, file_name, encrypt=encrypt)
    
    @retry_on_failure(max_retries=3, delay=2.0)
    async def download_document(self, ipfs_hash: str, decrypt: bool = True) -> bytes:
        """
        Download a document from IPFS
        
        Args:
            ipfs_hash: IPFS hash of the document
            decrypt: Whether to decrypt the content
            
        Returns:
            Raw document content
        """
        # Use Pinata gateway for reliable access
        gateway_url = f"https://gateway.pinata.cloud/ipfs/{ipfs_hash}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(gateway_url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to download from IPFS: {response.status}")
                
                content = await response.read()
        
        if decrypt:
            content = self._decrypt_content(content)
        
        return content
    
    async def pin_list(self, status: str = "pinned") -> List[Dict[str, Any]]:
        """
        Get list of pinned files from Pinata
        
        Args:
            status: Filter by status ('pinned', 'unpinned', 'all')
            
        Returns:
            List of pinned file metadata
        """
        if not self.pinata_jwt and not (self.pinata_api_key and self.pinata_api_secret):
            return []
        
        params = {"status": status}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.PINATA_API_URL}/data/pinList",
                headers=self._get_headers(),
                params=params
            ) as response:
                if response.status != 200:
                    raise Exception(f"Failed to get pin list: {response.status}")
                
                result = await response.json()
                return result.get('rows', [])
    
    async def unpin(self, ipfs_hash: str) -> bool:
        """
        Unpin a file from Pinata
        
        Args:
            ipfs_hash: IPFS hash to unpin
            
        Returns:
            True if successful
        """
        if not self.pinata_jwt and not (self.pinata_api_key and self.pinata_api_secret):
            return True
        
        async with aiohttp.ClientSession() as session:
            async with session.delete(
                f"{self.PINATA_API_URL}/pinning/unpin/{ipfs_hash}",
                headers=self._get_headers()
            ) as response:
                return response.status == 200
    
    async def upload_claim_documents(
        self,
        claim_id: str,
        documents: List[Dict[str, Any]]
    ) -> Tuple[List[str], Dict[str, str]]:
        """
        Upload all claim documents to IPFS
        
        Args:
            claim_id: Claim identifier
            documents: List of document dicts with 'content' and 'name'
            
        Returns:
            Tuple of (list of IPFS hashes, dict mapping doc names to hashes)
        """
        ipfs_hashes = []
        doc_map = {}
        
        for doc in documents:
            content = doc.get('content', b'')
            if isinstance(content, str):
                content = content.encode('utf-8')
            
            file_name = doc.get('name', f'document_{len(ipfs_hashes)}')
            
            result = await self.upload_document(
                content=content,
                file_name=f"{claim_id}_{file_name}",
                encrypt=True,
                metadata={
                    "claim_id": claim_id,
                    "original_name": file_name,
                    "type": doc.get('type', 'unknown')
                }
            )
            
            ipfs_hashes.append(result.ipfs_hash)
            doc_map[file_name] = result.ipfs_hash
        
        return ipfs_hashes, doc_map
    
    def get_gateway_url(self, ipfs_hash: str) -> str:
        """Get public gateway URL for an IPFS hash"""
        return f"https://gateway.pinata.cloud/ipfs/{ipfs_hash}"


# Singleton instance
_ipfs_service = None


def get_ipfs_service() -> IPFSService:
    """Get or create IPFS service instance"""
    global _ipfs_service
    if _ipfs_service is None:
        _ipfs_service = IPFSService()
    return _ipfs_service
