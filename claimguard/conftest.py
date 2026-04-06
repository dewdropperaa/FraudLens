"""
Pytest configuration: set required environment before any claimguard imports.
"""
import os


def pytest_configure(config) -> None:
    os.environ["ENVIRONMENT"] = "test"
    # 32 UTF-8 bytes for AES-256 key parsing
    os.environ["DOCUMENT_ENCRYPTION_KEY"] = "0" * 32
    os.environ["CLAIMAGUARD_API_KEYS"] = "test-api-key-for-ci"
    os.environ["CORS_ORIGINS"] = "http://testserver"
    # Block .env Pinata keys from loading into the process (simulated IPFS only in tests)
    os.environ["PINATA_JWT"] = ""
    os.environ["PINATA_API_KEY"] = ""
    os.environ["PINATA_API_SECRET"] = ""
    # Claim store: in-memory (see services.storage); no Firestore in tests.
    # Ensure claim tests do not hit live Sepolia / ABI encode paths (local .env may set these).
    os.environ["CONTRACT_ADDRESS"] = ""
    os.environ["SEPOLIA_PRIVATE_KEY"] = ""
    os.environ["PRIVATE_KEY"] = ""
