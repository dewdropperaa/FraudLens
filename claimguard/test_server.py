import pytest

# This file previously depended on an externally-running server.
# The suite is now covered end-to-end in `test_claims.py` using FastAPI's TestClient.
pytest.skip("Covered by test_claims.py (TestClient-based).", allow_module_level=True)
