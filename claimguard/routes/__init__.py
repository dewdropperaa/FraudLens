from claimguard.routes.auth import router as auth_router
from claimguard.routes.claims import router as claims_router
from claimguard.routes.v2 import router as v2_router

__all__ = ["auth_router", "claims_router", "v2_router"]

