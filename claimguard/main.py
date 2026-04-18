import base64
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

try:
    from google.api_core import exceptions as google_api_exceptions
except ImportError:  # pragma: no cover
    google_api_exceptions = None  # type: ignore[misc, assignment]

from claimguard.config import (
    cors_allow_credentials,
    get_cors_origins,
    load_environment,
    validate_required_settings,
)

from claimguard.firebase_config import is_test_environment
from claimguard.rate_limiting import limiter
from claimguard.middleware_body import MaxBodySizeMiddleware
from claimguard.routes import claims_router, v2_router

load_environment()

# Tiny transparent PNG — browsers request /favicon.ico; a 200 + body avoids spurious 404/empty errors.
_FAVICON_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_environment()
    validate_required_settings()
    # Fail fast at startup if Ollama is unavailable for ClaimGuard v2.
    from claimguard.v2 import get_v2_orchestrator

    get_v2_orchestrator()
    # Claims persist in Cloud Firestore (tests use in-memory store; see services.storage).
    if not is_test_environment():
        from claimguard.firestore_provision import ensure_default_firestore_database
        from claimguard.firebase_config import get_firestore_client

        ensure_default_firestore_database()
        get_firestore_client()
    yield


def create_app() -> FastAPI:
    application = FastAPI(
        title="ClaimGuard API",
        description="Multi-agent AI system for insurance claim verification",
        version="1.0.0",
        lifespan=lifespan,
    )

    @application.get("/favicon.ico", include_in_schema=False)
    async def _favicon() -> Response:
        return Response(
            content=_FAVICON_PNG,
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    application.state.limiter = limiter
    application.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    if google_api_exceptions is not None:

        @application.exception_handler(google_api_exceptions.GoogleAPIError)
        async def _google_api_error_handler(
            request: Request, exc: google_api_exceptions.GoogleAPIError
        ) -> JSONResponse:
            if isinstance(exc, google_api_exceptions.NotFound):
                return JSONResponse(
                    status_code=503,
                    content={
                        "detail": (
                            "Firestore is not available for this project (database may be missing). "
                            "Create a Firestore database in the Firebase or Google Cloud console."
                        )
                    },
                )
            return JSONResponse(
                status_code=503,
                content={"detail": str(exc) or "Google Cloud API error"},
            )

    application.add_middleware(MaxBodySizeMiddleware)

    origins = get_cors_origins()
    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=cors_allow_credentials(),
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["authorization", "content-type", "x-api-key"],
        expose_headers=["x-request-id"],
        max_age=600,
    )

    application.include_router(claims_router)
    application.include_router(v2_router)
    return application


app = create_app()


@app.get("/")
async def root():
    return {
        "message": "ClaimGuard API is running",
        "version": "1.0.0",
        "web_ui": (
            "The React dashboard is not this port. From repo: cd claimguard/frontend && npm run dev "
            "→ open http://localhost:5173 (API stays on :8000; Vite proxies /api to the backend)."
        ),
        "interactive_docs": "http://127.0.0.1:8000/docs",
        "endpoints": {
            "POST /claim": "Submit JSON claim (optional documents_base64[] for inline files)",
            "POST /claim/upload": "Multipart claim with file parts (extracts PDF/text/OCR)",
            "GET /claim/{id}": "Retrieve claim results by ID",
            "GET /claims": "List claims with pagination (filter=all|fraud|valid)",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("claimguard.main:app", host="0.0.0.0", port=8000)
