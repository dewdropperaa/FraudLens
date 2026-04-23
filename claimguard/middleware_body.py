from __future__ import annotations

import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


def _max_body_bytes() -> int:
    raw = os.getenv("MAX_REQUEST_BODY_BYTES", "262144").strip()
    try:
        n = int(raw)
        return max(1024, min(n, 10 * 1024 * 1024))
    except ValueError:
        return 262144


def _max_v2_analyze_body_bytes() -> int:
    raw = os.getenv("MAX_V2_ANALYZE_BODY_BYTES", "26214400").strip()
    try:
        n = int(raw)
        return max(1024, min(n, 100 * 1024 * 1024))
    except ValueError:
        return 26214400


class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    """Reject oversized request bodies using Content-Length (fail closed for chunked without length)."""

    def __init__(self, app, max_bytes: int | None = None):
        super().__init__(app)
        self._max = max_bytes if max_bytes is not None else _max_body_bytes()
        self._v2_analyze_max = _max_v2_analyze_body_bytes()

    async def dispatch(self, request: Request, call_next):
        if request.method in ("POST", "PUT", "PATCH", "DELETE"):
            max_bytes = self._v2_analyze_max if request.url.path == "/v2/claim/analyze" else self._max
            cl = request.headers.get("content-length")
            if cl is not None:
                try:
                    if int(cl) > max_bytes:
                        return JSONResponse(
                            status_code=413,
                            content={"detail": f"Request body exceeds maximum size of {max_bytes} bytes"},
                        )
                except ValueError:
                    return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length"})
        return await call_next(request)
