from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from services.logging_service import log_event
from services.logger import bind_request_id, reset_request_id

LOGGER = logging.getLogger(__name__)


def register_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_context_and_error_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        request.state.raw_body = await request.body()
        token = bind_request_id(request_id)

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except HTTPException as exc:
            latency_ms = int((time.perf_counter() - started) * 1000)
            log_event(
                LOGGER,
                logging.WARNING,
                event="request_http_error",
                request_id=request_id,
                latency_ms=latency_ms,
                error_type=type(exc).__name__,
            )
            response = JSONResponse(
                status_code=exc.status_code,
                content={
                    "request_id": request_id,
                    "status": "error",
                    "message": str(exc.detail),
                    "error_type": type(exc).__name__,
                },
            )
        except Exception as exc:
            latency_ms = int((time.perf_counter() - started) * 1000)
            log_event(
                LOGGER,
                logging.ERROR,
                event="request_error",
                request_id=request_id,
                latency_ms=latency_ms,
                error_type=type(exc).__name__,
            )
            response = JSONResponse(
                status_code=500,
                content={
                    "request_id": request_id,
                    "status": "error",
                    "message": "Internal server error",
                    "error_type": type(exc).__name__,
                },
            )

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Cache-Control"] = "no-store"
        reset_request_id(token)
        return response
