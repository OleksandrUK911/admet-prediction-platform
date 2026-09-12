"""FastAPI application: POST /admet-profile, GET /admet-profile/{id},
GET /history, GET /health, GET /metrics - see backend-spec/api-contract.md."""

import logging
import os
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from . import db
from .inference import InvalidSmilesError, ModelService
from .logging_config import log_requests_middleware, log_with_fields
from .schemas import (
    AdmetProfileRequest,
    AdmetProfileResponse,
    ErrorResponse,
    HealthResponse,
    HistoryItem,
)

model_service: ModelService | None = None
limiter = Limiter(key_func=get_remote_address)

RATE_LIMIT_RESPONSE = {429: {"model": ErrorResponse, "description": "Too many requests - rate limit exceeded."}}
VALIDATION_RESPONSE = {422: {"model": ErrorResponse, "description": "Invalid input (e.g. unparseable SMILES)."}}
NOT_FOUND_RESPONSE = {404: {"model": ErrorResponse, "description": "No profile found for this id."}}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_service
    # Loaded once at startup, not per-request.
    model_service = ModelService()
    db.init_db()
    yield


app = FastAPI(
    title="ADMET Prediction API",
    description="SMILES -> RDKit descriptors/fingerprints -> predicted ADMET profile across 16 tasks "
    "(solubility, BBB penetration, Tox21 panel, ClinTox toxicity/FDA approval). Research/educational project.",
    version="0.1.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.middleware("http")(log_requests_middleware)

# Frontend origin(s) are configured via CORS_ALLOWED_ORIGINS (comma-separated)
# so staging/prod can point at their real hosted origin without editing
# source. Falls back to local-dev defaults when unset.
_default_cors_origins = "http://localhost:5173,http://localhost:3000"
_cors_origins_env = os.environ.get("CORS_ALLOWED_ORIGINS", _default_cors_origins)
CORS_ALLOWED_ORIGINS = [origin.strip() for origin in _cors_origins_env.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(status_code=429, content={"detail": "Too many requests, please slow down."})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # Centralized so every unexpected (500) error gets one consistent
    # response shape and one place logging the full traceback - expected
    # errors (422 invalid SMILES, 404 not found, 429 rate limit) are raised
    # explicitly below and never reach this handler.
    log_with_fields(logging.ERROR, "unhandled_exception", path=request.url.path, error=str(exc))
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Liveness check - reports whether the model bundle is loaded and the database is reachable.",
)
def health() -> dict:
    model_loaded = model_service is not None
    try:
        db.list_history(limit=1)
        db_connected = True
    except sqlite3.Error:
        db_connected = False
    return {"status": "ok", "model_loaded": model_loaded, "db_connected": db_connected}


@app.get(
    "/metrics",
    summary="Prometheus metrics",
    description="Application metrics (request counts by endpoint/status, request latency "
    "histogram) in Prometheus text-exposition format. Populated by the same "
    "log_requests_middleware every request already flows through - see logging_config.py.",
)
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post(
    "/admet-profile",
    response_model=AdmetProfileResponse,
    responses={**VALIDATION_RESPONSE, **RATE_LIMIT_RESPONSE},
    summary="Predict a full ADMET profile from a SMILES string",
    description="Parses the given SMILES with RDKit, computes descriptors and fingerprints, and "
    "returns predictions across all 16 tasks (solubility, BBB penetration, Tox21 panel, clinical "
    "trial toxicity, FDA approval likelihood) plus an applicability-domain flag. Successful profiles "
    "are persisted and retrievable via GET /admet-profile/{id}. Rate limited to 30 requests/minute per client.",
)
@limiter.limit("30/minute")
def create_admet_profile(request: Request, body: AdmetProfileRequest) -> dict:
    try:
        result = model_service.predict(body.smiles)
    except InvalidSmilesError as exc:
        log_with_fields(logging.INFO, "admet_profile_rejected", reason="invalid_smiles", smiles=body.smiles)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    log_with_fields(logging.INFO, "admet_profile_ok", id=result["id"], smiles=result["smiles"])

    # Every successful profile is persisted server-side - the frontend
    # never writes history directly, see backend-spec/api-contract.md.
    db.insert_profile(
        id=result["id"],
        smiles=result["smiles"],
        profile_json=result,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    return result


@app.get(
    "/admet-profile/{profile_id}",
    response_model=AdmetProfileResponse,
    responses={**NOT_FOUND_RESPONSE},
    summary="Fetch a previously computed ADMET profile",
    description="Returns a previously computed profile by id, in the same shape as POST /admet-profile. "
    "Used by the comparison page and shareable result links.",
)
def get_admet_profile(profile_id: str) -> dict:
    profile = db.get_profile(profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="No profile found for this id")
    return profile


@app.get(
    "/history",
    response_model=list[HistoryItem],
    summary="Recent profile history",
    description="Returns the most recent successfully computed profiles (newest first), persisted "
    "server-side by POST /admet-profile.",
)
def history() -> list[dict]:
    return db.list_history()
