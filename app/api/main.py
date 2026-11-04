import logging
import secrets
from contextlib import asynccontextmanager
from os import getenv

from dotenv import load_dotenv
from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.concurrency import run_in_threadpool

from app.api.schemas import (
    ClassificationResponse,
    RetrievalCase,
    RetrievalResponse,
    SupportRequest,
    SupportResponse,
)
from app.api.service import SupportAgentService

load_dotenv()

logger = logging.getLogger(__name__)


def require_api_key(
    x_api_key: str | None = Header(default=None),
):
    expected_key = getenv("API_KEY")

    if not expected_key:
        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured.",
        )

    if not x_api_key or not secrets.compare_digest(
        x_api_key,
        expected_key,
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return True


limiter = Limiter(
    key_func=get_remote_address,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = SupportAgentService()
    service.load()
    app.state.agent_service = service

    yield


app = FastAPI(
    title="Hiver AI Support Agent",
    description="AI-powered AppleSupport customer support agent",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler,
)


allowed_origins = [
    origin.strip()
    for origin in getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=[
        "Content-Type",
        "Authorization",
        "X-API-Key",
    ],
)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "hiver-ai-support-agent",
    }


@app.get("/ready")
def readiness(request: Request):
    service = getattr(
        request.app.state,
        "agent_service",
        None,
    )

    if service is None or not service.loaded:
        raise HTTPException(
            status_code=503,
            detail="Support agent is not ready.",
        )

    return {
        "status": "ready",
        "service": "hiver-ai-support-agent",
    }


@app.get(
    "/api/v1/intents",
    dependencies=[Depends(require_api_key)],
)
@limiter.limit("60/minute")
async def intents(request: Request):
    return {
        "intents": [
            "battery_charging",
            "keyboard_text_input",
            "connectivity",
            "apple_id_account",
            "messaging",
            "apple_app_service",
            "photos_media",
            "purchase_payment_order",
            "software_update",
            "device_performance",
        ]
    }


@app.post(
    "/api/v1/classify",
    response_model=ClassificationResponse,
    dependencies=[Depends(require_api_key)],
)
@limiter.limit("30/minute")
async def classify(
    request: Request,
    payload: SupportRequest,
):
    service = request.app.state.agent_service

    try:
        intent, confidence = await run_in_threadpool(
            service.classify,
            payload.message,
        )

        return ClassificationResponse(
            message=payload.message,
            intent=intent,
            confidence=confidence,
        )

    except Exception:
        logger.exception("Classification failed")
        raise HTTPException(
            status_code=500,
            detail="Classification failed. Please try again.",
        )


@app.post(
    "/api/v1/retrieve",
    response_model=RetrievalResponse,
    dependencies=[Depends(require_api_key)],
)
@limiter.limit("20/minute")
async def retrieve(
    request: Request,
    payload: SupportRequest,
):
    service = request.app.state.agent_service

    try:
        cases = await run_in_threadpool(
            service.retrieve,
            payload.message,
        )

        return RetrievalResponse(
            message=payload.message,
            cases=[
                RetrievalCase(**case)
                for case in cases
            ],
        )

    except Exception:
        logger.exception("Historical case retrieval failed")
        raise HTTPException(
            status_code=500,
            detail="Historical case retrieval failed.",
        )


@app.post(
    "/api/v1/support",
    response_model=SupportResponse,
    dependencies=[Depends(require_api_key)],
)
@limiter.limit("10/minute")
async def support(
    request: Request,
    payload: SupportRequest,
):
    service = request.app.state.agent_service

    try:
        result = await run_in_threadpool(
            service.process,
            payload.message,
        )

        return SupportResponse(**result)

    except Exception:
        logger.exception("Support processing failed")
        raise HTTPException(
            status_code=500,
            detail="Support processing failed. Please try again.",
        )
