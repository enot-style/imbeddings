import base64
import json
import logging
import time
import uuid
from typing import List

import numpy as np
import anyio
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .config import settings
from .embeddings import embed_images
from .image_io import load_image_from_source
from .model import ModelBundle, load_model_bundle, resolve_device
from .supported_models import load_supported_model_ids
from .schemas import EmbeddingItem, EmbeddingRequest, EmbeddingResponse, Usage


app = FastAPI(
    title="imbeddings",
    version=settings.service_version,
    docs_url=None,
    redoc_url=None,
    openapi_url="/schema.json",
)

logger = logging.getLogger("imbeddings")
inference_limiter = anyio.CapacityLimiter(settings.max_concurrent_inference)


def _error_type_for_status(status_code: int) -> str:
    if status_code in {400, 404, 409, 422}:
        return "invalid_request_error"
    if status_code == 401:
        return "authentication_error"
    if status_code == 403:
        return "permission_error"
    if status_code == 429:
        return "rate_limit_error"
    if status_code >= 500:
        return "server_error"
    return "invalid_request_error"


def _error_payload(message: str, error_type: str) -> dict:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": None,
            "code": None,
        }
    }


def _log_request(
    request: Request,
    status_code: int,
    duration_ms: float,
) -> None:
    payload = {
        "request_id": getattr(request.state, "request_id", None),
        "method": request.method,
        "path": request.url.path,
        "status": status_code,
        "duration_ms": round(duration_ms, 2),
        "client": request.client.host if request.client else None,
    }
    model = getattr(request.state, "model", None)
    if model:
        payload["model"] = model
    input_count = getattr(request.state, "input_count", None)
    if input_count is not None:
        payload["input_count"] = input_count
    logger.info(json.dumps(payload))


async def _read_body_with_limit(request: Request, max_bytes: int) -> bytes:
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > max_bytes:
            raise ValueError("Request body too large")
    return bytes(body)


@app.middleware("http")
async def request_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = request_id
    start = time.perf_counter()

    if request.method in {"POST", "PUT", "PATCH"}:
        max_bytes = settings.max_request_bytes
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > max_bytes:
                    response = JSONResponse(
                        status_code=413,
                        content=_error_payload(
                            f"Request body too large (max {max_bytes} bytes)",
                            _error_type_for_status(413),
                        ),
                    )
                    response.headers["x-request-id"] = request_id
                    _log_request(request, 413, (time.perf_counter() - start) * 1000)
                    return response
            except ValueError:
                response = JSONResponse(
                    status_code=400,
                    content=_error_payload(
                        "Invalid Content-Length header",
                        _error_type_for_status(400),
                    ),
                )
                response.headers["x-request-id"] = request_id
                _log_request(request, 400, (time.perf_counter() - start) * 1000)
                return response

        try:
            body = await _read_body_with_limit(request, max_bytes)
        except ValueError:
            response = JSONResponse(
                status_code=413,
                content=_error_payload(
                    f"Request body too large (max {max_bytes} bytes)",
                    _error_type_for_status(413),
                ),
            )
            response.headers["x-request-id"] = request_id
            _log_request(request, 413, (time.perf_counter() - start) * 1000)
            return response

        request._body = body

    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["x-request-id"] = request_id
    _log_request(request, response.status_code, duration_ms)
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_payload(str(exc.detail), _error_type_for_status(exc.status_code)),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    details = []
    for error in exc.errors():
        loc = ".".join(str(item) for item in error.get("loc", []))
        msg = error.get("msg", "Invalid request")
        details.append(f"{loc}: {msg}" if loc else msg)
    message = "; ".join(details) if details else "Invalid request"
    return JSONResponse(
        status_code=400,
        content=_error_payload(message, "invalid_request_error"),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content=_error_payload("Internal server error", "server_error"),
    )


def _require_api_key(authorization: str | None = Header(None)) -> None:
    if not settings.api_keys:
        return
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing bearer token")
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = parts[1].strip()
    if not token or token not in settings.api_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")


async def _run_inference(fn, *args):
    timeout = settings.inference_timeout_seconds
    try:
        if timeout:
            with anyio.fail_after(timeout):
                return await anyio.to_thread.run_sync(
                    fn, *args, limiter=inference_limiter
                )
        return await anyio.to_thread.run_sync(fn, *args, limiter=inference_limiter)
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Model inference timed out") from exc


def _get_bundle(model_id: str) -> ModelBundle:
    try:
        return load_model_bundle(model_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _normalize_input(raw_input) -> List[str]:
    if isinstance(raw_input, str):
        if raw_input == "":
            raise ValueError("input must not be empty")
        return [raw_input]
    if isinstance(raw_input, list):
        if not raw_input:
            return []
        if all(isinstance(item, str) for item in raw_input):
            if any(item == "" for item in raw_input):
                raise ValueError("input must not be empty")
            return list(raw_input)
    raise ValueError("input must be a string or list of strings")


def _format_embedding(embedding: List[float], encoding_format: str) -> List[float] | str:
    if encoding_format == "float":
        return embedding
    if encoding_format == "base64":
        array = np.asarray(embedding, dtype=np.float32)
        return base64.b64encode(array.tobytes()).decode("ascii")
    raise HTTPException(status_code=400, detail="encoding_format must be 'float' or 'base64'")


def _apply_dimensions(embeddings: List[List[float]], dimensions: int | None) -> List[List[float]]:
    if dimensions is None:
        return embeddings
    if not embeddings:
        return embeddings
    base_dim = len(embeddings[0])
    if dimensions > base_dim:
        raise HTTPException(
            status_code=400,
            detail=f"dimensions must be <= {base_dim}",
        )
    if dimensions == base_dim:
        return embeddings
    return [vector[:dimensions] for vector in embeddings]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/info")
def info() -> dict:
    device = resolve_device()
    return {
        "service": "imbeddings",
        "version": settings.service_version,
        "device": str(device),
        "cuda_memory_fraction": settings.cuda_memory_fraction,
        "max_loaded_models": settings.max_loaded_models,
        "max_batch_size": settings.max_batch_size,
        "max_image_width": settings.max_image_width,
        "max_image_height": settings.max_image_height,
        "max_image_bytes": settings.max_image_bytes,
        "max_image_pixels": settings.max_image_pixels,
        "max_image_redirects": settings.max_image_redirects,
        "max_request_bytes": settings.max_request_bytes,
        "max_concurrent_inference": settings.max_concurrent_inference,
        "inference_timeout_seconds": settings.inference_timeout_seconds,
        "remote_image_request_timeout_seconds": settings.remote_image_request_timeout_seconds,
        "supported_models": load_supported_model_ids(),
    }


@app.post(
    "/v1/embeddings",
    response_model=EmbeddingResponse,
    dependencies=[Depends(_require_api_key)],
)
async def create_embeddings(payload: EmbeddingRequest, req: Request) -> EmbeddingResponse:
    if not payload.model:
        raise HTTPException(status_code=400, detail="model is required")

    try:
        inputs = _normalize_input(payload.input)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not inputs:
        raise HTTPException(status_code=400, detail="input must not be empty")
    if len(inputs) > settings.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"input array must be {settings.max_batch_size} dimensions or less",
        )

    req.state.model = payload.model
    req.state.input_count = len(inputs)

    images = []
    for index, item in enumerate(inputs):
        try:
            image = load_image_from_source(item)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image at index {index}: {exc}",
            ) from exc
        images.append(image)

    bundle = _get_bundle(payload.model)
    embeddings = await _run_inference(
        embed_images,
        images,
        bundle.processor,
        bundle.model,
        bundle.device,
        True,
        "cls",
    )

    embeddings = _apply_dimensions(embeddings, payload.dimensions)

    data = []
    for index, embedding in enumerate(embeddings):
        data.append(
            EmbeddingItem(
                index=index,
                embedding=_format_embedding(embedding, payload.encoding_format),
            )
        )

    return EmbeddingResponse(
        data=data,
        model=payload.model,
        usage=Usage(prompt_tokens=0, total_tokens=0),
    )
