import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


load_dotenv()

SERVICE_VERION = "0.1.0"


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"Invalid {name} value: {value}") from exc


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(f"Invalid {name} value: {value}") from exc


def _get_env_optional_float(name: str) -> Optional[float]:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(f"Invalid {name} value: {value}") from exc


def _get_env_api_keys(name: str) -> list[str]:
    value = os.getenv(name, "")
    if not value:
        return []
    normalized = value.replace(",", " ")
    return [item.strip() for item in normalized.split() if item.strip()]


@dataclass(frozen=True)
class Settings:
    device: str
    cuda_memory_fraction: Optional[float]
    max_loaded_models: int
    max_batch_size: int
    max_image_width: int
    max_image_height: int
    max_image_bytes: int
    max_image_pixels: int
    max_request_bytes: int
    max_concurrent_inference: int
    inference_timeout_seconds: float
    max_image_redirects: int
    remote_image_request_timeout_seconds: float
    service_version: str
    api_keys: list[str]


_max_image_width = _get_env_int("IMBEDDINGS_MAX_IMAGE_WIDTH", 256)
_max_image_height = _get_env_int("IMBEDDINGS_MAX_IMAGE_HEIGHT", 256)

settings = Settings(
    device=os.getenv("IMBEDDINGS_DEVICE", "auto"),
    cuda_memory_fraction=_get_env_optional_float("IMBEDDINGS_CUDA_MEMORY_FRACTION"),
    max_loaded_models=_get_env_int("IMBEDDINGS_MAX_LOADED_MODELS", 1),
    max_batch_size=_get_env_int("IMBEDDINGS_MAX_BATCH_SIZE", 4),
    max_image_width=_max_image_width,
    max_image_height=_max_image_height,
    max_image_bytes=_get_env_int("IMBEDDINGS_MAX_IMAGE_BYTES", 102_400),
    max_image_pixels=_get_env_int("IMBEDDINGS_MAX_IMAGE_PIXELS", 4_000_000),
    max_request_bytes=_get_env_int("IMBEDDINGS_MAX_REQUEST_BYTES", 2_000_000),
    max_concurrent_inference=_get_env_int("IMBEDDINGS_MAX_CONCURRENT_INFERENCE", 2),
    inference_timeout_seconds=_get_env_float("IMBEDDINGS_INFERENCE_TIMEOUT_SECONDS", 60.0),
    max_image_redirects=_get_env_int("IMBEDDINGS_MAX_IMAGE_REDIRECTS", 3),
    remote_image_request_timeout_seconds=_get_env_float(
        "IMBEDDINGS_REMOTE_IMAGE_REQUEST_TIMEOUT",
        10.0,
    ),
    service_version=SERVICE_VERION,
    api_keys=_get_env_api_keys("IMBEDDINGS_API_KEYS"),
)

if settings.max_loaded_models < 1:
    raise RuntimeError("IMBEDDINGS_MAX_LOADED_MODELS must be >= 1")

if settings.cuda_memory_fraction is not None and not (
    0.0 < settings.cuda_memory_fraction <= 1.0
):
    raise RuntimeError("IMBEDDINGS_CUDA_MEMORY_FRACTION must be > 0.0 and <= 1.0")

if settings.max_request_bytes < 1:
    raise RuntimeError("IMBEDDINGS_MAX_REQUEST_BYTES must be >= 1")

if settings.max_concurrent_inference < 1:
    raise RuntimeError("IMBEDDINGS_MAX_CONCURRENT_INFERENCE must be >= 1")

if settings.inference_timeout_seconds <= 0:
    raise RuntimeError("IMBEDDINGS_INFERENCE_TIMEOUT_SECONDS must be > 0")

if settings.max_image_pixels < 1:
    raise RuntimeError("IMBEDDINGS_MAX_IMAGE_PIXELS must be >= 1")

if settings.max_image_redirects < 0:
    raise RuntimeError("IMBEDDINGS_MAX_IMAGE_REDIRECTS must be >= 0")
