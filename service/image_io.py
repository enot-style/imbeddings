import base64
import ipaddress
from io import BytesIO
import socket
from urllib.parse import urljoin, urlparse

import httpx
from PIL import Image

from .config import settings


def _is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _strip_data_uri(value: str) -> str:
    if value.startswith("data:"):
        parts = value.split(",", 1)
        if len(parts) == 2:
            return parts[1]
    return value


def _enforce_limits(image: Image.Image) -> None:
    width, height = image.size
    if width > settings.max_image_width or height > settings.max_image_height:
        raise ValueError("Image dimensions exceed configured limits")


def _load_image_from_bytes(data: bytes) -> Image.Image:
    if len(data) > settings.max_image_bytes:
        raise ValueError("Image size exceeds configured byte limit")
    if settings.max_image_pixels > 0:
        Image.MAX_IMAGE_PIXELS = settings.max_image_pixels
    image = Image.open(BytesIO(data))
    image.load()
    _enforce_limits(image)
    return image.convert("RGB")


def _decode_base64_image(value: str) -> Image.Image:
    raw_value = _strip_data_uri(value).strip()
    try:
        data = base64.b64decode(raw_value, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ValueError("Invalid base64 image data") from exc
    return _load_image_from_bytes(data)

def _is_public_ip(ip: str) -> bool:
    address = ipaddress.ip_address(ip)
    return not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    )


def _ensure_public_host(hostname: str) -> None:
    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        ip = None

    if ip is not None:
        if not _is_public_ip(str(ip)):
            raise ValueError("URL resolves to a private or non-routable address")
        return

    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise ValueError("Unable to resolve host") from exc
    if not infos:
        raise ValueError("Unable to resolve host")
    for info in infos:
        ip = info[4][0]
        if not _is_public_ip(ip):
            raise ValueError("URL resolves to a private or non-routable address")


def _fetch_url_bytes(url: str) -> bytes:
    current_url = url
    max_redirects = settings.max_image_redirects
    for _ in range(max_redirects + 1):
        parsed = urlparse(current_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("Only http(s) URLs are allowed")
        if not parsed.hostname:
            raise ValueError("Invalid URL")

        _ensure_public_host(parsed.hostname)

        with httpx.Client(
            follow_redirects=False,
            timeout=settings.remote_image_request_timeout_seconds,
        ) as client:
            with client.stream("GET", current_url) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("Location")
                    if not location:
                        raise ValueError("Redirect response missing Location header")
                    current_url = urljoin(current_url, location)
                    continue

                response.raise_for_status()
                content_length = response.headers.get("Content-Length")
                if content_length:
                    try:
                        length = int(content_length)
                    except ValueError as exc:
                        raise ValueError("Invalid Content-Length header") from exc
                    if length > settings.max_image_bytes:
                        raise ValueError("Image size exceeds configured byte limit")

                data = bytearray()
                for chunk in response.iter_bytes():
                    data.extend(chunk)
                    if len(data) > settings.max_image_bytes:
                        raise ValueError("Image size exceeds configured byte limit")
                return bytes(data)

    raise ValueError("Too many redirects")


def load_image_from_source(source: str) -> Image.Image:
    if _is_url(source):
        data = _fetch_url_bytes(source)
        return _load_image_from_bytes(data)
    return _decode_base64_image(source)
