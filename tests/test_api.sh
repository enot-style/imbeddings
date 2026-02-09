#!/usr/bin/env bash
set -euo pipefail

SERVICE_URL="${IMBEDDINGS_SERVICE_URL:-http://localhost:${IMBEDDINGS_PORT:-8000}}"
MODEL_ID="${IMBEDDINGS_MODEL_ID:-facebook/dinov2-small}"
API_KEY="${IMBEDDINGS_API_KEY:-test-key}"

require() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require curl
require python3
require base64

expect_http() {
  local expected="$1"
  local actual="$2"
  if [[ "$actual" != "$expected" ]]; then
    echo "Expected HTTP $expected, got $actual" >&2
    exit 1
  fi
}

request() {
  local payload="$1"
  curl -sS -w "\n%{http_code}" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${API_KEY}" \
    -d "$payload" \
    "$SERVICE_URL/v1/embeddings"
}

request_no_auth() {
  local payload="$1"
  curl -sS -w "\n%{http_code}" \
    -H "Content-Type: application/json" \
    -d "$payload" \
    "$SERVICE_URL/v1/embeddings"
}

TMP_DIR=$(mktemp -d)
cleanup() { rm -rf "$TMP_DIR"; }
trap cleanup EXIT

# generate large pixel PNG (compressed, > max pixels)
python3 - <<'PY' "$TMP_DIR/huge_pixels.png"
import sys, struct, zlib
path=sys.argv[1]
width=800
height=800
r=g=b=0
row = bytes([0] + [r, g, b] * width)
raw = row * height
compressed = zlib.compress(raw, 9)

def chunk(tag, data):
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)

with open(path, "wb") as f:
    f.write(b"\x89PNG\r\n\x1a\n")
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    f.write(chunk(b"IHDR", ihdr))
    f.write(chunk(b"IDAT", compressed))
    f.write(chunk(b"IEND", b""))
PY

# generate oversized bytes PPM
python3 - <<'PY' "$TMP_DIR/huge_bytes.ppm"
import sys
path=sys.argv[1]
w=h=200
header=f"P6\n{w} {h}\n255\n".encode("ascii")
payload=bytes([0,0,0]) * (w*h)
with open(path, "wb") as f:
    f.write(header)
    f.write(payload)
PY

# read small image
SMALL_IMAGE_PATH="$(dirname "$0")/images/img.webp"
if [[ ! -f "$SMALL_IMAGE_PATH" ]]; then
  echo "Missing image: $SMALL_IMAGE_PATH" >&2
  exit 1
fi
small_b64=$(base64 < "$SMALL_IMAGE_PATH" | tr -d '\n')

huge_pixels_b64=$(base64 < "$TMP_DIR/huge_pixels.png" | tr -d '\n')
huge_bytes_b64=$(base64 < "$TMP_DIR/huge_bytes.ppm" | tr -d '\n')

# 1) auth required
body_and_status=$(request_no_auth "{\"model\":\"$MODEL_ID\",\"input\":\"$small_b64\"}")
status=$(echo "$body_and_status" | tail -n1)
expect_http 401 "$status"

# 2) basic request
body_and_status=$(request "{\"model\":\"$MODEL_ID\",\"input\":\"$small_b64\"}")
status=$(echo "$body_and_status" | tail -n1)
expect_http 200 "$status"
body=$(echo "$body_and_status" | sed '$d')
python3 - <<'PY' "$body"
import json, sys
payload=json.loads(sys.argv[1])
assert payload["object"] == "list"
item=payload["data"][0]
assert isinstance(item["embedding"], list)
assert len(item["embedding"]) > 0
PY

# 3) base64 output
body_and_status=$(request "{\"model\":\"$MODEL_ID\",\"input\":\"$small_b64\",\"encoding_format\":\"base64\"}")
status=$(echo "$body_and_status" | tail -n1)
expect_http 200 "$status"
body=$(echo "$body_and_status" | sed '$d')
python3 - <<'PY' "$body"
import base64, json, sys
payload=json.loads(sys.argv[1])
emb=payload["data"][0]["embedding"]
assert isinstance(emb, str)
raw=base64.b64decode(emb)
assert len(raw) % 4 == 0
assert len(raw) >= 4
PY

# 4) dimensions truncation
body_and_status=$(request "{\"model\":\"$MODEL_ID\",\"input\":\"$small_b64\",\"dimensions\":8}")
status=$(echo "$body_and_status" | tail -n1)
expect_http 200 "$status"
body=$(echo "$body_and_status" | sed '$d')
python3 - <<'PY' "$body"
import json, sys
payload=json.loads(sys.argv[1])
vec=payload["data"][0]["embedding"]
assert len(vec) == 8
PY

# 5) private URL blocked
body_and_status=$(request "{\"model\":\"$MODEL_ID\",\"input\":[\"http://127.0.0.1/image.png\"]}")
status=$(echo "$body_and_status" | tail -n1)
expect_http 400 "$status"

# 6) image bytes limit
body_and_status=$(request "{\"model\":\"$MODEL_ID\",\"input\":\"$huge_bytes_b64\"}")
status=$(echo "$body_and_status" | tail -n1)
expect_http 400 "$status"

# 7) image pixel limit
body_and_status=$(request "{\"model\":\"$MODEL_ID\",\"input\":\"$huge_pixels_b64\"}")
status=$(echo "$body_and_status" | tail -n1)
expect_http 400 "$status"

# 8) request size limit
payload=$(python3 - <<PY
import json
big = "a" * 400000
print(json.dumps({"model": "${MODEL_ID}", "input": big}))
PY
)
body_and_status=$(request "$payload")
status=$(echo "$body_and_status" | tail -n1)
expect_http 413 "$status"

# 9) schema is published
schema_status=$(curl -sS -o /dev/null -w "%{http_code}" "$SERVICE_URL/schema.json")
expect_http 200 "$schema_status"

echo "imbeddings API tests passed"
