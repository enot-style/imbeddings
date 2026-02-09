#!/usr/bin/env bash
set -euo pipefail

SERVICE_URL="${IMBEDDINGS_SERVICE_URL:-http://localhost:${IMBEDDINGS_PORT:-8000}}"
MODEL_ID="${IMBEDDINGS_MODEL_ID:-facebook/dinov3-vits16-pretrain-lvd1689m}"
IMAGES_DIR="$(dirname "$0")/images"

if [[ ! -d "$IMAGES_DIR" ]]; then
  echo "Missing images directory: $IMAGES_DIR" >&2
  exit 1
fi

image_path="$IMAGES_DIR/img.webp"
if [[ ! -f "$image_path" ]]; then
  echo "Missing image: $image_path" >&2
  exit 1
fi
if [[ -f "$image_path" ]]; then
  base64_data=$(base64 -i "$image_path" | tr -d '\n')
curl -sS "$SERVICE_URL/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d @- <<JSON
{
  "model": "$MODEL_ID",
  "input": [
    "$base64_data"
  ]
}
JSON
  echo
fi
