#!/usr/bin/env bash
set -euo pipefail

SERVICE_URL="${IMBEDDINGS_SERVICE_URL:-http://localhost:${IMBEDDINGS_PORT:-8000}}"
MODEL_ID="${IMBEDDINGS_MODEL_ID:-facebook/dinov3-vits16-pretrain-lvd1689m}"
IMAGES_DIR="$(dirname "$0")/images"

image_path_1="$IMAGES_DIR/cmp1.jpg"
image_path_2="$IMAGES_DIR/cmp2.jpg"

if [[ ! -f "$image_path_1" ]]; then
  echo "Missing image: $image_path_1" >&2
  exit 1
fi

if [[ ! -f "$image_path_2" ]]; then
  echo "Missing image: $image_path_2" >&2
  exit 1
fi

base64_data_1=$(base64 -i "$image_path_1" | tr -d '\n')
base64_data_2=$(base64 -i "$image_path_2" | tr -d '\n')

curl -sS "$SERVICE_URL/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d @- <<JSON
{
  "model": "$MODEL_ID",
  "input": [
    "$base64_data_1",
    "$base64_data_2"
  ]
}
JSON

echo
