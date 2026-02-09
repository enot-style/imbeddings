#!/usr/bin/env bash
set -euo pipefail

SERVICE_URL="${IMBEDDINGS_SERVICE_URL:-http://localhost:${IMBEDDINGS_PORT:-8000}}"
MODEL_ID="${IMBEDDINGS_MODEL_ID:-facebook/dinov3-vits16-pretrain-lvd1689m}"
IMAGES_DIR="$(dirname "$0")/images"
RETENTION="1h"

image_path_1="$IMAGES_DIR/cmp3.png"
image_path_2="$IMAGES_DIR/cmp4.png"

if [[ ! -f "$image_path_1" ]]; then
  echo "Missing image: $image_path_1" >&2
  exit 1
fi

if [[ ! -f "$image_path_2" ]]; then
  echo "Missing image: $image_path_2" >&2
  exit 1
fi

upload_file() {
  local file_path="$1"
  local response_file
  local http_code
  response_file=$(mktemp)

  if [[ -n "$RETENTION" ]]; then
    http_code=$(curl -sS -o "$response_file" -w "%{http_code}" \
      -F "reqtype=fileupload" \
      -F "time=$RETENTION" \
      -F "fileToUpload=@$file_path" \
      "https://litterbox.catbox.moe/resources/internals/api.php")
  else
    http_code=$(curl -sS -o "$response_file" -w "%{http_code}" \
      -F "reqtype=fileupload" \
      -F "fileToUpload=@$file_path" \
      "https://litterbox.catbox.moe/resources/internals/api.php")
  fi

  if [[ ! "$http_code" =~ ^2 ]]; then
    echo "Upload failed ($http_code) for $file_path" >&2
    cat "$response_file" >&2
    rm -f "$response_file"
    exit 1
  fi

  local url
  url=$(tr -d '\r\n' < "$response_file")
  rm -f "$response_file"

  if [[ -z "$url" ]]; then
    echo "Upload returned empty URL for $file_path" >&2
    exit 1
  fi

  echo "$url"
}

url_1=$(upload_file "$image_path_1")
url_2=$(upload_file "$image_path_2")

curl -sS "$SERVICE_URL/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d @- <<JSON
{
  "model": "$MODEL_ID",
  "input": [
    "$url_1",
    "$url_2"
  ]
}
JSON

echo
