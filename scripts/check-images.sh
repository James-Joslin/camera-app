#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
docker compose --env-file "${CAMERA_ENV_FILE:-$ROOT_DIR/.env.dev.example}" -f "$ROOT_DIR/compose.dev.yml" config --quiet
docker compose --env-file "${CAMERA_ENV_FILE:-$ROOT_DIR/.env.prod.example}" -f "$ROOT_DIR/compose.prod.yml" config --quiet
docker compose --env-file "${CAMERA_ENV_FILE:-$ROOT_DIR/.env.dev.example}" -f "$ROOT_DIR/compose.dev.yml" -f "$ROOT_DIR/compose.tools.yml" config --quiet
docker compose --env-file "${CAMERA_ENV_FILE:-$ROOT_DIR/.env.prod.example}" -f "$ROOT_DIR/compose.prod.yml" -f "$ROOT_DIR/compose.tools.yml" config --quiet
for file in api/Dockerfile api/Dockerfile.dev frontend/Dockerfile frontend/Dockerfile.dev migrations/Dockerfile migrations/Dockerfile.dev tools/storage-explorer/Dockerfile tools/netron/Dockerfile tools/browser-tests/Dockerfile; do
  context="${file%/*}"
  docker build --check -f "$ROOT_DIR/$file" "$ROOT_DIR/$context"
done

