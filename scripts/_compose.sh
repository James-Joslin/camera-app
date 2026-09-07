#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${CAMERA_ENV_FILE:-$ROOT_DIR/.env}"
COMPOSE_FILE="${CAMERA_COMPOSE_FILE:-$ROOT_DIR/compose.dev.yml}"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE. Copy .env.dev.example to .env first." >&2
  exit 1
fi
compose() { docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" "$@"; }

