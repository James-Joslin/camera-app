#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BASE_COMPOSE_FILE="$ROOT_DIR/docker-compose.yml"
DEV_TRAINING_COMPOSE_FILE="$ROOT_DIR/compose.training.yml"
TRAINING_COMPOSE_FILE="$ROOT_DIR/compose.training.prod.yml"
COMPOSE=(
    docker compose
    -f "$BASE_COMPOSE_FILE"
    -f "$DEV_TRAINING_COMPOSE_FILE"
    -f "$TRAINING_COMPOSE_FILE"
)

"${COMPOSE[@]}" up -d --wait azurite
"${COMPOSE[@]}" build training-job
"${COMPOSE[@]}" run --rm --no-deps training-job
