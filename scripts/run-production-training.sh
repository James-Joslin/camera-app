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
"${COMPOSE[@]}" up -d --no-deps --force-recreate training-job training-tensorboard

echo "Production training started in the background."
echo "TensorBoard: http://${TENSORBOARD_BIND_ADDRESS:-127.0.0.1}:${TENSORBOARD_HOST_PORT:-6006}"
echo "Status: docker compose -f docker-compose.yml -f compose.training.yml -f compose.training.prod.yml ps -a training-job"
echo "Logs:   docker compose -f docker-compose.yml -f compose.training.yml -f compose.training.prod.yml logs -f training-job"
