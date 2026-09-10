#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

docker compose -f "$ROOT_DIR/docker-compose.yml" -f "$ROOT_DIR/compose.training.yml" up -d azurite training

docker compose -f "$ROOT_DIR/docker-compose.yml" -f "$ROOT_DIR/compose.training.yml" \
    exec -T training ./releasePersonDetector.sh "$@"
