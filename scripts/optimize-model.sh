#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
docker build -t camera-software-model-tools -f "$ROOT_DIR/project/tools/Dockerfile" "$ROOT_DIR/project"
docker run --rm --user "$(id -u):$(id -g)" -v "$ROOT_DIR/project:/workspace/project" camera-software-model-tools "$@"

