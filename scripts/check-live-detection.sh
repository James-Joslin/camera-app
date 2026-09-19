#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# Run against an existing frontend. Camera/stream metadata and inference are
# mocked in the browser; this does not create users or modify camera settings.
docker build -t camera-software-browser-tests "$ROOT_DIR/tools/browser-tests"
docker run --rm --network host \
  -e CAMERA_TEST_URL="${CAMERA_TEST_URL:-http://localhost:3101}" \
  -v "$ROOT_DIR/frontend/tests:/test/tests:ro" \
  camera-software-browser-tests node /test/tests/live-detection.cjs
