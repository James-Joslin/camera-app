#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
: "${CAMERA_BENCH_STREAM_URL:?Set a same-origin HLS URL, for example /streams/CAMERA_ID/index.m3u8}"
: "${CAMERA_BENCH_HARDWARE_NOTES:?Describe hardware, model settings, idle conditions, and whether the footage is a replay}"
OUTPUT_DIR="${CAMERA_BENCH_OUTPUT_DIR:-/tmp/camera-browser-benchmark}"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
docker build -t camera-software-browser-tests "$ROOT_DIR/tools/browser-tests"
docker run --rm --network host \
  -e CAMERA_TEST_URL="${CAMERA_TEST_URL:-http://localhost:3101}" \
  -e CAMERA_BENCH_STREAM_URL \
  -e CAMERA_BENCH_HARDWARE_NOTES \
  -e CAMERA_BENCH_CAMERAS="${CAMERA_BENCH_CAMERAS:-4}" \
  -e CAMERA_BENCH_SECONDS="${CAMERA_BENCH_SECONDS:-30}" \
  -e CAMERA_BENCH_OUTPUT=/artifacts/browser-benchmark.json \
  -v "$ROOT_DIR/frontend/tests:/test/tests:ro" \
  -v "$OUTPUT_DIR:/artifacts" \
  camera-software-browser-tests node /test/tests/browser-benchmark.cjs
