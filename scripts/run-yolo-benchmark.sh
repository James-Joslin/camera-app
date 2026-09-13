#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE=(docker compose -f "$ROOT_DIR/docker-compose.yml" -f "$ROOT_DIR/compose.yolo-benchmark.yml")
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    cat <<'EOF'
Usage: ./scripts/run-yolo-benchmark.sh

Check/bootstrap CityPersons in Azurite, download a pretrained YOLO nano model, export
OpenVINO and benchmark person detection at 640x360 (internally padded to 640x384).
No training. Runs in the foreground; Ctrl+C stops the benchmark.

YOLO_MODEL=yolov8n (default) or yolo26n
VALIDATION_SAMPLES=500 CALIBRATION_SAMPLES=300 BENCHMARK_ITERATIONS=100
OPENVINO_THREADS=1 MAX_ACCURACY_DROP=0.01 YOLO_INT8=true YOLO_RUN_ID=<unique ID>
Set YOLO_INT8=false to skip INT8 calibration. KAGGLE_JSON_PATH is needed only
when the canonical dataset needs downloading. Results persist in the separate
yolo-benchmark-state volume under /state/runs/<run ID>/report.json.
EOF
    exit 0
fi
[[ $# == 0 ]] || { echo "Unknown argument: $1" >&2; exit 2; }
"${COMPOSE[@]}" up -d --wait azurite
"${COMPOSE[@]}" build yolo-benchmark
"${COMPOSE[@]}" run --rm --no-deps yolo-benchmark
