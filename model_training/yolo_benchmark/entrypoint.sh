#!/usr/bin/env bash
set -euo pipefail
ROOT=/opt/model_training
STATE_ROOT="${YOLO_STATE_ROOT:-/state}"
YOLO_MODEL="${YOLO_MODEL:-yolov8n}"
case "$YOLO_MODEL" in yolov8n|yolo26n) ;; *) echo 'YOLO_MODEL must be yolov8n or yolo26n' >&2; exit 2 ;; esac
export YOLO_MODEL
RUN_ID="${YOLO_RUN_ID:-$YOLO_MODEL-v$(date -u +%Y-%m-%dT%H%M%SZ)}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo 'Invalid YOLO_RUN_ID' >&2; exit 2; }
export YOLO_RUN_DIR="$STATE_ROOT/runs/$RUN_ID"
mkdir -p "$STATE_ROOT/runs"
mkdir "$YOLO_RUN_DIR" # Never overwrite a previous benchmark.
exec > >(tee "$YOLO_RUN_DIR/benchmark.log") 2>&1
validate_data() {
    python -m scripts.data.validate_citypersons_azurite \
        --container "${AZURITE_DATA_CONTAINER:-computer-vision-data}" \
        --preview-dir "$YOLO_RUN_DIR/dataset-validation" --preview-count 1 --verify-checksums
}
echo 'Checking canonical CityPersons data in Azurite...'
if ! validate_data; then
    echo 'Dataset validation failed; running canonical download/build/upload workflow.'
    DATASET_VERSION="${CITYPERSONS_DATASET_VERSION:-v$(date -u +%F).yolo$(date -u +%Y%m%dT%H%M%SZ)}"
    CITYPERSONS_DATASET_VERSION="$DATASET_VERSION" \
    CITYPERSONS_WORK_DIR="$STATE_ROOT/dataset-bootstrap/$DATASET_VERSION" \
    CITYPERSONS_PREVIEW_DIR="$YOLO_RUN_DIR/dataset-validation/bootstrap" \
    CITYPERSONS_PREVIEW_COUNT=1 CITYPERSONS_VERIFY_REMOTE_CHECKSUMS=true \
        "$ROOT/getCityPersons.sh"
    validate_data
fi
echo "Benchmarking pretrained $YOLO_MODEL; no training or model publication."
python -m yolo_benchmark.benchmark
