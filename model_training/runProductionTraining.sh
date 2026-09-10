#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_ROOT="${TRAINING_STATE_ROOT:-/state}"
DEFAULT_STAMP="$(date -u +%Y-%m-%dT%H%M%SZ)"
RUN_ID="${TRAINING_RUN_ID:-v$DEFAULT_STAMP}"
RELEASE_ID="${MODEL_RELEASE_ID:-$RUN_ID}"
DATA_CONTAINER="${AZURITE_DATA_CONTAINER:-computer-vision-data}"
MODEL_CONTAINER="${AZURITE_MODEL_CONTAINER:-computer-vision-models}"
MODEL_REMOTE_ROOT="${AZURITE_MODEL_REMOTE_ROOT:-person_detector_ssd/releases}"
MODEL_CURRENT_POINTER="${AZURITE_MODEL_CURRENT_POINTER:-person_detector_ssd/current.json}"
EPOCHS="${TRAINING_EPOCHS:-100}"
BATCH_SIZE="${TRAINING_BATCH_SIZE:-32}"
NUM_WORKERS="${TRAINING_NUM_WORKERS:-1}"
INPUT_SIZE="${TRAINING_INPUT_SIZE:-480}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-300}"
VALIDATION_SAMPLES="${VALIDATION_SAMPLES:-500}"
MAX_ACCURACY_DROP="${MAX_ACCURACY_DROP:-0.01}"
BENCHMARK_ITERATIONS="${BENCHMARK_ITERATIONS:-100}"
OPENVINO_THREADS="${OPENVINO_THREADS:-1}"
PREVIEW_COUNT="${CITYPERSONS_PREVIEW_COUNT:-1}"

usage() {
    cat <<'EOF'
Usage: ./runProductionTraining.sh

One-shot production job:
  1. Fully validate the current CityPersons dataset in Azurite.
  2. Download, build, upload, validate, and publish it when validation fails.
  3. Train and export FP32.
  4. Export/evaluate FP32 and FP16, calibrate/evaluate INT8, and enforce the
     configured INT8 accuracy gate.
  5. Publish a checksum-verified immutable model release and current pointer.

Configuration is supplied through environment variables. Common settings:
  TRAINING_EPOCHS             Training epochs (default: 100)
  TRAINING_BATCH_SIZE         Batch size (default: 32)
  TRAINING_NUM_WORKERS        DataLoader workers (default: 1)
  TRAINING_INPUT_SIZE         Square model input (default: 480)
  TRAINING_RUN_ID             Persistent run directory name
  MODEL_RELEASE_ID            Immutable model release ID
  CALIBRATION_SAMPLES         INT8 calibration records (default: 300)
  VALIDATION_SAMPLES          Optimization validation records (default: 500)
  MAX_ACCURACY_DROP           Maximum absolute INT8 AP drop (default: 0.01)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
if [[ $# -gt 0 ]]; then
    echo "Unknown argument: $1" >&2
    usage >&2
    exit 2
fi

for identifier in "$RUN_ID" "$RELEASE_ID"; do
    if [[ ! "$identifier" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
        echo "Invalid run or release ID: $identifier" >&2
        exit 2
    fi
done
for setting in "$EPOCHS" "$BATCH_SIZE" "$INPUT_SIZE" "$CALIBRATION_SAMPLES" "$VALIDATION_SAMPLES" \
    "$BENCHMARK_ITERATIONS" "$OPENVINO_THREADS" "$PREVIEW_COUNT"; do
    if [[ ! "$setting" =~ ^[1-9][0-9]*$ ]]; then
        echo "Expected a positive integer setting, got: $setting" >&2
        exit 2
    fi
done
if [[ ! "$NUM_WORKERS" =~ ^[0-9]+$ ]]; then
    echo "TRAINING_NUM_WORKERS must be zero or a positive integer" >&2
    exit 2
fi

export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export ENABLE_QUANTIZATION=false

RUN_DIR="${STATE_ROOT%/}/runs/$RUN_ID"
DATASET_VALIDATION_DIR="${STATE_ROOT%/}/dataset-validation/$RUN_ID"
RELEASE_ROOT="${STATE_ROOT%/}/releases"
RELEASE_EVALUATION_ROOT="${STATE_ROOT%/}/release-evaluations"
ACTIVE_MODEL_DIR="${STATE_ROOT%/}/active-models"
OFFICIAL_DIR="${STATE_ROOT%/}/citypersons-official"
mkdir -p "$RUN_DIR" "$DATASET_VALIDATION_DIR"

validate_current_dataset() {
    python -m scripts.data.validate_citypersons_azurite \
        --container "$DATA_CONTAINER" \
        --preview-dir "$DATASET_VALIDATION_DIR" \
        --preview-count "$PREVIEW_COUNT" \
        --verify-checksums
}

echo "============================================================"
echo "Production person-detector training job"
echo "Run ID: $RUN_ID"
echo "Model release ID: $RELEASE_ID"
echo "Epochs: $EPOCHS"
echo "State: $RUN_DIR"
echo "============================================================"

echo "Checking the current CityPersons dataset in Azurite..."
if validate_current_dataset 2>&1 | tee "$RUN_DIR/dataset-validation.log"; then
    echo "CityPersons is complete and valid; using the published version."
else
    DATASET_VERSION="${CITYPERSONS_DATASET_VERSION:-v$(date -u +%F).production$(date -u +%Y%m%dT%H%M%SZ)}"
    DATASET_WORK_DIR="${STATE_ROOT%/}/dataset-bootstrap/$DATASET_VERSION"
    echo "No complete, valid current CityPersons dataset was found."
    echo "Bootstrapping immutable dataset version: $DATASET_VERSION"
    CITYPERSONS_DATASET_VERSION="$DATASET_VERSION" \
    CITYPERSONS_WORK_DIR="$DATASET_WORK_DIR" \
    CITYPERSONS_PREVIEW_DIR="$DATASET_VALIDATION_DIR/bootstrap" \
    CITYPERSONS_PREVIEW_COUNT="$PREVIEW_COUNT" \
    CITYPERSONS_VERIFY_REMOTE_CHECKSUMS=true \
        "$SCRIPT_DIR/getCityPersons.sh" 2>&1 | tee "$RUN_DIR/dataset-bootstrap.log"

    echo "Re-validating the newly published CityPersons version..."
    validate_current_dataset 2>&1 | tee "$RUN_DIR/dataset-revalidation.log"
fi

echo "Starting FP32 training and OpenVINO export..."
(
    cd "$RUN_DIR"
    python -m person_detection.training.pipeline
) 2>&1 | tee "$RUN_DIR/training.log"

for artifact in best_model_fp32.pth person_detector_fp32.xml person_detector_fp32.bin; do
    if [[ ! -s "$RUN_DIR/$artifact" ]]; then
        echo "Training did not create a non-empty $artifact" >&2
        exit 1
    fi
done

echo "Starting optimization, evaluation, release verification, and publication..."
PERSON_DETECTOR_RELEASE_ROOT="$RELEASE_ROOT" \
PERSON_DETECTOR_EVALUATION_ROOT="$RELEASE_EVALUATION_ROOT" \
PERSON_DETECTOR_ACTIVE_MODEL_DIR="$ACTIVE_MODEL_DIR" \
CITYPERSONS_OFFICIAL_DIR="$OFFICIAL_DIR" \
    "$SCRIPT_DIR/releasePersonDetector.sh" \
        --release-id "$RELEASE_ID" \
        --checkpoint "$RUN_DIR/best_model_fp32.pth" \
        --input-size "$INPUT_SIZE" \
        --calibration-samples "$CALIBRATION_SAMPLES" \
        --validation-samples "$VALIDATION_SAMPLES" \
        --max-accuracy-drop "$MAX_ACCURACY_DROP" \
        --benchmark-iterations "$BENCHMARK_ITERATIONS" \
        --threads "$OPENVINO_THREADS" \
        --model-container "$MODEL_CONTAINER" \
        --remote-root "$MODEL_REMOTE_ROOT" \
        --current-pointer "$MODEL_CURRENT_POINTER" \
        2>&1 | tee "$RUN_DIR/release.log"

echo "============================================================"
echo "Production training job completed successfully."
echo "Immutable models: $MODEL_CONTAINER/${MODEL_REMOTE_ROOT%/}/$RELEASE_ID/"
echo "Current pointer: $MODEL_CONTAINER/$MODEL_CURRENT_POINTER"
echo "Persistent state: $RUN_DIR"
echo "============================================================"
