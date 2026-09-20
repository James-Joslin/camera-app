#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_ROOT="${TRAINING_STATE_ROOT:-$SCRIPT_DIR/output}"
DEFAULT_STAMP="$(date -u +%Y-%m-%dT%H%M%SZ)"
RUN_ID="${TRAINING_RUN_ID:-v$DEFAULT_STAMP}"
RELEASE_ID="${MODEL_RELEASE_ID:-$RUN_ID}"
DATA_CONTAINER="${AZURITE_DATA_CONTAINER:-computer-vision-data}"
MODEL_CONTAINER="${AZURITE_MODEL_CONTAINER:-computer-vision-models}"
MODEL_REMOTE_ROOT="${AZURITE_MODEL_REMOTE_ROOT:-person_detector_ssd/releases}"
MODEL_CURRENT_POINTER="${AZURITE_MODEL_CURRENT_POINTER:-person_detector_ssd/current.json}"
EPOCHS="${TRAINING_EPOCHS:-60}"
BATCH_SIZE="${TRAINING_BATCH_SIZE:-32}"
NUM_WORKERS="${TRAINING_NUM_WORKERS:-1}"
INPUT_HEIGHT="${TRAINING_INPUT_HEIGHT:-360}"
INPUT_WIDTH="${TRAINING_INPUT_WIDTH:-640}"
RELEASE_STATUS="${MODEL_RELEASE_STATUS:-experimental}"
MIN_MAP_50="${MIN_MAP_50:-0.25}"
MIN_MAP_50_95="${MIN_MAP_50_95:-0.10}"
MIN_RECALL_FPPI="${MIN_RECALL_FPPI:-0.20}"
CAMERA_METRICS="${CAMERA_DOMAIN_METRICS:-}"
BENCHMARK_SCORE_THRESHOLD="${BENCHMARK_SCORE_THRESHOLD:-0.5}"
PRE_NMS_TOPK="${PRE_NMS_TOPK:-1000}"
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
  3. Train the shared-head, anchor-free person detector with visible-box supervision
     and crowd-repulsion losses together. Keep the checkpoint with the highest
     validation detection accuracy (AP50:95).
  4. Evaluate the checkpoint, export FP32/FP16, calibrate INT8, verify all six
     XML/BIN files before benchmarking, and evaluate each exported precision.
  5. Publish a checksum-verified immutable model release and current pointer.

Configuration is supplied through environment variables. Common settings:
  TRAINING_BACKBONE           mobilenetv3_small (default) or mobilenetv4_conv_small
  TRAINING_USE_PAN            true (default); lightweight bottom-up neck
  TRAINING_REGRESSION_DEPTH   2 (default); 1 restores the original clean regression tower
  TRAINING_USE_STRIDE4        false (default); true opts into stride-4 clean_ltrb
  TRAINING_MODEL_VARIANT      clean_ltrb (shared-head, anchor-free; default), clean_anchor, or anchor
  TRAINING_VISIBLE_LOSS_WEIGHT  Training-only visible boxes (default: 0.25; set 0 to disable)
  TRAINING_REPGT_LOSS_WEIGHT    Neighbor-GT repulsion (default: 0.05; set 0 to disable)
  TRAINING_REPBOX_LOSS_WEIGHT   Cross-person prediction repulsion (default: 0.01; set 0 to disable)
  TRAINING_AUXILIARY_RAMP_EPOCHS  Zero to full occlusion-loss weight over this many epochs (default: 5)
  TRAINING_REPGT_SIGMA          RepGT smoothing threshold (default: 0.5)
  TRAINING_REPBOX_SIGMA         RepBox smoothing threshold (default: 0)
  TRAINING_REPBOX_PREDICTIONS_PER_GT  Maximum predictions per person for RepBox (default: 4)
  TRAINING_REPULSION_CHUNK_SIZE  Pairwise geometry chunk size (default: 64)
  TRAINING_CHECKPOINT_SELECTION ap (default), detection_loss, or final
  TRAINING_EPOCHS             Training epochs (default: 60)
  TRAINING_BATCH_SIZE         Batch size (default: 32)
  TRAINING_NUM_WORKERS        DataLoader workers (default: 1)
  TRAINING_INPUT_HEIGHT       Model canvas height (default: 360)
  TRAINING_INPUT_WIDTH        Model canvas width (default: 640)
  TRAINING_RUN_ID             Persistent run directory name; reuse it to resume
  TRAINING_AP_EVERY_N_EPOCHS   Validation AP cadence (default: 1; AP selection requires it)
  TRAINING_AP_SCORE_THRESHOLD  Low score floor used to build PR curves (default: 0.01)
  TRAINING_TENSORBOARD_ENABLED Write TensorBoard events (default: true)
  TRAINING_TENSORBOARD_LOG_DIR Run-relative event directory (default: tensorboard)
  MODEL_RELEASE_ID            Immutable model release ID
  CALIBRATION_SAMPLES         INT8 calibration records (default: 300)
  VALIDATION_SAMPLES          Optimization validation records (default: 500)
  MAX_ACCURACY_DROP           Maximum absolute INT8 AP drop (default: 0.01)
  MODEL_RELEASE_STATUS        experimental or production (default: experimental)
  MIN_MAP_50                  Production minimum mAP@0.50 (default: 0.25)
  MIN_MAP_50_95               Production minimum mAP@0.50:0.95 (default: 0.10)
  MIN_RECALL_FPPI             Production recall at FPPI 0.10 (default: 0.20)
  CAMERA_DOMAIN_METRICS       Required camera metrics JSON for production release
  BENCHMARK_SCORE_THRESHOLD   Production-like benchmark threshold (default: 0.5)
  PRE_NMS_TOPK                Candidates retained before NMS (default: 1000)
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
for setting in "$EPOCHS" "$BATCH_SIZE" "$INPUT_HEIGHT" "$INPUT_WIDTH" "$CALIBRATION_SAMPLES" "$VALIDATION_SAMPLES" "$PRE_NMS_TOPK" \
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
if (( INPUT_WIDTH <= INPUT_HEIGHT )); then
    echo "TRAINING_INPUT_WIDTH must be greater than TRAINING_INPUT_HEIGHT" >&2
    exit 2
fi
if [[ "$RELEASE_STATUS" != "experimental" && "$RELEASE_STATUS" != "production" ]]; then
    echo "MODEL_RELEASE_STATUS must be experimental or production" >&2
    exit 2
fi
if [[ "$RELEASE_STATUS" == "production" && -z "$CAMERA_METRICS" ]]; then
    echo "Production release requires CAMERA_DOMAIN_METRICS" >&2
    exit 2
fi

export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export ENABLE_QUANTIZATION=false
# Visibility supervision and crowd repulsion are enabled in normal production training.
export TRAINING_USE_PAN="${TRAINING_USE_PAN:-true}"
export TRAINING_REGRESSION_DEPTH="${TRAINING_REGRESSION_DEPTH:-2}"
export TRAINING_EPOCHS="$EPOCHS"
export TRAINING_BATCH_SIZE="$BATCH_SIZE"
export TRAINING_NUM_WORKERS="$NUM_WORKERS"
export TRAINING_EXPORT_OPENVINO=false
export TRAINING_VISIBLE_LOSS_WEIGHT="${TRAINING_VISIBLE_LOSS_WEIGHT:-0.25}"
export TRAINING_REPGT_LOSS_WEIGHT="${TRAINING_REPGT_LOSS_WEIGHT:-0.05}"
export TRAINING_REPBOX_LOSS_WEIGHT="${TRAINING_REPBOX_LOSS_WEIGHT:-0.01}"
export TRAINING_CHECKPOINT_SELECTION="${TRAINING_CHECKPOINT_SELECTION:-ap}"
export TRAINING_AP_EVERY_N_EPOCHS="${TRAINING_AP_EVERY_N_EPOCHS:-1}"
export TRAINING_INPUT_HEIGHT="$INPUT_HEIGHT"
export TRAINING_INPUT_WIDTH="$INPUT_WIDTH"

RUN_DIR="${STATE_ROOT%/}/runs/$RUN_ID"
CHECKPOINT_DIR="$RUN_DIR/checkpoints"
LOG_DIR="$RUN_DIR/logs"
export TRAINING_TENSORBOARD_LOG_DIR="$RUN_DIR/tensorboard"
DATASET_VALIDATION_DIR="$RUN_DIR/dataset-validation"
RELEASE_ROOT="${STATE_ROOT%/}/releases"
ACTIVE_MODEL_DIR="${STATE_ROOT%/}/active-models"
OFFICIAL_DIR="${STATE_ROOT%/}/cache/citypersons-official"
# Existing flat runs must not silently restart when the layout changes.
if [[ -f "$RUN_DIR/last_training_checkpoint.pth" || -f "$RUN_DIR/best_model_fp32.pth" ]]; then
    echo "Old flat run layout found at $RUN_DIR. Use a new TRAINING_RUN_ID; existing checkpoints are preserved." >&2
    exit 2
fi
mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR" "$DATASET_VALIDATION_DIR"

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
if validate_current_dataset 2>&1 | tee "$LOG_DIR/dataset-validation.log"; then
    echo "CityPersons is complete and valid; using the published version."
else
    DATASET_VERSION="${CITYPERSONS_DATASET_VERSION:-v$(date -u +%F).production$(date -u +%Y%m%dT%H%M%SZ)}"
    DATASET_WORK_DIR="${STATE_ROOT%/}/cache/dataset-bootstrap/$DATASET_VERSION"
    echo "No complete, valid current CityPersons dataset was found."
    echo "Bootstrapping immutable dataset version: $DATASET_VERSION"
    CITYPERSONS_DATASET_VERSION="$DATASET_VERSION" \
    CITYPERSONS_WORK_DIR="$DATASET_WORK_DIR" \
    CITYPERSONS_PREVIEW_DIR="$DATASET_VALIDATION_DIR/bootstrap" \
    CITYPERSONS_PREVIEW_COUNT="$PREVIEW_COUNT" \
    CITYPERSONS_VERIFY_REMOTE_CHECKSUMS=true \
        "$SCRIPT_DIR/getdata.sh" 2>&1 | tee "$LOG_DIR/dataset-bootstrap.log"

    echo "Re-validating the newly published CityPersons version..."
    validate_current_dataset 2>&1 | tee "$LOG_DIR/dataset-revalidation.log"
fi

echo "Starting training and AP-based checkpoint selection..."
(
    cd "$CHECKPOINT_DIR"
    python -m person_detection.training.pipeline
) 2>&1 | tee "$LOG_DIR/training.log"

for artifact in best_model_fp32.pth last_training_checkpoint.pth; do
    if [[ ! -s "$CHECKPOINT_DIR/$artifact" ]]; then
        echo "Training did not create a non-empty $artifact" >&2
        exit 1
    fi
done

echo "Starting optimization, evaluation, release verification, and publication..."
RELEASE_ARGS=(
    --release-id "$RELEASE_ID"
    --checkpoint "$CHECKPOINT_DIR/best_model_fp32.pth"
    --input-height "$INPUT_HEIGHT"
    --input-width "$INPUT_WIDTH"
    --release-status "$RELEASE_STATUS"
    --min-map-50 "$MIN_MAP_50"
    --min-map-50-95 "$MIN_MAP_50_95"
    --min-recall-fppi "$MIN_RECALL_FPPI"
    --benchmark-threshold "$BENCHMARK_SCORE_THRESHOLD"
    --pre-nms-topk "$PRE_NMS_TOPK"
    --calibration-samples "$CALIBRATION_SAMPLES"
    --validation-samples "$VALIDATION_SAMPLES"
    --max-accuracy-drop "$MAX_ACCURACY_DROP"
    --benchmark-iterations "$BENCHMARK_ITERATIONS"
    --threads "$OPENVINO_THREADS"
    --model-container "$MODEL_CONTAINER"
    --remote-root "$MODEL_REMOTE_ROOT"
    --current-pointer "$MODEL_CURRENT_POINTER"
)
if [[ -n "$CAMERA_METRICS" ]]; then
    RELEASE_ARGS+=(--camera-metrics "$CAMERA_METRICS")
fi
PERSON_DETECTOR_RELEASE_ROOT="$RELEASE_ROOT" \
PERSON_DETECTOR_ACTIVE_MODEL_DIR="$ACTIVE_MODEL_DIR" \
CITYPERSONS_OFFICIAL_DIR="$OFFICIAL_DIR" \
    "$SCRIPT_DIR/releasePersonDetector.sh" "${RELEASE_ARGS[@]}" \
        2>&1 | tee "$LOG_DIR/release.log"

echo "============================================================"
echo "Production training job completed successfully."
echo "Immutable models: $MODEL_CONTAINER/${MODEL_REMOTE_ROOT%/}/$RELEASE_ID/"
echo "Current pointer: $MODEL_CONTAINER/$MODEL_CURRENT_POINTER"
echo "Persistent state: $RUN_DIR"
echo "============================================================"
