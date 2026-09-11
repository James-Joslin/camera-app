#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CHECKPOINT="best_model_fp32.pth"
RELEASE_ID="v$(date -u +%Y-%m-%dT%H%M%SZ)"
INPUT_HEIGHT=360
INPUT_WIDTH=640
RELEASE_STATUS="${MODEL_RELEASE_STATUS:-experimental}"
MIN_MAP_50="${MIN_MAP_50:-0.25}"
MIN_MAP_50_95="${MIN_MAP_50_95:-0.10}"
MIN_RECALL_FPPI="${MIN_RECALL_FPPI:-0.20}"
CAMERA_METRICS="${CAMERA_DOMAIN_METRICS:-}"
BENCHMARK_SCORE_THRESHOLD="${BENCHMARK_SCORE_THRESHOLD:-0.5}"
PRE_NMS_TOPK="${PRE_NMS_TOPK:-1000}"
CALIBRATION_SAMPLES=300
VALIDATION_SAMPLES=500
MAX_ACCURACY_DROP=0.01
BENCHMARK_ITERATIONS=100
THREADS=1
MODEL_CONTAINER="${AZURITE_MODEL_CONTAINER:-computer-vision-models}"
REMOTE_ROOT="person_detector_ssd/releases"
CURRENT_POINTER="${AZURITE_MODEL_CURRENT_POINTER:-person_detector_ssd/current.json}"
RELEASE_ROOT="${PERSON_DETECTOR_RELEASE_ROOT:-$SCRIPT_DIR/releases}"
EVALUATION_ROOT="${PERSON_DETECTOR_EVALUATION_ROOT:-$SCRIPT_DIR/release_evaluations}"
ACTIVE_MODEL_DIR="${PERSON_DETECTOR_ACTIVE_MODEL_DIR:-$SCRIPT_DIR/optimized}"
OFFICIAL_DIR="${CITYPERSONS_OFFICIAL_DIR:-$SCRIPT_DIR/.citypersons-official}"

usage() {
    cat <<'EOF'
Usage: ./releasePersonDetector.sh [options]

Evaluate the FP32 detector, create and accuracy-check FP16/INT8 variants,
publish an immutable release to Azurite, and promote accepted models locally.

Options:
  --release-id ID              Immutable release name (default: UTC timestamp)
  --checkpoint PATH            PyTorch checkpoint (default: best_model_fp32.pth)
  --input-height PIXELS        Model canvas height (default: 360)
  --input-width PIXELS         Model canvas width (default: 640; must exceed height)
  --release-status STATUS      experimental or production (default: experimental)
  --min-map-50 VALUE           Production minimum mAP@0.50 (default: 0.25)
  --min-map-50-95 VALUE        Production minimum mAP@0.50:0.95 (default: 0.10)
  --min-recall-fppi VALUE      Production minimum recall at FPPI 0.10 (default: 0.20)
  --camera-metrics PATH        Representative camera-domain metrics JSON
  --benchmark-threshold VALUE Production-like benchmark score threshold (default: 0.5)
  --pre-nms-topk COUNT         Candidates retained before NMS (default: 1000)
  --calibration-samples COUNT  INT8 calibration records (default: 300)
  --validation-samples COUNT   Optimization validation records (default: 500)
  --max-accuracy-drop VALUE    Maximum absolute INT8 AP drop (default: 0.01)
  --benchmark-iterations COUNT Benchmark iterations (default: 100)
  --threads COUNT              OpenVINO inference threads (default: 1)
  --model-container NAME       Azurite container (default: computer-vision-models)
  --remote-root PREFIX         Blob prefix before the release ID
  --current-pointer BLOB       Mutable pointer published after verification
  -h, --help                   Show this help
EOF
}

require_value() {
    if [[ $# -lt 2 || -z "$2" ]]; then
        echo "Missing value for $1" >&2
        usage >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --release-id)
            require_value "$@"
            RELEASE_ID="$2"
            shift 2
            ;;
        --checkpoint)
            require_value "$@"
            CHECKPOINT="$2"
            shift 2
            ;;
        --input-height)
            require_value "$@"
            INPUT_HEIGHT="$2"
            shift 2
            ;;
        --input-width)
            require_value "$@"
            INPUT_WIDTH="$2"
            shift 2
            ;;
        --release-status)
            require_value "$@"
            RELEASE_STATUS="$2"
            shift 2
            ;;
        --min-map-50)
            require_value "$@"
            MIN_MAP_50="$2"
            shift 2
            ;;
        --min-map-50-95)
            require_value "$@"
            MIN_MAP_50_95="$2"
            shift 2
            ;;
        --min-recall-fppi)
            require_value "$@"
            MIN_RECALL_FPPI="$2"
            shift 2
            ;;
        --camera-metrics)
            require_value "$@"
            CAMERA_METRICS="$2"
            shift 2
            ;;
        --benchmark-threshold)
            require_value "$@"
            BENCHMARK_SCORE_THRESHOLD="$2"
            shift 2
            ;;
        --pre-nms-topk)
            require_value "$@"
            PRE_NMS_TOPK="$2"
            shift 2
            ;;
        --calibration-samples)
            require_value "$@"
            CALIBRATION_SAMPLES="$2"
            shift 2
            ;;
        --validation-samples)
            require_value "$@"
            VALIDATION_SAMPLES="$2"
            shift 2
            ;;
        --max-accuracy-drop)
            require_value "$@"
            MAX_ACCURACY_DROP="$2"
            shift 2
            ;;
        --benchmark-iterations)
            require_value "$@"
            BENCHMARK_ITERATIONS="$2"
            shift 2
            ;;
        --threads)
            require_value "$@"
            THREADS="$2"
            shift 2
            ;;
        --model-container)
            require_value "$@"
            MODEL_CONTAINER="$2"
            shift 2
            ;;
        --remote-root)
            require_value "$@"
            REMOTE_ROOT="$2"
            shift 2
            ;;
        --current-pointer)
            require_value "$@"
            CURRENT_POINTER="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! "$RELEASE_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "Invalid release ID: $RELEASE_ID" >&2
    exit 2
fi
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    exit 2
fi
if [[ ! "$INPUT_HEIGHT" =~ ^[1-9][0-9]*$ || ! "$INPUT_WIDTH" =~ ^[1-9][0-9]*$ ]]; then
    echo "Input height and width must be positive integers" >&2
    exit 2
fi
if (( INPUT_WIDTH <= INPUT_HEIGHT )); then
    echo "Input width must be greater than input height" >&2
    exit 2
fi
if [[ "$RELEASE_STATUS" != "experimental" && "$RELEASE_STATUS" != "production" ]]; then
    echo "Release status must be experimental or production" >&2
    exit 2
fi
if [[ "$RELEASE_STATUS" == "production" && -z "$CAMERA_METRICS" ]]; then
    echo "Production release requires --camera-metrics" >&2
    exit 2
fi
if [[ -n "$CAMERA_METRICS" && ! -f "$CAMERA_METRICS" ]]; then
    echo "Camera metrics not found: $CAMERA_METRICS" >&2
    exit 2
fi

RELEASE_DIR="${RELEASE_ROOT%/}/$RELEASE_ID"
MODELS_DIR="$RELEASE_DIR/models"
EVALUATION_DIR="${EVALUATION_ROOT%/}/$RELEASE_ID/fp32"
EVALUATION_REPORT_DIR="$RELEASE_DIR/evaluation/fp32"
REMOTE_PREFIX="${REMOTE_ROOT%/}/$RELEASE_ID"

if [[ -e "$RELEASE_DIR" ]]; then
    echo "Local release already exists: $RELEASE_DIR" >&2
    echo "Choose a new --release-id; releases are immutable." >&2
    exit 2
fi

mkdir -p "$MODELS_DIR" "$EVALUATION_DIR" "$EVALUATION_REPORT_DIR" "$RELEASE_DIR/checkpoint"

echo "Creating FP32, FP16, and INT8 variants for release $RELEASE_ID..."
OPTIMIZATION_ARGS=(
    --checkpoint "$CHECKPOINT"
    --output-dir "$MODELS_DIR"
    --input-height "$INPUT_HEIGHT"
    --input-width "$INPUT_WIDTH"
    --calibration-samples "$CALIBRATION_SAMPLES"
    --validation-samples "$VALIDATION_SAMPLES"
    --selection-seed 1337
    --max-accuracy-drop "$MAX_ACCURACY_DROP"
    --evaluation-score-threshold 0.01
    --benchmark-score-threshold "$BENCHMARK_SCORE_THRESHOLD"
    --nms-threshold 0.5
    --pre-nms-topk "$PRE_NMS_TOPK"
    --benchmark-iterations "$BENCHMARK_ITERATIONS"
    --threads "$THREADS"
    --release-status "$RELEASE_STATUS"
    --min-map-50 "$MIN_MAP_50"
    --min-map-50-95 "$MIN_MAP_50_95"
    --min-recall-fppi "$MIN_RECALL_FPPI"
)
if [[ -n "$CAMERA_METRICS" ]]; then
    OPTIMIZATION_ARGS+=(--camera-metrics "$CAMERA_METRICS")
fi
python -m person_detection.optimization.pipeline "${OPTIMIZATION_ARGS[@]}"

if [[ ! -d "$OFFICIAL_DIR/evaluation/eval_script" ]]; then
    echo "Downloading the checksum-pinned official CityPersons evaluator..."
    python -m scripts.data.download_citypersons_annotations --output "$OFFICIAL_DIR"
fi

echo "Running complete FP32 project and official CityPersons evaluation..."
python -m person_detection.evaluation.inference \
    --model-path "$MODELS_DIR/person_detector_fp32.xml" \
    --model-type openvino \
    --input-height "$INPUT_HEIGHT" \
    --input-width "$INPUT_WIDTH" \
    --pre-nms-topk "$PRE_NMS_TOPK" \
    --coco-map \
    --map-threshold 0.01 \
    --nms-threshold 0.5 \
    --recall-fppi 0.1 \
    --max-images 100000 \
    --official-evaluator-dir "$OFFICIAL_DIR/evaluation/eval_script" \
    --output-dir "$EVALUATION_DIR"

cp "$CHECKPOINT" "$RELEASE_DIR/checkpoint/best_model_fp32.pth"
for report in \
    "$EVALUATION_DIR/evaluation_metrics.json" \
    "$EVALUATION_DIR/map_results.txt" \
    "$EVALUATION_DIR"/citypersons_official_*.json \
    "$EVALUATION_DIR"/citypersons_official_*.txt; do
    if [[ -f "$report" ]]; then
        cp "$report" "$EVALUATION_REPORT_DIR/"
    fi
done

echo "Publishing and checksum-verifying the immutable Azurite release..."
python -m scripts.models.publish_release \
    --release-dir "$RELEASE_DIR" \
    --release-id "$RELEASE_ID" \
    --container "$MODEL_CONTAINER" \
    --prefix "$REMOTE_PREFIX" \
    --current-pointer "$CURRENT_POINTER"

mkdir -p "$ACTIVE_MODEL_DIR"
cp "$MODELS_DIR"/person_detector_*.xml "$ACTIVE_MODEL_DIR/"
cp "$MODELS_DIR"/person_detector_*.bin "$ACTIVE_MODEL_DIR/"
cp "$MODELS_DIR/calibration_manifest.json" "$ACTIVE_MODEL_DIR/"
cp "$MODELS_DIR/optimization_report.json" "$ACTIVE_MODEL_DIR/"
cp "$RELEASE_DIR/release_manifest.json" "$ACTIVE_MODEL_DIR/"

echo
echo "Release complete ($RELEASE_STATUS)."
echo "  Local release: $RELEASE_DIR"
echo "  Evaluation previews: $EVALUATION_DIR"
echo "  Active local models: $ACTIVE_MODEL_DIR"
echo "  Azurite release: $MODEL_CONTAINER/$REMOTE_PREFIX"
echo "  Current release pointer: $MODEL_CONTAINER/$CURRENT_POINTER"
echo "  Active XML after FastAPI restart: /models/person_detector_int8.xml"
