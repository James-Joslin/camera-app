#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CHECKPOINT="best_model_fp32.pth"
RELEASE_ID="v$(date -u +%Y-%m-%dT%H%M%SZ)"
INPUT_SIZE=480
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
  --input-size PIXELS          Model input size (default: 480)
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
        --input-size)
            require_value "$@"
            INPUT_SIZE="$2"
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
python -m person_detection.optimization.pipeline \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$MODELS_DIR" \
    --input-size "$INPUT_SIZE" \
    --calibration-samples "$CALIBRATION_SAMPLES" \
    --validation-samples "$VALIDATION_SAMPLES" \
    --selection-seed 1337 \
    --max-accuracy-drop "$MAX_ACCURACY_DROP" \
    --score-threshold 0.01 \
    --nms-threshold 0.5 \
    --benchmark-iterations "$BENCHMARK_ITERATIONS" \
    --threads "$THREADS"

if [[ ! -d "$OFFICIAL_DIR/evaluation/eval_script" ]]; then
    echo "Downloading the checksum-pinned official CityPersons evaluator..."
    python -m scripts.data.download_citypersons_annotations --output "$OFFICIAL_DIR"
fi

echo "Running complete FP32 project and official CityPersons evaluation..."
python -m person_detection.evaluation.inference \
    --model-path "$MODELS_DIR/person_detector_fp32.xml" \
    --model-type openvino \
    --input-size "$INPUT_SIZE" \
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
echo "Release complete."
echo "  Local release: $RELEASE_DIR"
echo "  Evaluation previews: $EVALUATION_DIR"
echo "  Active local models: $ACTIVE_MODEL_DIR"
echo "  Azurite release: $MODEL_CONTAINER/$REMOTE_PREFIX"
echo "  Current release pointer: $MODEL_CONTAINER/$CURRENT_POINTER"
echo "  Active XML after FastAPI restart: /models/person_detector_int8.xml"
