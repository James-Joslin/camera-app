#!/usr/bin/env bash
set -euo pipefail

AZURITE_CONTAINER="${AZURITE_DATA_CONTAINER:-computer-vision-data}"
if [[ -n "${CITYPERSONS_WORK_DIR:-}" ]]; then
    TEMP_DIR="$CITYPERSONS_WORK_DIR"
    mkdir -p "$TEMP_DIR"
    REMOVE_ON_SUCCESS=false
else
    TEMP_DIR=$(mktemp -d)
    REMOVE_ON_SUCCESS=true
fi

cleanup() {
    status=$?
    if [[ "$REMOVE_ON_SUCCESS" == true && $status -eq 0 ]]; then
        rm -rf "$TEMP_DIR"
    elif [[ "$REMOVE_ON_SUCCESS" == true ]]; then
        echo "Download and extracted files were preserved after the failure: $TEMP_DIR" >&2
        echo "Resume with: CITYPERSONS_WORK_DIR=$TEMP_DIR CITYPERSONS_RESUME_UPLOAD=true ./getCityPersons.sh" >&2
    else
        echo "Keeping CityPersons work directory: $TEMP_DIR"
    fi
}
trap cleanup EXIT
echo "Created temporary directory: $TEMP_DIR"

OUTER_ARCHIVE="$TEMP_DIR/citypersons.zip"
if [[ -f "$OUTER_ARCHIVE" ]]; then
    echo "Using existing download: $OUTER_ARCHIVE"
else
    echo "Downloading CityPersons dataset..."
    kaggle datasets download hakurei/citypersons -p "$TEMP_DIR"
fi

INNER_ARCHIVE="$TEMP_DIR/CityPersons"
if [[ -f "$INNER_ARCHIVE" ]]; then
    echo "Using existing inner archive: $INNER_ARCHIVE"
else
    echo "Extracting outer archive..."
    7z x -y "$OUTER_ARCHIVE" -o"$TEMP_DIR"
fi

if [[ ! -f "$INNER_ARCHIVE" ]]; then
    echo "Could not find inner CityPersons archive: $INNER_ARCHIVE" >&2
    exit 1
fi

EXTRACT_DIR="$TEMP_DIR/CityPersons-extracted"
EXTRACT_MARKER="$EXTRACT_DIR/.extraction-complete"
if [[ -f "$EXTRACT_MARKER" ]]; then
    echo "Using existing extracted dataset: $EXTRACT_DIR"
else
    echo "Extracting inner archive..."
    mkdir -p "$EXTRACT_DIR"
    7z x -y "$INNER_ARCHIVE" -o"$EXTRACT_DIR"
    touch "$EXTRACT_MARKER"
fi

find_image_tree() {
    local directory_name candidate
    for directory_name in leftImg8bit images; do
        while IFS= read -r candidate; do
            [[ -d "$candidate/train" ]] || continue
            if find "$candidate/train" -type f \
                \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) \
                -print -quit | grep -q .; then
                printf '%s\n' "$candidate"
                return 0
            fi
        done < <(find "$EXTRACT_DIR" -type d -name "$directory_name")
    done
    return 0
}

IMAGE_DIR=$(find_image_tree)
if [[ -z "$IMAGE_DIR" ]]; then
    echo "Could not locate a populated CityPersons image tree with a train split." >&2
    exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DATASET_VERSION="${CITYPERSONS_DATASET_VERSION:-v$(date -u +%F)}"
if [[ ! "$DATASET_VERSION" =~ ^v[0-9]{4}-[0-9]{2}-[0-9]{2}([._-][A-Za-z0-9]+)*$ ]]; then
    echo "Invalid CITYPERSONS_DATASET_VERSION: $DATASET_VERSION" >&2
    exit 1
fi
VERSION_PREFIX="datasets/citypersons/$DATASET_VERSION"
ANNOTATION_DIR="$TEMP_DIR/official-annotations"
BUILD_DIR="$TEMP_DIR/version-build"

echo "Downloading pinned official CityPersons annotations..."
python3 "$SCRIPT_DIR/download_citypersons_annotations.py" --output "$ANNOTATION_DIR"

echo "Building immutable dataset metadata for $VERSION_PREFIX..."
if [[ -f "$BUILD_DIR/manifest.json" ]]; then
    echo "Using existing version metadata: $BUILD_DIR"
else
    if [[ -d "$BUILD_DIR" ]]; then
        rm -rf "$BUILD_DIR"
    fi
    python3 "$SCRIPT_DIR/build_citypersons_version.py" \
        --images "$IMAGE_DIR" \
        --annotations "$ANNOTATION_DIR" \
        --source-archive "$OUTER_ARCHIVE" \
        --output "$BUILD_DIR" \
        --version "$DATASET_VERSION" \
        --version-prefix "$VERSION_PREFIX"
fi

RESET_ARGS=()
if [[ "${CITYPERSONS_RESET_CONTAINER:-false}" == true ]]; then
    RESET_ARGS+=(--reset-container)
fi
IMMUTABLE_ARGS=(--require-empty-prefix "$VERSION_PREFIX")
if [[ "${CITYPERSONS_RESUME_UPLOAD:-false}" == true ]]; then
    IMMUTABLE_ARGS=()
fi

echo "Uploading versioned images from $IMAGE_DIR..."
python3 "$SCRIPT_DIR/upload_to_azurite.py" \
    --source "$IMAGE_DIR" \
    --container "$AZURITE_CONTAINER" \
    --prefix "$VERSION_PREFIX/images" \
    "${IMMUTABLE_ARGS[@]}" \
    "${RESET_ARGS[@]}" \
    --include '*.png' \
    --include '*.jpg' \
    --include '*.jpeg' \
    --exclude '*(1).*'

echo "Uploading canonical annotations, generated labels, and manifests..."
python3 "$SCRIPT_DIR/upload_to_azurite.py" \
    --source "$BUILD_DIR" \
    --container "$AZURITE_CONTAINER" \
    --prefix "$VERSION_PREFIX"

PREVIEW_DIR="${CITYPERSONS_PREVIEW_DIR:-$SCRIPT_DIR/validation_preview/$DATASET_VERSION}"
PREVIEW_COUNT="${CITYPERSONS_PREVIEW_COUNT:-12}"
CHECKSUM_ARGS=(--verify-checksums)
if [[ "${CITYPERSONS_VERIFY_REMOTE_CHECKSUMS:-true}" != true ]]; then
    CHECKSUM_ARGS=(--no-verify-checksums)
fi

echo "Validating the complete uploaded version and rendering $PREVIEW_COUNT previews..."
python3 "$SCRIPT_DIR/validate_citypersons_azurite.py" \
    --container "$AZURITE_CONTAINER" \
    --dataset-prefix "$VERSION_PREFIX" \
    --preview-dir "$PREVIEW_DIR" \
    --preview-count "$PREVIEW_COUNT" \
    "${CHECKSUM_ARGS[@]}" \
    --publish-current

echo "All done! Immutable dataset version uploaded, validated, and published."
echo "Current version: $VERSION_PREFIX"
echo "Visual validation contact sheet: $PREVIEW_DIR/contact_sheet.jpg"
