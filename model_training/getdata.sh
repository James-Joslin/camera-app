#!/usr/bin/env bash
set -euo pipefail

# Build and publish the pooled CityPersons + CrowdHuman body-detection data.
AZURITE_CONTAINER="${AZURITE_DATA_CONTAINER:-computer-vision-data}"
WORK_DIR_VAR="${DATASET_WORK_DIR:-${CITYPERSONS_WORK_DIR:-}}"
if [[ -n "$WORK_DIR_VAR" ]]; then
    TEMP_DIR="$WORK_DIR_VAR"
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
        echo "Downloaded/extracted files were preserved after the failure: $TEMP_DIR" >&2
        echo "Resume with: DATASET_WORK_DIR=$TEMP_DIR DATASET_RESUME_UPLOAD=true ./getdata.sh" >&2
    else
        echo "Keeping dataset work directory: $TEMP_DIR"
    fi
}
trap cleanup EXIT

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXPLICIT_DATASET_VERSION="${DATASET_VERSION:-${CITYPERSONS_DATASET_VERSION:-}}"
DATASET_VERSION="${DATASET_VERSION:-${CITYPERSONS_DATASET_VERSION:-v$(date -u +%F)}}"
if [[ ! "$DATASET_VERSION" =~ ^v[0-9]{4}-[0-9]{2}-[0-9]{2}([._-][A-Za-z0-9]+)*$ ]]; then
    echo "Invalid DATASET_VERSION: $DATASET_VERSION" >&2
    exit 1
fi
VERSION_PREFIX="datasets/citypersons/$DATASET_VERSION"

PREVIEW_COUNT="${CITYPERSONS_PREVIEW_COUNT:-${DATASET_PREVIEW_COUNT:-12}}"
CHECKSUM_ARGS=(--verify-checksums)
if [[ "${CITYPERSONS_VERIFY_REMOTE_CHECKSUMS:-${DATASET_VERIFY_REMOTE_CHECKSUMS:-true}}" != true ]]; then
    CHECKSUM_ARGS=(--no-verify-checksums)
fi

# Reuse a complete pooled version already published in Azurite.  An explicit
# version requests a new immutable build; the default invocation reuses the
# current valid version and avoids downloading the multi-gigabyte archives.
if [[ -z "$EXPLICIT_DATASET_VERSION" && "${DATASET_REUSE_EXISTING:-true}" == true ]]; then
    if python3 "$SCRIPT_DIR/scripts/data/validate_citypersons_azurite.py" \
        --container "$AZURITE_CONTAINER" --check-only \
        --require-dataset citypersons-crowdhuman "${CHECKSUM_ARGS[@]}"; then
        echo "A complete pooled CityPersons + CrowdHuman dataset already exists in Azurite; skipping downloads."
        exit 0
    fi
    echo "No complete pooled dataset is currently published; starting download/build."
fi

CITYPERSONS_ARCHIVE="$TEMP_DIR/citypersons.zip"
if [[ ! -f "$CITYPERSONS_ARCHIVE" ]]; then
    echo "Downloading CityPersons dataset..."
    kaggle datasets download hakurei/citypersons -p "$TEMP_DIR"
fi
CITYPERSONS_INNER_ARCHIVE="$TEMP_DIR/CityPersons"
if [[ ! -f "$CITYPERSONS_INNER_ARCHIVE" ]]; then
    7z x -y "$CITYPERSONS_ARCHIVE" -o"$TEMP_DIR"
fi
if [[ ! -f "$CITYPERSONS_INNER_ARCHIVE" ]]; then
    echo "Could not find inner CityPersons archive: $CITYPERSONS_INNER_ARCHIVE" >&2
    exit 1
fi
CITYPERSONS_EXTRACT_DIR="$TEMP_DIR/CityPersons-extracted"
if [[ ! -f "$CITYPERSONS_EXTRACT_DIR/.extraction-complete" ]]; then
    mkdir -p "$CITYPERSONS_EXTRACT_DIR"
    7z x -y "$CITYPERSONS_INNER_ARCHIVE" -o"$CITYPERSONS_EXTRACT_DIR"
    touch "$CITYPERSONS_EXTRACT_DIR/.extraction-complete"
fi

find_citypersons_image_tree() {
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
        done < <(find "$CITYPERSONS_EXTRACT_DIR" -type d -name "$directory_name")
    done
}
CITYPERSONS_IMAGE_DIR=$(find_citypersons_image_tree)
if [[ -z "$CITYPERSONS_IMAGE_DIR" ]]; then
    echo "Could not locate a populated CityPersons image tree with a train split." >&2
    exit 1
fi

CROWDHUMAN_DOWNLOAD_DIR="$TEMP_DIR/CrowdHuman-downloads"
CROWDHUMAN_EXTRACT_DIR="$TEMP_DIR/CrowdHuman-extracted"
CROWDHUMAN_UPLOAD_DIR="$TEMP_DIR/CrowdHuman-images"
python3 "$SCRIPT_DIR/scripts/data/download_crowdhuman.py" --output "$CROWDHUMAN_DOWNLOAD_DIR"
mkdir -p "$CROWDHUMAN_EXTRACT_DIR"
for archive in "$CROWDHUMAN_DOWNLOAD_DIR"/CrowdHuman_train*.zip; do
    marker="$CROWDHUMAN_EXTRACT_DIR/.$(basename "$archive").complete"
    if [[ ! -f "$marker" ]]; then
        7z x -y "$archive" -o"$CROWDHUMAN_EXTRACT_DIR"
        touch "$marker"
    fi
done

# Flatten the three CrowdHuman archives into stable blob names without copying
# the image bytes; hard links keep the temporary workspace small.
mkdir -p "$CROWDHUMAN_UPLOAD_DIR"
while IFS= read -r -d '' image_path; do
    image_name="${image_path##*/}"
    if [[ -e "$CROWDHUMAN_UPLOAD_DIR/$image_name" ]]; then
        # A failed/resumed run may already have created this hard link. Reuse
        # it when the bytes match; reject a real filename collision.
        if cmp -s "$image_path" "$CROWDHUMAN_UPLOAD_DIR/$image_name"; then
            continue
        fi
        echo "Conflicting CrowdHuman image filename: $image_name" >&2
        exit 1
    fi
    ln "$image_path" "$CROWDHUMAN_UPLOAD_DIR/$image_name"
done < <(find "$CROWDHUMAN_EXTRACT_DIR" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) -print0)

ANNOTATION_DIR="$TEMP_DIR/official-annotations"
BUILD_DIR="$TEMP_DIR/version-build"
python3 "$SCRIPT_DIR/scripts/data/download_citypersons_annotations.py" --output "$ANNOTATION_DIR"
if [[ ! -f "$BUILD_DIR/manifest.json" ]]; then
    rm -rf "$BUILD_DIR"
    python3 "$SCRIPT_DIR/scripts/data/build_pooled_version.py" \
        --images "$CITYPERSONS_IMAGE_DIR" \
        --annotations "$ANNOTATION_DIR" \
        --crowdhuman-images "$CROWDHUMAN_UPLOAD_DIR" \
        --crowdhuman-annotations "$CROWDHUMAN_DOWNLOAD_DIR/annotation_train.odgt" \
        --source-archive "$CITYPERSONS_ARCHIVE" \
        --source-archive "$CROWDHUMAN_DOWNLOAD_DIR/CrowdHuman_train01.zip" \
        --source-archive "$CROWDHUMAN_DOWNLOAD_DIR/CrowdHuman_train02.zip" \
        --source-archive "$CROWDHUMAN_DOWNLOAD_DIR/CrowdHuman_train03.zip" \
        --output "$BUILD_DIR" --version "$DATASET_VERSION" --version-prefix "$VERSION_PREFIX"
fi

RESET_ARGS=()
if [[ "${CITYPERSONS_RESET_CONTAINER:-${DATASET_RESET_CONTAINER:-false}}" == true ]]; then
    RESET_ARGS+=(--reset-container)
fi
IMMUTABLE_ARGS=(--require-empty-prefix "$VERSION_PREFIX")
if [[ "${DATASET_RESUME_UPLOAD:-${CITYPERSONS_RESUME_UPLOAD:-false}}" == true ]]; then
    IMMUTABLE_ARGS=()
fi

python3 "$SCRIPT_DIR/scripts/data/upload_to_azurite.py" \
    --source "$CITYPERSONS_IMAGE_DIR" --container "$AZURITE_CONTAINER" \
    --prefix "$VERSION_PREFIX/images" "${IMMUTABLE_ARGS[@]}" "${RESET_ARGS[@]}" \
    --include '*.png' --include '*.jpg' --include '*.jpeg' --exclude '*(1).*'
python3 "$SCRIPT_DIR/scripts/data/upload_to_azurite.py" \
    --source "$CROWDHUMAN_UPLOAD_DIR" --container "$AZURITE_CONTAINER" \
    --prefix "$VERSION_PREFIX/images/train/crowdhuman"
python3 "$SCRIPT_DIR/scripts/data/upload_to_azurite.py" \
    --source "$BUILD_DIR" --container "$AZURITE_CONTAINER" --prefix "$VERSION_PREFIX"

PREVIEW_DIR="${CITYPERSONS_PREVIEW_DIR:-${DATASET_PREVIEW_DIR:-$SCRIPT_DIR/validation_preview/$DATASET_VERSION}}"
python3 "$SCRIPT_DIR/scripts/data/validate_citypersons_azurite.py" \
    --container "$AZURITE_CONTAINER" --dataset-prefix "$VERSION_PREFIX" \
    --preview-dir "$PREVIEW_DIR" --preview-count "$PREVIEW_COUNT" \
    "${CHECKSUM_ARGS[@]}" --publish-current
echo "Published pooled CityPersons + CrowdHuman version: $VERSION_PREFIX"
