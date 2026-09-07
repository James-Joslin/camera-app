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
        echo "Resume with: CITYPERSONS_WORK_DIR=$TEMP_DIR ./getCityPersons.sh" >&2
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
if find "$EXTRACT_DIR" -type d -name train -print -quit 2>/dev/null | grep -q .; then
    echo "Using existing extracted dataset: $EXTRACT_DIR"
else
    echo "Extracting inner archive..."
    mkdir -p "$EXTRACT_DIR"
    7z x -y "$INNER_ARCHIVE" -o"$EXTRACT_DIR"
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

find_yolo_label_tree() {
    local candidate
    while IFS= read -r candidate; do
        [[ -d "$candidate/train" ]] || continue
        if find "$candidate/train" -type f -name '*.txt' -print -quit | grep -q .; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done < <(find "$EXTRACT_DIR" -type d -name labels)
    return 0
}

IMAGE_DIR=$(find_image_tree)
if [[ -z "$IMAGE_DIR" ]]; then
    echo "Could not locate a populated CityPersons image tree with a train split." >&2
    exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
YOLO_LABEL_DIR=$(find_yolo_label_tree)
if [[ -n "$YOLO_LABEL_DIR" ]]; then
    echo "Using existing YOLO labels: $YOLO_LABEL_DIR"
else
    ANNOTATION_DIR=$(find "$EXTRACT_DIR" -type d -name gtBboxCityPersons -print -quit)
    if [[ -z "$ANNOTATION_DIR" ]]; then
        echo "Could not locate YOLO labels or a gtBboxCityPersons annotation directory." >&2
        echo "Top-level extracted directories:" >&2
        find "$EXTRACT_DIR" -maxdepth 3 -type d | sort | head -n 80 >&2
        exit 1
    fi

    YOLO_LABEL_DIR="$TEMP_DIR/labels"
    echo "Converting annotations from $ANNOTATION_DIR to YOLO labels..."
    python3 "$SCRIPT_DIR/convert_citypersons_to_yolo.py" \
        --annotations "$ANNOTATION_DIR" \
        --output "$YOLO_LABEL_DIR"
fi

echo "Uploading images from $IMAGE_DIR..."
python3 "$SCRIPT_DIR/upload_to_azurite.py" \
    --source "$IMAGE_DIR" \
    --container "$AZURITE_CONTAINER" \
    --prefix images \
    --include '*.png' \
    --include '*.jpg' \
    --include '*.jpeg'
echo "Uploading labels from $YOLO_LABEL_DIR..."
python3 "$SCRIPT_DIR/upload_to_azurite.py" \
    --source "$YOLO_LABEL_DIR" \
    --container "$AZURITE_CONTAINER" \
    --prefix labels \
    --include '*.txt'
echo "All done! Dataset uploaded successfully."
