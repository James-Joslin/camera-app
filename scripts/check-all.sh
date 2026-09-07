#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
"$ROOT_DIR/scripts/check-backend.sh"
"$ROOT_DIR/scripts/check-python.sh"
"$ROOT_DIR/scripts/check-images.sh"

