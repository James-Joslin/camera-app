#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHONPATH="$ROOT_DIR/fastapi" python3 -m pytest -q "$ROOT_DIR/fastapi/tests"

