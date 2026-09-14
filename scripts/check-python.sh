#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHONPATH="$ROOT_DIR/api/tests" python3 -m pytest -q "$ROOT_DIR/api/tests/test_reference_runtime.py"

