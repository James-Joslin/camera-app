#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
npm --prefix "$ROOT_DIR/frontend" run format:check

