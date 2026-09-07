#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_compose.sh"
compose up -d --build db azurite migrations api fastapi frontend
compose ps

