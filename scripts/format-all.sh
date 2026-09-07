#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/format-backend.sh"
"$(dirname "$0")/format-frontend.sh"

