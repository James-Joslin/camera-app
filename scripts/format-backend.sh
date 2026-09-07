#!/usr/bin/env bash
set -euo pipefail
dotnet format "$(cd "$(dirname "$0")/.." && pwd)/api/cameraApi.csproj" --verify-no-changes

