#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
dotnet restore "$ROOT_DIR/api/cameraApi.csproj"
dotnet build "$ROOT_DIR/api/cameraApi.csproj" --no-restore
dotnet test "$ROOT_DIR/api.Tests/cameraApi.Tests.csproj" --no-restore

