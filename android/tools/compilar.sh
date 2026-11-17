#!/usr/bin/env bash
# Uso: bash tools/compilar.sh [release]
set -euo pipefail
source "$(dirname "$0")/env.sh"; cd "$RAIZ"
TAREA="assembleDebug"; [ "${1:-}" = release ] && TAREA="assembleRelease"
./gradlew --console=plain "$TAREA"
