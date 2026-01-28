#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${SIGIL_IMAGE:-sigil:local}"
CONFIG_DIR="${SIGIL_CONFIG_DIR:-$HOME/.sigil}"
WORKSPACE_DIR="${SIGIL_WORKSPACE_DIR:-$HOME/clawd}"
PROFILE_FILE="${SIGIL_PROFILE_FILE:-$HOME/.profile}"

PROFILE_MOUNT=()
if [[ -f "$PROFILE_FILE" ]]; then
  PROFILE_MOUNT=(-v "$PROFILE_FILE":/home/node/.profile:ro)
fi

echo "==> Build image: $IMAGE_NAME"
docker build -t "$IMAGE_NAME" -f "$ROOT_DIR/Dockerfile" "$ROOT_DIR"

echo "==> Run live model tests (profile keys)"
docker run --rm -t \
  --entrypoint bash \
  -e COREPACK_ENABLE_DOWNLOAD_PROMPT=0 \
  -e HOME=/home/node \
  -e NODE_OPTIONS=--disable-warning=ExperimentalWarning \
  -e SIGIL_LIVE_TEST=1 \
  -e SIGIL_LIVE_MODELS="${SIGIL_LIVE_MODELS:-all}" \
  -e SIGIL_LIVE_PROVIDERS="${SIGIL_LIVE_PROVIDERS:-}" \
  -e SIGIL_LIVE_MODEL_TIMEOUT_MS="${SIGIL_LIVE_MODEL_TIMEOUT_MS:-}" \
  -e SIGIL_LIVE_REQUIRE_PROFILE_KEYS="${SIGIL_LIVE_REQUIRE_PROFILE_KEYS:-}" \
  -v "$CONFIG_DIR":/home/node/.sigil \
  -v "$WORKSPACE_DIR":/home/node/clawd \
  "${PROFILE_MOUNT[@]}" \
  "$IMAGE_NAME" \
  -lc "set -euo pipefail; [ -f \"$HOME/.profile\" ] && source \"$HOME/.profile\" || true; cd /app && pnpm test:live"
