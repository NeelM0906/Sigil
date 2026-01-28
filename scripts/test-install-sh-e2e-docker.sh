#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${SIGIL_INSTALL_E2E_IMAGE:-sigil-install-e2e:local}"
INSTALL_URL="${SIGIL_INSTALL_URL:-https://sigil.bot/install.sh}"

OPENAI_API_KEY="${OPENAI_API_KEY:-}"
ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"
ANTHROPIC_API_TOKEN="${ANTHROPIC_API_TOKEN:-}"
SIGIL_E2E_MODELS="${SIGIL_E2E_MODELS:-}"

echo "==> Build image: $IMAGE_NAME"
docker build \
  -t "$IMAGE_NAME" \
  -f "$ROOT_DIR/scripts/docker/install-sh-e2e/Dockerfile" \
  "$ROOT_DIR/scripts/docker/install-sh-e2e"

echo "==> Run E2E installer test"
docker run --rm \
  -e SIGIL_INSTALL_URL="$INSTALL_URL" \
  -e SIGIL_INSTALL_TAG="${SIGIL_INSTALL_TAG:-latest}" \
  -e SIGIL_E2E_MODELS="$SIGIL_E2E_MODELS" \
  -e SIGIL_INSTALL_E2E_PREVIOUS="${SIGIL_INSTALL_E2E_PREVIOUS:-}" \
  -e SIGIL_INSTALL_E2E_SKIP_PREVIOUS="${SIGIL_INSTALL_E2E_SKIP_PREVIOUS:-0}" \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
  -e ANTHROPIC_API_TOKEN="$ANTHROPIC_API_TOKEN" \
  "$IMAGE_NAME"
