#!/usr/bin/env bash
set -euo pipefail

cd /repo

export SIGIL_STATE_DIR="/tmp/sigil-test"
export SIGIL_CONFIG_PATH="${SIGIL_STATE_DIR}/sigil.json"

echo "==> Seed state"
mkdir -p "${SIGIL_STATE_DIR}/credentials"
mkdir -p "${SIGIL_STATE_DIR}/agents/main/sessions"
echo '{}' >"${SIGIL_CONFIG_PATH}"
echo 'creds' >"${SIGIL_STATE_DIR}/credentials/marker.txt"
echo 'session' >"${SIGIL_STATE_DIR}/agents/main/sessions/sessions.json"

echo "==> Reset (config+creds+sessions)"
pnpm sigil reset --scope config+creds+sessions --yes --non-interactive

test ! -f "${SIGIL_CONFIG_PATH}"
test ! -d "${SIGIL_STATE_DIR}/credentials"
test ! -d "${SIGIL_STATE_DIR}/agents/main/sessions"

echo "==> Recreate minimal config"
mkdir -p "${SIGIL_STATE_DIR}/credentials"
echo '{}' >"${SIGIL_CONFIG_PATH}"

echo "==> Uninstall (state only)"
pnpm sigil uninstall --state --yes --non-interactive

test ! -d "${SIGIL_STATE_DIR}"

echo "OK"
