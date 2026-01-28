---
summary: "Run multiple Sigil Gateways on one host (isolation, ports, and profiles)"
read_when:
  - Running more than one Gateway on the same machine
  - You need isolated config/state/ports per Gateway
---
# Multiple Gateways (same host)

Most setups should use one Gateway because a single Gateway can handle multiple messaging connections and agents. If you need stronger isolation or redundancy (e.g., a rescue bot), run separate Gateways with isolated profiles/ports.

## Isolation checklist (required)
- `SIGIL_CONFIG_PATH` — per-instance config file
- `SIGIL_STATE_DIR` — per-instance sessions, creds, caches
- `agents.defaults.workspace` — per-instance workspace root
- `gateway.port` (or `--port`) — unique per instance
- Derived ports (browser/canvas) must not overlap

If these are shared, you will hit config races and port conflicts.

## Recommended: profiles (`--profile`)

Profiles auto-scope `SIGIL_STATE_DIR` + `SIGIL_CONFIG_PATH` and suffix service names.

```bash
# main
sigil --profile main setup
sigil --profile main gateway --port 18789

# rescue
sigil --profile rescue setup
sigil --profile rescue gateway --port 19001
```

Per-profile services:
```bash
sigil --profile main gateway install
sigil --profile rescue gateway install
```

## Rescue-bot guide

Run a second Gateway on the same host with its own:
- profile/config
- state dir
- workspace
- base port (plus derived ports)

This keeps the rescue bot isolated from the main bot so it can debug or apply config changes if the primary bot is down.

Port spacing: leave at least 20 ports between base ports so the derived browser/canvas/CDP ports never collide.

### How to install (rescue bot)

```bash
# Main bot (existing or fresh, without --profile param)
# Runs on port 18789 + Chrome CDC/Canvas/... Ports 
sigil onboard
sigil gateway install

# Rescue bot (isolated profile + ports)
sigil --profile rescue onboard
# Notes: 
# - workspace name will be postfixed with -rescue per default
# - Port should be at least 18789 + 20 Ports, 
#   better choose completely different base port, like 19789,
# - rest of the onboarding is the same as normal

# To install the service (if not happened automatically during onboarding)
sigil --profile rescue gateway install
```

## Port mapping (derived)

Base port = `gateway.port` (or `SIGIL_GATEWAY_PORT` / `--port`).

- browser control service port = base + 2 (loopback only)
- `canvasHost.port = base + 4`
- Browser profile CDP ports auto-allocate from `browser.controlPort + 9 .. + 108`

If you override any of these in config or env, you must keep them unique per instance.

## Browser/CDP notes (common footgun)

- Do **not** pin `browser.cdpUrl` to the same values on multiple instances.
- Each instance needs its own browser control port and CDP range (derived from its gateway port).
- If you need explicit CDP ports, set `browser.profiles.<name>.cdpPort` per instance.
- Remote Chrome: use `browser.profiles.<name>.cdpUrl` (per profile, per instance).

## Manual env example

```bash
SIGIL_CONFIG_PATH=~/.sigil/main.json \
SIGIL_STATE_DIR=~/.sigil-main \
sigil gateway --port 18789

SIGIL_CONFIG_PATH=~/.sigil/rescue.json \
SIGIL_STATE_DIR=~/.sigil-rescue \
sigil gateway --port 19001
```

## Quick checks

```bash
sigil --profile main status
sigil --profile rescue status
sigil --profile rescue browser status
```
