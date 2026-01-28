---
summary: "CLI reference for `sigil config` (get/set/unset config values)"
read_when:
  - You want to read or edit config non-interactively
---

# `sigil config`

Config helpers: get/set/unset values by path. Run without a subcommand to open
the configure wizard (same as `sigil configure`).

## Examples

```bash
sigil config get browser.executablePath
sigil config set browser.executablePath "/usr/bin/google-chrome"
sigil config set agents.defaults.heartbeat.every "2h"
sigil config set agents.list[0].tools.exec.node "node-id-or-name"
sigil config unset tools.web.search.apiKey
```

## Paths

Paths use dot or bracket notation:

```bash
sigil config get agents.defaults.workspace
sigil config get agents.list[0].id
```

Use the agent list index to target a specific agent:

```bash
sigil config get agents.list
sigil config set agents.list[1].tools.exec.node "node-id-or-name"
```

## Values

Values are parsed as JSON5 when possible; otherwise they are treated as strings.
Use `--json` to require JSON5 parsing.

```bash
sigil config set agents.defaults.heartbeat.every "0m"
sigil config set gateway.port 19001 --json
sigil config set channels.whatsapp.groups '["*"]' --json
```

Restart the gateway after edits.
