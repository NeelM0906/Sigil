---
summary: "CLI reference for `sigil voicecall` (voice-call plugin command surface)"
read_when:
  - You use the voice-call plugin and want the CLI entry points
  - You want quick examples for `voicecall call|continue|status|tail|expose`
---

# `sigil voicecall`

`voicecall` is a plugin-provided command. It only appears if the voice-call plugin is installed and enabled.

Primary doc:
- Voice-call plugin: [Voice Call](/plugins/voice-call)

## Common commands

```bash
sigil voicecall status --call-id <id>
sigil voicecall call --to "+15555550123" --message "Hello" --mode notify
sigil voicecall continue --call-id <id> --message "Any questions?"
sigil voicecall end --call-id <id>
```

## Exposing webhooks (Tailscale)

```bash
sigil voicecall expose --mode serve
sigil voicecall expose --mode funnel
sigil voicecall unexpose
```

Security note: only expose the webhook endpoint to networks you trust. Prefer Tailscale Serve over Funnel when possible.

