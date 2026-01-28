---
summary: "CLI reference for `sigil reset` (reset local state/config)"
read_when:
  - You want to wipe local state while keeping the CLI installed
  - You want a dry-run of what would be removed
---

# `sigil reset`

Reset local config/state (keeps the CLI installed).

```bash
sigil reset
sigil reset --dry-run
sigil reset --scope config+creds+sessions --yes --non-interactive
```

