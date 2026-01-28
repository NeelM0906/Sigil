---
summary: "CLI reference for `sigil logs` (tail gateway logs via RPC)"
read_when:
  - You need to tail Gateway logs remotely (without SSH)
  - You want JSON log lines for tooling
---

# `sigil logs`

Tail Gateway file logs over RPC (works in remote mode).

Related:
- Logging overview: [Logging](/logging)

## Examples

```bash
sigil logs
sigil logs --follow
sigil logs --json
sigil logs --limit 500
```

