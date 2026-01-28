# 🦞 Sigil — Personal AI Assistant

<p align="center">
  <img src="https://raw.githubusercontent.com/sigil/sigil/main/docs/whatsapp-clawd.jpg" alt="Sigil" width="400">
</p>

<p align="center">
  <strong>EXFOLIATE! EXFOLIATE!</strong>
</p>

<p align="center">
  <a href="https://github.com/NeelM0906/Bomboclat/actions/workflows/ci.yml?branch=main"><img src="https://img.shields.io/github/actions/workflow/status/sigil/sigil/ci.yml?branch=main&style=for-the-badge" alt="CI status"></a>
  <a href="https://github.com/NeelM0906/Bomboclat/releases"><img src="https://img.shields.io/github/v/release/sigil/sigil?include_prereleases&style=for-the-badge" alt="GitHub release"></a>
  <a href="https://deepwiki.com/sigil/sigil"><img src="https://img.shields.io/badge/DeepWiki-sigil-111111?style=for-the-badge" alt="DeepWiki"></a>
  <a href="#community-coming-soon"><img src="https://img.shields.io/discord/1456350064065904867?label=Discord&logo=discord&logoColor=white&color=5865F2&style=for-the-badge" alt="Discord"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge" alt="MIT License"></a>
</p>

**Sigil** is a *personal AI assistant* you run on your own devices.
It answers you on the channels you already use (WhatsApp, Telegram, Slack, Discord, Google Chat, Signal, iMessage, Microsoft Teams, WebChat), plus extension channels like BlueBubbles, Matrix, Zalo, and Zalo Personal. It can speak and listen on macOS/iOS/Android, and can render a live Canvas you control. The Gateway is just the control plane — the product is the assistant.

If you want a personal, single-user assistant that feels local, fast, and always-on, this is it.

[Website](https://your-domain.com) · [Docs](#docs-coming-soon) · [Getting Started](#docs-coming-soon) · [Updating](#docs-coming-soon) · [Showcase](#docs-coming-soon) · [FAQ](#docs-coming-soon) · [Wizard](#docs-coming-soon) · [Nix](https://github.com/sigil/nix-sigil) · [Docker](#docs-coming-soon) · [Discord](#community-coming-soon)

Preferred setup: run the onboarding wizard (`sigil onboard`). It walks through gateway, workspace, channels, and skills. The CLI wizard is the recommended path and works on **macOS, Linux, and Windows (via WSL2; strongly recommended)**.
Works with npm, pnpm, or bun.
New install? Start here: [Getting started](#docs-coming-soon)

**Subscriptions (OAuth):**
- **[Anthropic](https://www.anthropic.com/)** (Claude Pro/Max)
- **[OpenAI](https://openai.com/)** (ChatGPT/Codex)

Model note: while any model is supported, I strongly recommend **Anthropic Pro/Max (100/200) + Opus 4.5** for long‑context strength and better prompt‑injection resistance. See [Onboarding](#docs-coming-soon).

## Models (selection + auth)

- Models config + CLI: [Models](#docs-coming-soon)
- Auth profile rotation (OAuth vs API keys) + fallbacks: [Model failover](#docs-coming-soon)

## Install (recommended)

Runtime: **Node ≥22**.

```bash
npm install -g sigil@latest
# or: pnpm add -g sigil@latest

sigil onboard --install-daemon
```

The wizard installs the Gateway daemon (launchd/systemd user service) so it stays running.
Legacy note: `sigil` remains available as a compatibility shim.

## Quick start (TL;DR)

Runtime: **Node ≥22**.

Full beginner guide (auth, pairing, channels): [Getting started](#docs-coming-soon)

```bash
sigil onboard --install-daemon

sigil gateway --port 18789 --verbose

# Send a message
sigil message send --to +1234567890 --message "Hello from Sigil"

# Talk to the assistant (optionally deliver back to any connected channel: WhatsApp/Telegram/Slack/Discord/Google Chat/Signal/iMessage/BlueBubbles/Microsoft Teams/Matrix/Zalo/Zalo Personal/WebChat)
sigil agent --message "Ship checklist" --thinking high
```

Upgrading? [Updating guide](#docs-coming-soon) (and run `sigil doctor`).

## Development channels

- **stable**: tagged releases (`vYYYY.M.D` or `vYYYY.M.D-<patch>`), npm dist-tag `latest`.
- **beta**: prerelease tags (`vYYYY.M.D-beta.N`), npm dist-tag `beta` (macOS app may be missing).
- **dev**: moving head of `main`, npm dist-tag `dev` (when published).

Switch channels (git + npm): `sigil update --channel stable|beta|dev`.
Details: [Development channels](#docs-coming-soon).

## From source (development)

Prefer `pnpm` for builds from source. Bun is optional for running TypeScript directly.

```bash
git clone https://github.com/NeelM0906/Bomboclat.git
cd sigil

pnpm install
pnpm ui:build # auto-installs UI deps on first run
pnpm build

pnpm sigil onboard --install-daemon

# Dev loop (auto-reload on TS changes)
pnpm gateway:watch
```

Note: `pnpm sigil ...` runs TypeScript directly (via `tsx`). `pnpm build` produces `dist/` for running via Node / the packaged `sigil` binary.

## Security defaults (DM access)

Sigil connects to real messaging surfaces. Treat inbound DMs as **untrusted input**.

Full security guide: [Security](#docs-coming-soon)

Default behavior on Telegram/WhatsApp/Signal/iMessage/Microsoft Teams/Discord/Google Chat/Slack:
- **DM pairing** (`dmPolicy="pairing"` / `channels.discord.dm.policy="pairing"` / `channels.slack.dm.policy="pairing"`): unknown senders receive a short pairing code and the bot does not process their message.
- Approve with: `sigil pairing approve <channel> <code>` (then the sender is added to a local allowlist store).
- Public inbound DMs require an explicit opt-in: set `dmPolicy="open"` and include `"*"` in the channel allowlist (`allowFrom` / `channels.discord.dm.allowFrom` / `channels.slack.dm.allowFrom`).

Run `sigil doctor` to surface risky/misconfigured DM policies.

## Highlights

- **[Local-first Gateway](#docs-coming-soon)** — single control plane for sessions, channels, tools, and events.
- **[Multi-channel inbox](#docs-coming-soon)** — WhatsApp, Telegram, Slack, Discord, Google Chat, Signal, iMessage, BlueBubbles, Microsoft Teams, Matrix, Zalo, Zalo Personal, WebChat, macOS, iOS/Android.
- **[Multi-agent routing](#docs-coming-soon)** — route inbound channels/accounts/peers to isolated agents (workspaces + per-agent sessions).
- **[Voice Wake](#docs-coming-soon) + [Talk Mode](#docs-coming-soon)** — always-on speech for macOS/iOS/Android with ElevenLabs.
- **[Live Canvas](#docs-coming-soon)** — agent-driven visual workspace with [A2UI](#docs-coming-soon).
- **[First-class tools](#docs-coming-soon)** — browser, canvas, nodes, cron, sessions, and Discord/Slack actions.
- **[Companion apps](#docs-coming-soon)** — macOS menu bar app + iOS/Android [nodes](#docs-coming-soon).
- **[Onboarding](#docs-coming-soon) + [skills](#docs-coming-soon)** — wizard-driven setup with bundled/managed/workspace skills.


## Everything we built so far

### Core platform
- [Gateway WS control plane](#docs-coming-soon) with sessions, presence, config, cron, webhooks, [Control UI](#docs-coming-soon), and [Canvas host](#docs-coming-soon).
- [CLI surface](#docs-coming-soon): gateway, agent, send, [wizard](#docs-coming-soon), and [doctor](#docs-coming-soon).
- [Pi agent runtime](#docs-coming-soon) in RPC mode with tool streaming and block streaming.
- [Session model](#docs-coming-soon): `main` for direct chats, group isolation, activation modes, queue modes, reply-back. Group rules: [Groups](#docs-coming-soon).
- [Media pipeline](#docs-coming-soon): images/audio/video, transcription hooks, size caps, temp file lifecycle. Audio details: [Audio](#docs-coming-soon).

### Channels
- [Channels](#docs-coming-soon): [WhatsApp](#docs-coming-soon) (Baileys), [Telegram](#docs-coming-soon) (grammY), [Slack](#docs-coming-soon) (Bolt), [Discord](#docs-coming-soon) (discord.js), [Google Chat](#docs-coming-soon) (Chat API), [Signal](#docs-coming-soon) (signal-cli), [iMessage](#docs-coming-soon) (imsg), [BlueBubbles](#docs-coming-soon) (extension), [Microsoft Teams](#docs-coming-soon) (extension), [Matrix](#docs-coming-soon) (extension), [Zalo](#docs-coming-soon) (extension), [Zalo Personal](#docs-coming-soon) (extension), [WebChat](#docs-coming-soon).
- [Group routing](#docs-coming-soon): mention gating, reply tags, per-channel chunking and routing. Channel rules: [Channels](#docs-coming-soon).

### Apps + nodes
- [macOS app](#docs-coming-soon): menu bar control plane, [Voice Wake](#docs-coming-soon)/PTT, [Talk Mode](#docs-coming-soon) overlay, [WebChat](#docs-coming-soon), debug tools, [remote gateway](#docs-coming-soon) control.
- [iOS node](#docs-coming-soon): [Canvas](#docs-coming-soon), [Voice Wake](#docs-coming-soon), [Talk Mode](#docs-coming-soon), camera, screen recording, Bonjour pairing.
- [Android node](#docs-coming-soon): [Canvas](#docs-coming-soon), [Talk Mode](#docs-coming-soon), camera, screen recording, optional SMS.
- [macOS node mode](#docs-coming-soon): system.run/notify + canvas/camera exposure.

### Tools + automation
- [Browser control](#docs-coming-soon): dedicated clawd Chrome/Chromium, snapshots, actions, uploads, profiles.
- [Canvas](#docs-coming-soon): [A2UI](#docs-coming-soon) push/reset, eval, snapshot.
- [Nodes](#docs-coming-soon): camera snap/clip, screen record, [location.get](#docs-coming-soon), notifications.
- [Cron + wakeups](#docs-coming-soon); [webhooks](#docs-coming-soon); [Gmail Pub/Sub](#docs-coming-soon).
- [Skills platform](#docs-coming-soon): bundled, managed, and workspace skills with install gating + UI.

### Runtime + safety
- [Channel routing](#docs-coming-soon), [retry policy](#docs-coming-soon), and [streaming/chunking](#docs-coming-soon).
- [Presence](#docs-coming-soon), [typing indicators](#docs-coming-soon), and [usage tracking](#docs-coming-soon).
- [Models](#docs-coming-soon), [model failover](#docs-coming-soon), and [session pruning](#docs-coming-soon).
- [Security](#docs-coming-soon) and [troubleshooting](#docs-coming-soon).

### Ops + packaging
- [Control UI](#docs-coming-soon) + [WebChat](#docs-coming-soon) served directly from the Gateway.
- [Tailscale Serve/Funnel](#docs-coming-soon) or [SSH tunnels](#docs-coming-soon) with token/password auth.
- [Nix mode](#docs-coming-soon) for declarative config; [Docker](#docs-coming-soon)-based installs.
- [Doctor](#docs-coming-soon) migrations, [logging](#docs-coming-soon).

## How it works (short)

```
WhatsApp / Telegram / Slack / Discord / Google Chat / Signal / iMessage / BlueBubbles / Microsoft Teams / Matrix / Zalo / Zalo Personal / WebChat
               │
               ▼
┌───────────────────────────────┐
│            Gateway            │
│       (control plane)         │
│     ws://127.0.0.1:18789      │
└──────────────┬────────────────┘
               │
               ├─ Pi agent (RPC)
               ├─ CLI (sigil …)
               ├─ WebChat UI
               ├─ macOS app
               └─ iOS / Android nodes
```

## Key subsystems

- **[Gateway WebSocket network](#docs-coming-soon)** — single WS control plane for clients, tools, and events (plus ops: [Gateway runbook](#docs-coming-soon)).
- **[Tailscale exposure](#docs-coming-soon)** — Serve/Funnel for the Gateway dashboard + WS (remote access: [Remote](#docs-coming-soon)).
- **[Browser control](#docs-coming-soon)** — clawd‑managed Chrome/Chromium with CDP control.
- **[Canvas + A2UI](#docs-coming-soon)** — agent‑driven visual workspace (A2UI host: [Canvas/A2UI](#docs-coming-soon)).
- **[Voice Wake](#docs-coming-soon) + [Talk Mode](#docs-coming-soon)** — always‑on speech and continuous conversation.
- **[Nodes](#docs-coming-soon)** — Canvas, camera snap/clip, screen record, `location.get`, notifications, plus macOS‑only `system.run`/`system.notify`.

## Tailscale access (Gateway dashboard)

Sigil can auto-configure Tailscale **Serve** (tailnet-only) or **Funnel** (public) while the Gateway stays bound to loopback. Configure `gateway.tailscale.mode`:

- `off`: no Tailscale automation (default).
- `serve`: tailnet-only HTTPS via `tailscale serve` (uses Tailscale identity headers by default).
- `funnel`: public HTTPS via `tailscale funnel` (requires shared password auth).

Notes:
- `gateway.bind` must stay `loopback` when Serve/Funnel is enabled (Sigil enforces this).
- Serve can be forced to require a password by setting `gateway.auth.mode: "password"` or `gateway.auth.allowTailscale: false`.
- Funnel refuses to start unless `gateway.auth.mode: "password"` is set.
- Optional: `gateway.tailscale.resetOnExit` to undo Serve/Funnel on shutdown.

Details: [Tailscale guide](#docs-coming-soon) · [Web surfaces](#docs-coming-soon)

## Remote Gateway (Linux is great)

It’s perfectly fine to run the Gateway on a small Linux instance. Clients (macOS app, CLI, WebChat) can connect over **Tailscale Serve/Funnel** or **SSH tunnels**, and you can still pair device nodes (macOS/iOS/Android) to execute device‑local actions when needed.

- **Gateway host** runs the exec tool and channel connections by default.
- **Device nodes** run device‑local actions (`system.run`, camera, screen recording, notifications) via `node.invoke`.
In short: exec runs where the Gateway lives; device actions run where the device lives.

Details: [Remote access](#docs-coming-soon) · [Nodes](#docs-coming-soon) · [Security](#docs-coming-soon)

## macOS permissions via the Gateway protocol

The macOS app can run in **node mode** and advertises its capabilities + permission map over the Gateway WebSocket (`node.list` / `node.describe`). Clients can then execute local actions via `node.invoke`:

- `system.run` runs a local command and returns stdout/stderr/exit code; set `needsScreenRecording: true` to require screen-recording permission (otherwise you’ll get `PERMISSION_MISSING`).
- `system.notify` posts a user notification and fails if notifications are denied.
- `canvas.*`, `camera.*`, `screen.record`, and `location.get` are also routed via `node.invoke` and follow TCC permission status.

Elevated bash (host permissions) is separate from macOS TCC:

- Use `/elevated on|off` to toggle per‑session elevated access when enabled + allowlisted.
- Gateway persists the per‑session toggle via `sessions.patch` (WS method) alongside `thinkingLevel`, `verboseLevel`, `model`, `sendPolicy`, and `groupActivation`.

Details: [Nodes](#docs-coming-soon) · [macOS app](#docs-coming-soon) · [Gateway protocol](#docs-coming-soon)

## Agent to Agent (sessions_* tools)

- Use these to coordinate work across sessions without jumping between chat surfaces.
- `sessions_list` — discover active sessions (agents) and their metadata.
- `sessions_history` — fetch transcript logs for a session.
- `sessions_send` — message another session; optional reply‑back ping‑pong + announce step (`REPLY_SKIP`, `ANNOUNCE_SKIP`).

Details: [Session tools](#docs-coming-soon)

## Skills registry (ClawdHub)

ClawdHub is a minimal skill registry. With ClawdHub enabled, the agent can search for skills automatically and pull in new ones as needed.

[ClawdHub](https://ClawdHub.com)

## Chat commands

Send these in WhatsApp/Telegram/Slack/Google Chat/Microsoft Teams/WebChat (group commands are owner-only):

- `/status` — compact session status (model + tokens, cost when available)
- `/new` or `/reset` — reset the session
- `/compact` — compact session context (summary)
- `/think <level>` — off|minimal|low|medium|high|xhigh (GPT-5.2 + Codex models only)
- `/verbose on|off`
- `/usage off|tokens|full` — per-response usage footer
- `/restart` — restart the gateway (owner-only in groups)
- `/activation mention|always` — group activation toggle (groups only)

## Apps (optional)

The Gateway alone delivers a great experience. All apps are optional and add extra features.

If you plan to build/run companion apps, follow the platform runbooks below.

### macOS (Sigil.app) (optional)

- Menu bar control for the Gateway and health.
- Voice Wake + push-to-talk overlay.
- WebChat + debug tools.
- Remote gateway control over SSH.

Note: signed builds required for macOS permissions to stick across rebuilds (see `docs/mac/permissions.md`).

### iOS node (optional)

- Pairs as a node via the Bridge.
- Voice trigger forwarding + Canvas surface.
- Controlled via `sigil nodes …`.

Runbook: [iOS connect](#docs-coming-soon).

### Android node (optional)

- Pairs via the same Bridge + pairing flow as iOS.
- Exposes Canvas, Camera, and Screen capture commands.
- Runbook: [Android connect](#docs-coming-soon).

## Agent workspace + skills

- Workspace root: `~/clawd` (configurable via `agents.defaults.workspace`).
- Injected prompt files: `AGENTS.md`, `SOUL.md`, `TOOLS.md`.
- Skills: `~/clawd/skills/<skill>/SKILL.md`.

## Configuration

Minimal `~/.sigil/sigil.json` (model + defaults):

```json5
{
  agent: {
    model: "anthropic/claude-opus-4-5"
  }
}
```

[Full configuration reference (all keys + examples).](#docs-coming-soon)

## Security model (important)

- **Default:** tools run on the host for the **main** session, so the agent has full access when it’s just you.
- **Group/channel safety:** set `agents.defaults.sandbox.mode: "non-main"` to run **non‑main sessions** (groups/channels) inside per‑session Docker sandboxes; bash then runs in Docker for those sessions.
- **Sandbox defaults:** allowlist `bash`, `process`, `read`, `write`, `edit`, `sessions_list`, `sessions_history`, `sessions_send`, `sessions_spawn`; denylist `browser`, `canvas`, `nodes`, `cron`, `discord`, `gateway`.

Details: [Security guide](#docs-coming-soon) · [Docker + sandboxing](#docs-coming-soon) · [Sandbox config](#docs-coming-soon)

### [WhatsApp](#docs-coming-soon)

- Link the device: `pnpm sigil channels login` (stores creds in `~/.sigil/credentials`).
- Allowlist who can talk to the assistant via `channels.whatsapp.allowFrom`.
- If `channels.whatsapp.groups` is set, it becomes a group allowlist; include `"*"` to allow all.

### [Telegram](#docs-coming-soon)

- Set `TELEGRAM_BOT_TOKEN` or `channels.telegram.botToken` (env wins).
- Optional: set `channels.telegram.groups` (with `channels.telegram.groups."*".requireMention`); when set, it is a group allowlist (include `"*"` to allow all). Also `channels.telegram.allowFrom` or `channels.telegram.webhookUrl` as needed.

```json5
{
  channels: {
    telegram: {
      botToken: "123456:ABCDEF"
    }
  }
}
```

### [Slack](#docs-coming-soon)

- Set `SLACK_BOT_TOKEN` + `SLACK_APP_TOKEN` (or `channels.slack.botToken` + `channels.slack.appToken`).

### [Discord](#docs-coming-soon)

- Set `DISCORD_BOT_TOKEN` or `channels.discord.token` (env wins).
- Optional: set `commands.native`, `commands.text`, or `commands.useAccessGroups`, plus `channels.discord.dm.allowFrom`, `channels.discord.guilds`, or `channels.discord.mediaMaxMb` as needed.

```json5
{
  channels: {
    discord: {
      token: "1234abcd"
    }
  }
}
```

### [Signal](#docs-coming-soon)

- Requires `signal-cli` and a `channels.signal` config section.

### [iMessage](#docs-coming-soon)

- macOS only; Messages must be signed in.
- If `channels.imessage.groups` is set, it becomes a group allowlist; include `"*"` to allow all.

### [Microsoft Teams](#docs-coming-soon)

- Configure a Teams app + Bot Framework, then add a `msteams` config section.
- Allowlist who can talk via `msteams.allowFrom`; group access via `msteams.groupAllowFrom` or `msteams.groupPolicy: "open"`.

### [WebChat](#docs-coming-soon)

- Uses the Gateway WebSocket; no separate WebChat port/config.

Browser control (optional):

```json5
{
  browser: {
    enabled: true,
    color: "#FF4500"
  }
}
```

## Docs

Use these when you’re past the onboarding flow and want the deeper reference.
- [Start with the docs index for navigation and “what’s where.”](#docs-coming-soon)
- [Read the architecture overview for the gateway + protocol model.](#docs-coming-soon)
- [Use the full configuration reference when you need every key and example.](#docs-coming-soon)
- [Run the Gateway by the book with the operational runbook.](#docs-coming-soon)
- [Learn how the Control UI/Web surfaces work and how to expose them safely.](#docs-coming-soon)
- [Understand remote access over SSH tunnels or tailnets.](#docs-coming-soon)
- [Follow the onboarding wizard flow for a guided setup.](#docs-coming-soon)
- [Wire external triggers via the webhook surface.](#docs-coming-soon)
- [Set up Gmail Pub/Sub triggers.](#docs-coming-soon)
- [Learn the macOS menu bar companion details.](#docs-coming-soon)
- [Platform guides: Windows (WSL2)](#docs-coming-soon), [Linux](#docs-coming-soon), [macOS](#docs-coming-soon), [iOS](#docs-coming-soon), [Android](#docs-coming-soon)
- [Debug common failures with the troubleshooting guide.](#docs-coming-soon)
- [Review security guidance before exposing anything.](#docs-coming-soon)

## Advanced docs (discovery + control)

- [Discovery + transports](#docs-coming-soon)
- [Bonjour/mDNS](#docs-coming-soon)
- [Gateway pairing](#docs-coming-soon)
- [Remote gateway README](#docs-coming-soon)
- [Control UI](#docs-coming-soon)
- [Dashboard](#docs-coming-soon)

## Operations & troubleshooting

- [Health checks](#docs-coming-soon)
- [Gateway lock](#docs-coming-soon)
- [Background process](#docs-coming-soon)
- [Browser troubleshooting (Linux)](#docs-coming-soon)
- [Logging](#docs-coming-soon)

## Deep dives

- [Agent loop](#docs-coming-soon)
- [Presence](#docs-coming-soon)
- [TypeBox schemas](#docs-coming-soon)
- [RPC adapters](#docs-coming-soon)
- [Queue](#docs-coming-soon)

## Workspace & skills

- [Skills config](#docs-coming-soon)
- [Default AGENTS](#docs-coming-soon)
- [Templates: AGENTS](#docs-coming-soon)
- [Templates: BOOTSTRAP](#docs-coming-soon)
- [Templates: IDENTITY](#docs-coming-soon)
- [Templates: SOUL](#docs-coming-soon)
- [Templates: TOOLS](#docs-coming-soon)
- [Templates: USER](#docs-coming-soon)

## Platform internals

- [macOS dev setup](#docs-coming-soon)
- [macOS menu bar](#docs-coming-soon)
- [macOS voice wake](#docs-coming-soon)
- [iOS node](#docs-coming-soon)
- [Android node](#docs-coming-soon)
- [Windows (WSL2)](#docs-coming-soon)
- [Linux app](#docs-coming-soon)

## Email hooks (Gmail)

- [docs.your-domain.com/gmail-pubsub](#docs-coming-soon)

## Clawd

Sigil was built for **Clawd**, a space lobster AI assistant. 🦞
by Peter Steinberger and the community.

- [clawd.me](https://clawd.me)
- [soul.md](https://soul.md)
- [steipete.me](https://steipete.me)

## Community

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines, maintainers, and how to submit PRs.
AI/vibe-coded PRs welcome! 🤖

