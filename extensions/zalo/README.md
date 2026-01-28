# @sigil/zalo

Zalo channel plugin for Sigil (Bot API).

## Install (local checkout)

```bash
sigil plugins install ./extensions/zalo
```

## Install (npm)

```bash
sigil plugins install @sigil/zalo
```

Onboarding: select Zalo and confirm the install prompt to fetch the plugin automatically.

## Config

```json5
{
  channels: {
    zalo: {
      enabled: true,
      botToken: "12345689:abc-xyz",
      dmPolicy: "pairing",
      proxy: "http://proxy.local:8080"
    }
  }
}
```

## Webhook mode

```json5
{
  channels: {
    zalo: {
      webhookUrl: "https://example.com/zalo-webhook",
      webhookSecret: "your-secret-8-plus-chars",
      webhookPath: "/zalo-webhook"
    }
  }
}
```

If `webhookPath` is omitted, the plugin uses the webhook URL path.

Restart the gateway after config changes.
