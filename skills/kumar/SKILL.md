---
name: kumar_learning
description: Agent automatically learns from Kumar's conversations
auto_learn: true
---

# Kumar Auto-Learning Agent

Agent learns from:
1. Historical: 32 past Kumar teaching conversations (CSV)
2. Live: Every new conversation with Kumar automatically saved

## Setup (One Time)
```bash
python setup.py
```

Creates `kumar_knowledge.json` with 32 historical examples.

## How It Works

### When YOU chat:
1. Bot queries `kumar_knowledge.json`
2. Finds similar Kumar teachings
3. Responds based on Kumar's patterns

### When KUMAR chats:
1. Bot responds
2. AUTOMATICALLY saves conversation to `kumar_knowledge.json`
3. Next time: Bot knows this conversation too

No manual intervention needed. Bot learns automatically.

## Integration with Sigil

Add to bot's conversation handler:
```javascript
// After each conversation with Kumar
if (user.name === "Kumar") {
    exec(`python skills/kumar/learn_from_conversation.py "${userMessage}" "${botResponse}" --kumar`);
}
```

Bot automatically learns from every Kumar conversation.