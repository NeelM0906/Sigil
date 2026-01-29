#!/bin/bash

echo ""
echo "========================================"
echo "  Bomboclat Skill Setup"
echo "========================================"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Create skills directory in user's Clawdbot workspace
if [ ! -d ~/clawd/skills ]; then
    echo "Creating skills directory..."
    mkdir -p ~/clawd/skills
fi

# Copy unblinded-knowledge skill
echo "Installing Unblinded Knowledge skill..."
cp -r "$SCRIPT_DIR/skills/unblinded-knowledge" ~/clawd/skills/

echo ""
echo "========================================"
echo "  ✅ Skill Installed Successfully!"
echo "========================================"
echo ""
echo "Location: ~/clawd/skills/unblinded-knowledge"
echo ""
echo "========================================"
echo "  Next Steps:"
echo "========================================"
echo ""
echo "1. Install Python dependencies:"
echo "   pip install pinecone openai"
echo ""
echo "2. Set your API keys:"
echo "   export PINECONE_API_KEY='your-pinecone-key'"
echo "   export OPENAI_API_KEY='your-openai-key'"
echo ""
echo "   (Get keys from: https://app.pinecone.io and https://platform.openai.com)"
echo ""
echo "3. Start Clawdbot:"
echo "   pnpm clawdbot gateway --port 18789"
echo ""
echo "4. Activate on WhatsApp:"
echo "   \"activate unblinded knowledge\""
echo ""
echo "========================================"
echo ""