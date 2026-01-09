"""Task 1.3: Verify deepagents works with a minimal hello-world agent."""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load .env from project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

from deepagents import create_deep_agent
from langchain_core.messages import HumanMessage


# Skip condition for integration tests
_skip_integration = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY") or
    os.environ.get("SKIP_INTEGRATION_TESTS", "").lower() == "true",
    reason="Requires real ANTHROPIC_API_KEY and integration tests enabled"
)


@_skip_integration
def test_hello_world():
    """Create a minimal agent and verify it responds."""
    print("Creating deepagents agent...")

    # Create agent with default model (claude-opus)
    agent = create_deep_agent()

    print("Invoking agent with simple message...")
    result = agent.invoke({
        "messages": [HumanMessage(content="Hello! Please respond with exactly: 'Hello from deepagents!'")]
    })

    # Extract AI response
    ai_messages = [msg for msg in result.get("messages", []) if msg.type == "ai"]

    assert ai_messages, "Expected AI response messages but none were received"
    response = ai_messages[-1].content
    print(f"\n[PASS] Agent responded: {response}")
    assert "messages" in result


@_skip_integration
def test_with_opus():
    """Test with Claude Opus model (used for main builder)."""
    print("\nTesting with Claude Opus...")

    agent = create_deep_agent(
        model="anthropic:claude-opus-4-5-20251101"
    )

    result = agent.invoke({
        "messages": [HumanMessage(content="What is 2 + 2? Answer with just the number.")]
    })

    ai_messages = [msg for msg in result.get("messages", []) if msg.type == "ai"]

    assert ai_messages, "Expected AI response from Opus but none were received"
    response = ai_messages[-1].content
    print(f"[PASS] Opus responded: {response}")
    assert "messages" in result


if __name__ == "__main__":
    print("=" * 50)
    print("Task 1.3: deepagents Hello World Test")
    print("=" * 50)

    try:
        test_hello_world()
        test_with_opus()
        print("\n" + "=" * 50)
        print("[PASS] All tests passed! deepagents is working correctly.")
    except AssertionError as e:
        print("\n" + "=" * 50)
        print(f"[FAIL] Test failed: {e}")
    except Exception as e:
        print("\n" + "=" * 50)
        print(f"[ERROR] Test error: {e}")
    print("=" * 50)
