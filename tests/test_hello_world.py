"""Task 1.3: Verify deepagents works with a minimal hello-world agent."""

from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

from deepagents import create_deep_agent
from langchain_core.messages import HumanMessage


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

    if ai_messages:
        response = ai_messages[-1].content
        print(f"\n[PASS] Agent responded: {response}")
        return True
    else:
        print("\n[FAIL] No AI response received")
        return False


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

    if ai_messages:
        response = ai_messages[-1].content
        print(f"[PASS] Opus responded: {response}")
        return True
    else:
        print("[FAIL] No Opus response received")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("Task 1.3: deepagents Hello World Test")
    print("=" * 50)

    success1 = test_hello_world()
    success2 = test_with_opus()

    print("\n" + "=" * 50)
    if success1 and success2:
        print("[PASS] All tests passed! deepagents is working correctly.")
    else:
        print("[FAIL] Some tests failed. Check output above.")
    print("=" * 50)
