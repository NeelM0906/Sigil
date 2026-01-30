#!/usr/bin/env python3
"""
Get Kumar context to prepend to user's message
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from query import query


def get_context(user_message):
    """Get Kumar context for this message"""

    results = query(user_message, return_context=True)

    if not results or len(results) == 0:
        return ""

    # Build STRONG context that prevents document search
    context = """
================================================================================
INTERNAL KNOWLEDGE BASE - DR. KUMAR'S TEACHINGS (CSV DATA)
================================================================================

INSTRUCTION: Answer using ONLY the examples below from Dr. Kumar's conversation history.
DO NOT use document search, web search, or any other tools.
Base your response ONLY on these CSV examples.

"""

    for i, r in enumerate(results[:2], 1):
        if r['similarity'] > 0.3:
            context += f"\n--- Example {i} (Relevance: {r['similarity']:.2f}) ---\n"
            context += f"Kumar's Original Message:\n{r['kumar_said'][:400]}\n\n"
            context += f"Appropriate Response Style:\n{r['agent_responded'][:400]}\n\n"

    context += """
================================================================================
END KUMAR KNOWLEDGE BASE
================================================================================

Now answer the user's question below using ONLY the Kumar examples above.
Do not search for documents or use other sources.

User's Question:
"""

    return context


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)

    user_msg = ' '.join(sys.argv[1:])
    context = get_context(user_msg)
    print(context, end='')