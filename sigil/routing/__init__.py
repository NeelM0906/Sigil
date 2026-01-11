"""Routing module for Sigil v2 framework.

This module implements intent classification and complexity assessment to route
user requests to appropriate handlers. It is the final task in Phase 3 (Foundation
Layer) and provides the bridge between user interfaces and execution handlers.

Key Components:
    Intent: Enumeration of possible user intents (create, run, query, modify, etc.).
    IntentClassifier: Classifies user messages into intents with confidence scores.
    ComplexityAssessor: Assesses request complexity on a 0.0-1.0 scale.
    RouteDecision: Dataclass containing the complete routing decision.
    Router: Main router that orchestrates classification and routing decisions.

Usage:
    from sigil.routing import Router, Intent, RouteDecision
    from sigil.config import get_settings

    # Initialize router with settings for feature flag access
    settings = get_settings()
    router = Router(settings)

    # Route a user message
    decision = router.route("Create an agent that can search the web")

    # Inspect the decision
    print(f"Intent: {decision.intent}")          # Intent.CREATE_AGENT
    print(f"Handler: {decision.handler_name}")   # "builder"
    print(f"Complexity: {decision.complexity}")  # 0.45 (example)
    print(f"Use planning: {decision.use_planning}")  # True if complexity > 0.5

Handler Mapping:
    - CREATE_AGENT -> "builder"
    - RUN_AGENT -> "executor"
    - QUERY_MEMORY -> "memory_query"
    - MODIFY_AGENT -> "agent_modifier"
    - SYSTEM_COMMAND -> "system_command"
    - GENERAL_CHAT -> "chat"

Example:
    >>> from sigil.routing import Router, Intent
    >>> from sigil.config import get_settings
    >>> router = Router(get_settings())
    >>> decision = router.route("build a sales qualification agent")
    >>> decision.intent == Intent.CREATE_AGENT
    True
    >>> decision.handler_name
    'builder'
"""

from sigil.routing.router import (
    Intent,
    IntentClassifier,
    ComplexityAssessor,
    RouteDecision,
    Router,
)

__all__ = [
    "Intent",
    "IntentClassifier",
    "ComplexityAssessor",
    "RouteDecision",
    "Router",
]
