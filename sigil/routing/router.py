"""Basic routing layer for Sigil v2 framework.

This module implements intent classification and complexity assessment to route
user requests to appropriate handlers. It provides a keyword-based classification
system with optional LLM fallback for ambiguous cases.

The routing layer is part of the Foundation Layer (Phase 3.7) and integrates
with the existing configuration system for feature flag access.

Key Components:
    Intent: Enumeration of possible user intents.
    IntentClassifier: Classifies user messages into intents with confidence scores.
    ComplexityAssessor: Assesses the complexity of a request on a 0.0-1.0 scale.
    RouteDecision: Dataclass containing the complete routing decision.
    Router: Main router class that orchestrates classification and routing.

Usage:
    from sigil.routing import Router, Intent, RouteDecision
    from sigil.config import get_settings

    settings = get_settings()
    router = Router(settings)

    decision = router.route("Create an agent that can search the web")
    print(f"Intent: {decision.intent}")
    print(f"Handler: {decision.handler_name}")
    print(f"Complexity: {decision.complexity}")

Example:
    >>> router = Router(get_settings())
    >>> decision = router.route("run the sales agent")
    >>> decision.intent
    <Intent.RUN_AGENT: 'run_agent'>
    >>> decision.handler_name
    'executor'
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sigil.config.settings import SigilSettings


# =============================================================================
# Module Logger
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Intent Enumeration
# =============================================================================


class Intent(Enum):
    """Enumeration of possible user intents.

    Each intent represents a distinct category of user request that maps
    to a specific handler in the Sigil framework.

    Attributes:
        CREATE_AGENT: User wants to create, build, or design a new agent.
        RUN_AGENT: User wants to execute, run, or demo an existing agent.
        QUERY_MEMORY: User wants to search, find, or recall information.
        MODIFY_AGENT: User wants to change, update, or edit an agent.
        WORKFLOW_KNOWLEDGE: User wants workflow knowledge or best practices.
        SYSTEM_COMMAND: User issues a system command (e.g., /help, /status).
        GENERAL_CHAT: General conversation that doesn't fit other categories.
    """

    CREATE_AGENT = "create_agent"
    RUN_AGENT = "run_agent"
    QUERY_MEMORY = "query_memory"
    MODIFY_AGENT = "modify_agent"
    WORKFLOW_KNOWLEDGE = "workflow_knowledge"  # NEW: Added for workflow knowledge queries
    SYSTEM_COMMAND = "system_command"
    GENERAL_CHAT = "general_chat"


# =============================================================================
# Intent Classifier
# =============================================================================


class IntentClassifier:
    """Classifies user messages into intents using keyword matching.

    The classifier uses a rule-based approach with keyword patterns to
    determine the most likely intent of a user message. It returns both
    the classified intent and a confidence score.

    The keyword-based approach is fast and deterministic, making it suitable
    for MVP use. An optional LLM fallback can be added for ambiguous cases.

    Attributes:
        _keyword_patterns: Dictionary mapping intents to (keywords, weight) tuples.

    Example:
        >>> classifier = IntentClassifier()
        >>> intent, confidence = classifier.classify("create a new sales agent")
        >>> intent
        <Intent.CREATE_AGENT: 'create_agent'>
        >>> confidence > 0.7
        True
    """

    # Keywords mapped to intents with relative weights
    # Higher weight keywords are more definitive indicators
    _INTENT_KEYWORDS: dict[Intent, list[tuple[str, float]]] = {
        Intent.CREATE_AGENT: [
            ("create", 1.0),
            ("build", 1.0),
            ("design", 0.9),
            ("make", 0.8),
            ("new agent", 1.0),
            ("generate agent", 1.0),
            ("construct", 0.8),
        ],
        Intent.RUN_AGENT: [
            ("run", 1.0),
            ("execute", 1.0),
            ("use", 0.7),
            ("with", 0.5),
            ("demo", 0.9),
            ("start", 0.8),
            ("launch", 0.9),
            ("try", 0.6),
        ],
        Intent.QUERY_MEMORY: [
            ("find", 0.9),
            ("search", 1.0),
            ("what", 0.6),
            ("remember", 1.0),
            ("recall", 1.0),
            ("know", 0.7),
            ("where", 0.6),
            ("when", 0.6),
            ("who", 0.6),
            ("retrieve", 0.9),
            ("lookup", 0.9),
        ],
        Intent.MODIFY_AGENT: [
            ("change", 1.0),
            ("update", 1.0),
            ("edit", 1.0),
            ("modify", 1.0),
            ("alter", 0.9),
            ("revise", 0.9),
            ("adjust", 0.8),
            ("tweak", 0.8),
            ("configure", 0.8),
        ],
        # NEW: Workflow knowledge patterns
        Intent.WORKFLOW_KNOWLEDGE: [
            # Question patterns
            ("how do", 1.0),
            ("how to", 1.0),
            ("how should i", 1.0),
            ("how can i", 0.9),
            ("what's a good", 0.9),
            ("what is a good", 0.9),
            ("best way to", 0.9),
            ("best practice", 1.0),
            
            # Example patterns
            ("show me examples", 1.0),
            ("give me examples", 1.0),
            ("example of", 0.9),
            ("examples of", 0.9),
            
            # Comparison patterns
            ("compare", 0.9),
            ("comparison", 0.9),
            ("different approaches", 1.0),
            ("different ways", 0.9),
            
            # Sales/workflow specific patterns
            ("handle objections", 1.0),
            ("objection handling", 1.0),
            ("collect email", 1.0),
            ("email collection", 1.0),
            ("schedule meeting", 0.9),
            ("scheduling", 0.7),
            ("qualify leads", 1.0),
            ("lead qualification", 1.0),
            ("cold call", 0.9),
            ("cold calling", 0.9),
            ("opening line", 0.9),
            ("opening script", 0.9),
            ("close deals", 0.9),
            ("closing technique", 0.9),
            
            # General workflow patterns
            ("script for", 0.9),
            ("approach to", 0.8),
            ("technique for", 0.8),
            ("strategy for", 0.8),
            ("method for", 0.8),
            ("process for", 0.8),
            ("workflow for", 0.9),
        ],
        Intent.SYSTEM_COMMAND: [
            ("/", 1.0),
            ("help", 0.7),
            ("list", 0.6),
            ("status", 0.8),
            ("version", 0.7),
            ("config", 0.7),
            ("settings", 0.7),
        ],
    }

    def __init__(self) -> None:
        """Initialize the intent classifier.

        Compiles keyword patterns for efficient matching.
        """
        self._compiled_patterns: dict[Intent, list[tuple[re.Pattern, float]]] = {}
        for intent, keywords in self._INTENT_KEYWORDS.items():
            self._compiled_patterns[intent] = [
                (re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE), weight)
                for kw, weight in keywords
            ]
        # Special case for "/" at start of message
        self._system_command_pattern = re.compile(r"^/\w+")

    def classify(self, message: str) -> tuple[Intent, float]:
        """Classify a message into an intent with confidence score.

        Uses keyword matching to determine intent. Scores are computed based
        on the number and weight of matching keywords relative to message length.

        Args:
            message: The user message to classify.

        Returns:
            A tuple of (Intent, confidence) where confidence is 0.0-1.0.
            Returns (Intent.GENERAL_CHAT, 1.0) if no specific intent matches.

        Example:
            >>> classifier = IntentClassifier()
            >>> intent, conf = classifier.classify("build a customer support bot")
            >>> intent == Intent.CREATE_AGENT
            True
        """
        if not message or not message.strip():
            logger.debug("Empty message, defaulting to GENERAL_CHAT")
            return (Intent.GENERAL_CHAT, 1.0)

        message_lower = message.lower().strip()

        # Check for explicit system command (starts with /)
        if self._system_command_pattern.match(message_lower):
            logger.debug("Detected system command prefix '/'")
            return (Intent.SYSTEM_COMMAND, 1.0)

        # NEW: Check for workflow knowledge queries FIRST (before other intents)
        # This is important because workflow patterns can overlap with general queries
        for pattern, weight in self._compiled_patterns.get(Intent.WORKFLOW_KNOWLEDGE, []):
            if pattern.search(message_lower):
                logger.debug(f"Detected workflow knowledge pattern: {pattern.pattern}")
                return (Intent.WORKFLOW_KNOWLEDGE, 0.9)

        # Score each intent based on keyword matches
        intent_scores: dict[Intent, float] = {}
        for intent, patterns in self._compiled_patterns.items():
            # Skip WORKFLOW_KNOWLEDGE as we already checked it
            if intent == Intent.WORKFLOW_KNOWLEDGE:
                continue
                
            total_score = 0.0
            match_count = 0
            for pattern, weight in patterns:
                if pattern.search(message_lower):
                    total_score += weight
                    match_count += 1

            if match_count > 0:
                # Normalize by number of matches and boost for multiple matches
                intent_scores[intent] = total_score * (1 + 0.1 * (match_count - 1))

        if not intent_scores:
            logger.debug("No keyword matches, defaulting to GENERAL_CHAT")
            return (Intent.GENERAL_CHAT, 0.8)

        # Find best matching intent
        best_intent = max(intent_scores, key=intent_scores.get)  # type: ignore[arg-type]
        best_score = intent_scores[best_intent]

        # Normalize confidence to 0-1 range
        # Higher scores indicate more confident matches
        confidence = min(1.0, best_score / 2.0)

        logger.debug(
            "Classification: intent=%s, confidence=%.2f, scores=%s",
            best_intent.value,
            confidence,
            {k.value: round(v, 2) for k, v in intent_scores.items()},
        )

        return (best_intent, confidence)


# =============================================================================
# Complexity Assessor
# =============================================================================


class ComplexityAssessor:
    """Assesses the complexity of a request on a 0.0-1.0 scale.

    Complexity is determined by factors like:
    - Number of words in the message
    - Presence of tool-specific keywords
    - Presence of multi-step indicators
    - Intent-specific patterns

    The complexity score is used to determine:
    - Which reasoning strategy to use (direct, CoT, ToT, etc.)
    - Whether to engage planning subsystem
    - Whether to use contract verification

    Example:
        >>> assessor = ComplexityAssessor()
        >>> complexity = assessor.assess("run the sales agent", Intent.RUN_AGENT)
        >>> 0.0 <= complexity <= 1.0
        True
    """

    # Tool-related keywords that indicate complexity
    _TOOL_KEYWORDS = [
        "search",
        "find",
        "analyze",
        "compare",
        "create",
        "generate",
        "schedule",
        "send",
        "calculate",
        "fetch",
        "retrieve",
    ]

    # Multi-step indicators
    _MULTI_STEP_KEYWORDS = [
        "then",
        "after",
        "next",
        "finally",
        "first",
        "second",
        "also",
        "additionally",
        "and then",
        "before",
    ]

    def assess(self, message: str, intent: Intent) -> float:
        """Assess the complexity of a message.

        Args:
            message: The user message to assess.
            intent: The classified intent of the message.

        Returns:
            A complexity score from 0.0 to 1.0.

        Example:
            >>> assessor = ComplexityAssessor()
            >>> complexity = assessor.assess("search the web and create a report", Intent.RUN_AGENT)
            >>> complexity > 0.5
            True
        """
        if not message:
            return 0.0

        message_lower = message.lower()
        complexity = 0.0

        # Base complexity from message length
        word_count = len(message.split())
        if word_count <= 5:
            complexity += 0.1
        elif word_count <= 10:
            complexity += 0.2
        elif word_count <= 20:
            complexity += 0.3
        else:
            complexity += 0.4

        # Check for tool keywords
        tool_count = sum(1 for kw in self._TOOL_KEYWORDS if kw in message_lower)
        complexity += min(0.3, tool_count * 0.1)

        # Check for multi-step indicators
        multi_step_count = sum(1 for kw in self._MULTI_STEP_KEYWORDS if kw in message_lower)
        complexity += min(0.2, multi_step_count * 0.1)

        # Intent-specific adjustments
        if intent == Intent.CREATE_AGENT:
            complexity += 0.2  # Agent creation is inherently complex
        elif intent == Intent.MODIFY_AGENT:
            complexity += 0.15  # Modification needs understanding existing agent
        elif intent == Intent.WORKFLOW_KNOWLEDGE:  # NEW: Workflow knowledge complexity
            complexity += 0.25  # Requires semantic search and RAG

        # Cap at 1.0
        return min(1.0, complexity)


# =============================================================================
# Route Decision
# =============================================================================


@dataclass
class RouteDecision:
    """Dataclass containing the complete routing decision.

    This is the output of the Router.route() method and contains all
    information needed to execute the request.

    Attributes:
        intent: The classified intent.
        confidence: Confidence score for the intent classification (0.0-1.0).
        complexity: Complexity score for the request (0.0-1.0).
        handler_name: Name of the handler to execute the request.
        use_planning: Whether to engage the planning subsystem.
        use_memory: Whether to engage the memory subsystem.
        use_contracts: Whether to engage the contract verification subsystem.
        metadata: Additional metadata about the routing decision.

    Example:
        >>> decision = RouteDecision(
        ...     intent=Intent.CREATE_AGENT,
        ...     confidence=0.95,
        ...     complexity=0.7,
        ...     handler_name="builder",
        ...     use_planning=True,
        ...     use_memory=False,
        ...     use_contracts=True,
        ... )
        >>> decision.handler_name
        'builder'
    """

    intent: Intent
    confidence: float
    complexity: float
    handler_name: str
    use_planning: bool = False
    use_memory: bool = False
    use_contracts: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)  # type: ignore[misc]

    def to_dict(self) -> dict[str, Any]:  # type: ignore[misc]
        """Convert the routing decision to a dictionary.

        Returns:
            Dictionary representation of the routing decision.
        """
        return {
            "intent": self.intent.value,
            "confidence": self.confidence,
            "complexity": self.complexity,
            "handler_name": self.handler_name,
            "use_planning": self.use_planning,
            "use_memory": self.use_memory,
            "use_contracts": self.use_contracts,
            "metadata": self.metadata,
        }


# =============================================================================
# Router
# =============================================================================


class Router:
    """Main router class that orchestrates intent classification and routing.

    The router combines intent classification and complexity assessment to
    make routing decisions. It respects feature flags from settings to
    determine which subsystems can be engaged.

    Handler Mapping:
        - CREATE_AGENT -> "builder"
        - RUN_AGENT -> "executor"
        - QUERY_MEMORY -> "memory_query"
        - MODIFY_AGENT -> "agent_modifier"
        - WORKFLOW_KNOWLEDGE -> "workflow_knowledge"
        - SYSTEM_COMMAND -> "system_command"
        - GENERAL_CHAT -> "chat"

    Subsystem Rules:
        - Planning: Enabled if complexity > 0.5 AND use_planning flag is True
        - Memory: Enabled if needs context AND use_memory flag is True
        - Contracts: Enabled if complexity > 0.7 AND use_contracts flag is True

    Attributes:
        settings: The SigilSettings instance for feature flag access.
        classifier: The IntentClassifier instance.
        assessor: The ComplexityAssessor instance.

    Example:
        >>> from sigil.config import get_settings
        >>> router = Router(get_settings())
        >>> decision = router.route("build a new agent for web scraping")
        >>> decision.handler_name
        'builder'
    """

    # Intent to handler mapping
    _HANDLER_MAP: dict[Intent, str] = {
        Intent.CREATE_AGENT: "builder",
        Intent.RUN_AGENT: "executor",
        Intent.QUERY_MEMORY: "memory_query",
        Intent.MODIFY_AGENT: "agent_modifier",
        Intent.WORKFLOW_KNOWLEDGE: "workflow_knowledge",  # NEW: Handler for workflow knowledge
        Intent.SYSTEM_COMMAND: "system_command",
        Intent.GENERAL_CHAT: "chat",
    }

    # Intents that benefit from memory context
    _MEMORY_RELEVANT_INTENTS = {
        Intent.CREATE_AGENT,  # May need past agent patterns
        Intent.RUN_AGENT,  # May need agent context
        Intent.QUERY_MEMORY,  # Obviously needs memory
        Intent.MODIFY_AGENT,  # Needs existing agent state
        # NOTE: WORKFLOW_KNOWLEDGE does NOT use memory - it uses semantic search
    }

    def __init__(self, settings: "SigilSettings") -> None:
        """Initialize the router with settings.

        Args:
            settings: SigilSettings instance for feature flag access.
        """
        self.settings = settings
        self.classifier = IntentClassifier()
        self.assessor = ComplexityAssessor()
        logger.debug("Router initialized with feature flags: %s", settings.get_active_features())

    def route(self, message: str) -> RouteDecision:
        """Route a user message to the appropriate handler.

        Performs intent classification, complexity assessment, and determines
        which subsystems to engage based on settings and complexity.

        Args:
            message: The user message to route.

        Returns:
            A RouteDecision containing the complete routing decision.

        Example:
            >>> router = Router(get_settings())
            >>> decision = router.route("run the sales bot with John from Acme")
            >>> decision.intent
            <Intent.RUN_AGENT: 'run_agent'>
            >>> decision.handler_name
            'executor'
        """
        # Step 1: Classify intent
        intent, confidence = self.classifier.classify(message)

        # Step 2: Assess complexity
        complexity = self.assessor.assess(message, intent)

        # Step 3: Select handler
        handler_name = self._HANDLER_MAP.get(intent, "chat")

        # Step 4: Determine subsystem flags
        use_planning = self._should_use_planning(complexity, message, intent)
        use_memory = self._should_use_memory(intent, complexity)
        use_contracts = self._should_use_contracts(complexity)

        # Create decision
        decision = RouteDecision(
            intent=intent,
            confidence=confidence,
            complexity=complexity,
            handler_name=handler_name,
            use_planning=use_planning,
            use_memory=use_memory,
            use_contracts=use_contracts,
            metadata={
                "message_length": len(message),
                "routing_enabled": self.settings.use_routing,
            },
        )

        # Log the decision at DEBUG level
        logger.debug(
            "Route decision: intent=%s, confidence=%.2f, complexity=%.2f, "
            "handler=%s, planning=%s, memory=%s, contracts=%s",
            intent.value,
            confidence,
            complexity,
            handler_name,
            use_planning,
            use_memory,
            use_contracts,
        )

        return decision

    def _should_use_planning(self, complexity: float, message: str = "", intent: Intent = None) -> bool:  # type: ignore[assignment]
        """Determine if planning subsystem should be engaged.

        Planning is enabled if:
        - The use_planning feature flag is True
        - AND either:
          - The complexity score is greater than 0.2, OR
          - The message contains tool-related keywords, OR
          - The intent is WORKFLOW_KNOWLEDGE (needs tool execution)

        Args:
            complexity: The assessed complexity score.
            message: The user message to check for tool keywords.
            intent: The classified intent.

        Returns:
            True if planning should be used.
        """
        if not self.settings.use_planning:
            return False

        # Always use planning for workflow knowledge queries
        if intent == Intent.WORKFLOW_KNOWLEDGE:
            return True

        # Check for tool keywords - if found, enable planning regardless of complexity
        message_lower = message.lower()
        assessor = self.assessor  # Access the shared assessor with tool keywords
        for keyword in assessor._TOOL_KEYWORDS:
            if keyword in message_lower:
                return True

        # Otherwise, use complexity threshold (lowered from 0.5 to 0.2)
        return complexity > 0.2

    def _should_use_memory(self, intent: Intent, complexity: float) -> bool:
        """Determine if memory subsystem should be engaged.

        Memory is enabled if:
        - The use_memory feature flag is True
        - The intent benefits from memory context

        Note: WORKFLOW_KNOWLEDGE does NOT use memory - it uses semantic search instead

        Args:
            intent: The classified intent.
            complexity: The assessed complexity score (unused but available).

        Returns:
            True if memory should be used.
        """
        return self.settings.use_memory and intent in self._MEMORY_RELEVANT_INTENTS

    def _should_use_contracts(self, complexity: float) -> bool:
        """Determine if contract verification should be engaged.

        Contracts are enabled if:
        - The use_contracts feature flag is True
        - The complexity score is greater than 0.7

        Args:
            complexity: The assessed complexity score.

        Returns:
            True if contracts should be used.
        """
        return self.settings.use_contracts and complexity > 0.7

    def get_handler_for_intent(self, intent: Intent) -> str:
        """Get the handler name for a specific intent.

        Utility method for looking up handler names.

        Args:
            intent: The intent to look up.

        Returns:
            The handler name for the intent.
        """
        return self._HANDLER_MAP.get(intent, "chat")


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "Intent",
    "IntentClassifier",
    "ComplexityAssessor",
    "RouteDecision",
    "Router",
]
