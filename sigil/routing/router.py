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
        SYSTEM_COMMAND: User issues a system command (e.g., /help, /status).
        GENERAL_CHAT: General conversation that doesn't fit other categories.
    """

    CREATE_AGENT = "create_agent"
    RUN_AGENT = "run_agent"
    QUERY_MEMORY = "query_memory"
    MODIFY_AGENT = "modify_agent"
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

        # Score each intent based on keyword matches
        intent_scores: dict[Intent, float] = {}
        for intent, patterns in self._compiled_patterns.items():
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

        # Calculate confidence based on score magnitude and margin
        # Higher scores and clearer margins = higher confidence
        second_best_score = 0.0
        for intent, score in intent_scores.items():
            if intent != best_intent and score > second_best_score:
                second_best_score = score

        margin = best_score - second_best_score
        # Confidence based on absolute score (capped at 3.0) and margin
        base_confidence = min(best_score / 3.0, 1.0)
        margin_boost = min(margin / 2.0, 0.2)
        confidence = min(base_confidence + margin_boost, 1.0)

        logger.debug(
            f"Classified intent: {best_intent.value} "
            f"(score={best_score:.2f}, confidence={confidence:.2f})"
        )

        return (best_intent, confidence)


# =============================================================================
# Complexity Assessor
# =============================================================================


class ComplexityAssessor:
    """Assesses the complexity of a user request.

    Complexity is scored on a 0.0 (trivial) to 1.0 (critical) scale based on
    multiple factors. The score influences which subsystems (planning, memory,
    contracts) are engaged during request handling.

    Complexity Factors:
        - Message length: Longer messages tend to be more complex.
        - Tool requirements: Requests mentioning more tools are more complex.
        - Domain specificity: Specialized vocabulary indicates complexity.
        - Decision complexity: Multiple choices/options increase complexity.

    Each factor is normalized to 0.0-1.0 and weighted equally (0.25 each).

    Example:
        >>> assessor = ComplexityAssessor()
        >>> score = assessor.assess("hello", Intent.GENERAL_CHAT)
        >>> score < 0.3
        True
        >>> complex_msg = "Create an agent that can search the web, access CRM, "
        >>> complex_msg += "schedule meetings, and send follow-up emails with "
        >>> complex_msg += "personalized content based on lead qualification."
        >>> score = assessor.assess(complex_msg, Intent.CREATE_AGENT)
        >>> score > 0.5
        True
    """

    # Domain-specific vocabulary indicators
    _DOMAIN_KEYWORDS = [
        "qualification", "bant", "pipeline", "funnel", "conversion",
        "compliance", "regulatory", "hipaa", "gdpr", "security",
        "authentication", "authorization", "oauth", "jwt", "token",
        "embedding", "vector", "semantic", "rag", "llm", "prompt",
        "integration", "middleware", "orchestration", "workflow",
        "sentiment", "classification", "entity", "extraction",
    ]

    # Decision complexity indicators
    _DECISION_KEYWORDS = [
        "if", "when", "unless", "otherwise", "depending", "based on",
        "choose", "decide", "select", "option", "alternative",
        "either", "or", "both", "multiple", "various", "different",
        "prioritize", "rank", "compare", "evaluate", "assess",
    ]

    # Maximum message length for normalization
    _MAX_LENGTH = 500

    def __init__(self) -> None:
        """Initialize the complexity assessor.

        Compiles regex patterns for efficient keyword matching.
        """
        self._domain_pattern = re.compile(
            r"\b(" + "|".join(re.escape(kw) for kw in self._DOMAIN_KEYWORDS) + r")\b",
            re.IGNORECASE,
        )
        self._decision_pattern = re.compile(
            r"\b(" + "|".join(re.escape(kw) for kw in self._DECISION_KEYWORDS) + r")\b",
            re.IGNORECASE,
        )

    def assess(self, message: str, intent: Intent) -> float:
        """Assess the complexity of a message.

        Args:
            message: The user message to assess.
            intent: The classified intent (may influence complexity).

        Returns:
            Complexity score from 0.0 (trivial) to 1.0 (critical).

        Example:
            >>> assessor = ComplexityAssessor()
            >>> assessor.assess("hi", Intent.GENERAL_CHAT)
            0.1  # Very simple
            >>> assessor.assess(
            ...     "Create a multi-step workflow with CRM integration",
            ...     Intent.CREATE_AGENT
            ... ) > 0.5
            True
        """
        if not message or not message.strip():
            return 0.0

        message_lower = message.lower().strip()

        # Factor 1: Message length (0.0 - 1.0)
        length_factor = min(len(message) / self._MAX_LENGTH, 1.0)

        # Factor 2: Domain specificity (0.0 - 1.0)
        domain_matches = self._domain_pattern.findall(message_lower)
        # Cap at 4 domain terms for max score
        unique_domains = len(set(domain_matches))
        domain_factor = min(unique_domains / 4.0, 1.0)

        # Factor 3: Decision complexity (0.0 - 1.0)
        decision_matches = self._decision_pattern.findall(message_lower)
        # Cap at 4 decision indicators for max score
        unique_decisions = len(set(decision_matches))
        decision_factor = min(unique_decisions / 4.0, 1.0)

        # Weight each factor (3 factors now)
        complexity = (
            length_factor * 0.34
            + domain_factor * 0.33
            + decision_factor * 0.33
        )

        # Intent-based adjustments
        # CREATE and MODIFY tend to be more complex than RUN
        if intent in (Intent.CREATE_AGENT, Intent.MODIFY_AGENT):
            complexity = min(complexity * 1.2, 1.0)  # 20% boost
        elif intent == Intent.SYSTEM_COMMAND:
            complexity = min(complexity * 0.5, 0.3)  # Cap at 0.3 for system commands
        elif intent == Intent.GENERAL_CHAT:
            complexity = min(complexity * 0.8, 0.5)  # Cap at 0.5 for chat

        logger.debug(
            f"Complexity assessment: {complexity:.2f} "
            f"(length={length_factor:.2f}, "
            f"domain={domain_factor:.2f}, decision={decision_factor:.2f})"
        )

        return round(complexity, 3)


# =============================================================================
# Route Decision
# =============================================================================


@dataclass
class RouteDecision:
    """Complete routing decision for a user request.

    Contains all information needed to route a request to the appropriate
    handler with the correct subsystem configuration.

    Attributes:
        intent: The classified intent of the request.
        confidence: Confidence score for the intent classification (0.0-1.0).
        complexity: Complexity score of the request (0.0-1.0).
        handler_name: Name of the handler to route to.
        use_planning: Whether to engage the planning subsystem.
        use_memory: Whether to engage the memory subsystem.
        use_contracts: Whether to engage the contract verification subsystem.
        metadata: Additional routing metadata.

    Example:
        >>> decision = RouteDecision(
        ...     intent=Intent.CREATE_AGENT,
        ...     confidence=0.95,
        ...     complexity=0.75,
        ...     handler_name="builder",
        ...     use_planning=True,
        ...     use_memory=True,
        ...     use_contracts=True,
        ... )
        >>> decision.should_use_advanced_features()
        True
    """

    intent: Intent
    confidence: float
    complexity: float
    handler_name: str
    use_planning: bool = False
    use_memory: bool = False
    use_contracts: bool = False
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate decision values after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0.0-1.0, got {self.confidence}")
        if not 0.0 <= self.complexity <= 1.0:
            raise ValueError(f"Complexity must be 0.0-1.0, got {self.complexity}")

    def should_use_advanced_features(self) -> bool:
        """Check if any advanced subsystems are engaged.

        Returns:
            True if planning, memory, or contracts are enabled.
        """
        return self.use_planning or self.use_memory or self.use_contracts

    def to_dict(self) -> dict:
        """Convert decision to dictionary for serialization.

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
        Intent.SYSTEM_COMMAND: "system_command",
        Intent.GENERAL_CHAT: "chat",
    }

    # Intents that benefit from memory context
    _MEMORY_RELEVANT_INTENTS = {
        Intent.CREATE_AGENT,  # May need past agent patterns
        Intent.RUN_AGENT,  # May need agent context
        Intent.QUERY_MEMORY,  # Obviously needs memory
        Intent.MODIFY_AGENT,  # Needs existing agent state
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
        use_planning = self._should_use_planning(message, complexity)
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

    def _should_use_planning(self, message: str, complexity: float) -> bool:
        """Determine if planning is needed based on complexity only.

        Planning is enabled if:
        - The use_planning feature flag is True
        - AND the complexity score is greater than 0.3

        The LLM will decide if tools are needed - no keyword matching here.

        Args:
            message: The user message (unused, kept for API compatibility).
            complexity: The assessed complexity score.

        Returns:
            True if planning should be used.
        """
        # LLM will decide if tools are needed - no keyword matching
        return self.settings.use_planning and complexity > 0.3

    def _should_use_memory(self, intent: Intent, complexity: float) -> bool:
        """Determine if memory subsystem should be engaged.

        Memory is enabled if:
        - The use_memory feature flag is True
        - The intent benefits from memory context

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
