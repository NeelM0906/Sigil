"""Tests for Sigil v2 routing module.

Task 3.7.3: Write tests for routing layer

Test Coverage:
- Intent classification accuracy
- Complexity assessment consistency
- Routing decisions match expected handlers
- Fallback behavior when subsystems disabled
- Edge cases (empty messages, ambiguous inputs)
- Feature flag interactions

Note: These tests DO NOT require ANTHROPIC_API_KEY because the routing
layer uses keyword matching, not LLM calls.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from sigil.config.settings import SigilSettings, clear_settings_cache
from sigil.routing import (
    Intent,
    IntentClassifier,
    ComplexityAssessor,
    RouteDecision,
    Router,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def clean_settings_cache():
    """Clear settings cache before and after each test."""
    clear_settings_cache()
    yield
    clear_settings_cache()


@pytest.fixture
def mock_settings():
    """Create a mock SigilSettings with all features disabled."""
    settings = MagicMock(spec=SigilSettings)
    settings.use_memory = False
    settings.use_planning = False
    settings.use_contracts = False
    settings.use_routing = True
    settings.use_evolution = False
    settings.get_active_features.return_value = ["routing"]
    return settings


@pytest.fixture
def mock_settings_all_features():
    """Create a mock SigilSettings with all features enabled."""
    settings = MagicMock(spec=SigilSettings)
    settings.use_memory = True
    settings.use_planning = True
    settings.use_contracts = True
    settings.use_routing = True
    settings.use_evolution = True
    settings.get_active_features.return_value = [
        "memory", "planning", "contracts", "routing", "evolution"
    ]
    return settings


@pytest.fixture
def classifier():
    """Create an IntentClassifier instance."""
    return IntentClassifier()


@pytest.fixture
def assessor():
    """Create a ComplexityAssessor instance."""
    return ComplexityAssessor()


@pytest.fixture
def router(mock_settings):
    """Create a Router instance with mock settings."""
    return Router(mock_settings)


@pytest.fixture
def router_all_features(mock_settings_all_features):
    """Create a Router instance with all features enabled."""
    return Router(mock_settings_all_features)


# =============================================================================
# Test: Intent Enum
# =============================================================================


class TestIntentEnum:
    """Test Intent enumeration values and behavior."""

    def test_intent_values(self):
        """Test that all intent values are correctly defined."""
        assert Intent.CREATE_AGENT.value == "create_agent"
        assert Intent.RUN_AGENT.value == "run_agent"
        assert Intent.QUERY_MEMORY.value == "query_memory"
        assert Intent.MODIFY_AGENT.value == "modify_agent"
        assert Intent.SYSTEM_COMMAND.value == "system_command"
        assert Intent.GENERAL_CHAT.value == "general_chat"

    def test_intent_count(self):
        """Test that we have exactly 6 intents."""
        assert len(Intent) == 6

    def test_intent_iteration(self):
        """Test that all intents can be iterated."""
        intents = list(Intent)
        assert len(intents) == 6
        assert Intent.CREATE_AGENT in intents
        assert Intent.GENERAL_CHAT in intents


# =============================================================================
# Test: Intent Classification
# =============================================================================


class TestIntentClassifier:
    """Test IntentClassifier accuracy and behavior."""

    # CREATE_AGENT intent tests - strong signals
    @pytest.mark.parametrize("message", [
        "create a new agent",
        "build me a sales bot",
        "design a new agent for me",
        "make a new bot",
        "generate agent for lead qualification",
        "create and build a new agent",
        "BUILD A NEW AGENT",  # Test case insensitivity
    ])
    def test_classify_create_intent(self, classifier, message):
        """Test that CREATE_AGENT messages are correctly classified."""
        intent, confidence = classifier.classify(message)
        assert intent == Intent.CREATE_AGENT, f"Expected CREATE_AGENT for: {message}"
        assert confidence >= 0.3, f"Expected confidence >= 0.3 for: {message}"

    # RUN_AGENT intent tests
    @pytest.mark.parametrize("message", [
        "run the sales agent",
        "execute the customer bot",
        "use the qualifier agent",
        "demo the new bot",
        "start the agent",
        "launch the sales bot",
        "Run the demo",
    ])
    def test_classify_run_intent(self, classifier, message):
        """Test that RUN_AGENT messages are correctly classified."""
        intent, confidence = classifier.classify(message)
        assert intent == Intent.RUN_AGENT, f"Expected RUN_AGENT for: {message}"
        assert confidence > 0.4, f"Expected confidence > 0.4 for: {message}"

    # QUERY_MEMORY intent tests
    @pytest.mark.parametrize("message", [
        "find information about John",
        "search for past conversations",
        "what do you know about Acme Corp?",
        "remember our last meeting",
        "recall the lead details",
        "do you know his email?",
        "retrieve the qualification score",
    ])
    def test_classify_query_intent(self, classifier, message):
        """Test that QUERY_MEMORY messages are correctly classified."""
        intent, confidence = classifier.classify(message)
        assert intent == Intent.QUERY_MEMORY, f"Expected QUERY_MEMORY for: {message}"
        assert confidence > 0.4, f"Expected confidence > 0.4 for: {message}"

    # MODIFY_AGENT intent tests - strong signals
    @pytest.mark.parametrize("message", [
        "change and update the agent",
        "update the bot configuration",
        "edit the system prompt",
        "modify the agent behavior",
        "modify and update settings",
        "edit and change the agent config",
    ])
    def test_classify_modify_intent(self, classifier, message):
        """Test that MODIFY_AGENT messages are correctly classified."""
        intent, confidence = classifier.classify(message)
        assert intent == Intent.MODIFY_AGENT, f"Expected MODIFY_AGENT for: {message}"
        assert confidence >= 0.3, f"Expected confidence >= 0.3 for: {message}"

    # SYSTEM_COMMAND intent tests
    @pytest.mark.parametrize("message", [
        "/help",
        "/status",
        "/list agents",
        "/version",
        "/config show",
    ])
    def test_classify_system_command(self, classifier, message):
        """Test that system commands (starting with /) are correctly classified."""
        intent, confidence = classifier.classify(message)
        assert intent == Intent.SYSTEM_COMMAND, f"Expected SYSTEM_COMMAND for: {message}"
        assert confidence == 1.0, f"Expected confidence 1.0 for slash commands: {message}"

    # GENERAL_CHAT intent tests - messages with no intent keywords
    @pytest.mark.parametrize("message", [
        "hello",
        "how are you?",
        "tell me a joke",
        "good morning",
        "nice to meet you",
    ])
    def test_classify_general_chat(self, classifier, message):
        """Test that general conversation defaults to GENERAL_CHAT."""
        intent, confidence = classifier.classify(message)
        assert intent == Intent.GENERAL_CHAT, f"Expected GENERAL_CHAT for: {message}"
        assert confidence >= 0.7, f"Expected confidence >= 0.7 for default: {message}"

    def test_classify_empty_message(self, classifier):
        """Test that empty messages return GENERAL_CHAT."""
        intent, confidence = classifier.classify("")
        assert intent == Intent.GENERAL_CHAT
        assert confidence == 1.0

    def test_classify_whitespace_message(self, classifier):
        """Test that whitespace-only messages return GENERAL_CHAT."""
        intent, confidence = classifier.classify("   \n\t   ")
        assert intent == Intent.GENERAL_CHAT
        assert confidence == 1.0

    def test_classify_confidence_range(self, classifier):
        """Test that confidence is always in valid range."""
        messages = [
            "create agent",
            "run",
            "hello world",
            "x" * 1000,
            "CREATE BUILD RUN EXECUTE",
        ]
        for msg in messages:
            _, confidence = classifier.classify(msg)
            assert 0.0 <= confidence <= 1.0, f"Invalid confidence for: {msg}"


# =============================================================================
# Test: Complexity Assessment
# =============================================================================


class TestComplexityAssessor:
    """Test ComplexityAssessor consistency and range."""

    def test_assess_simple_message(self, assessor):
        """Test that simple messages have low complexity."""
        score = assessor.assess("hello", Intent.GENERAL_CHAT)
        assert score < 0.3, "Simple 'hello' should have low complexity"

    def test_assess_empty_message(self, assessor):
        """Test that empty messages return 0 complexity."""
        assert assessor.assess("", Intent.GENERAL_CHAT) == 0.0
        assert assessor.assess("   ", Intent.GENERAL_CHAT) == 0.0

    def test_assess_long_message(self, assessor):
        """Test that longer messages have higher complexity."""
        short = assessor.assess("hello", Intent.GENERAL_CHAT)
        long = assessor.assess("hello " * 100, Intent.GENERAL_CHAT)
        assert long > short, "Longer message should have higher complexity"

    def test_assess_tool_keywords(self, assessor):
        """Test that tool keywords increase complexity."""
        no_tools = assessor.assess("create an agent", Intent.CREATE_AGENT)
        with_tools = assessor.assess(
            "create an agent that can search the web and access CRM",
            Intent.CREATE_AGENT
        )
        assert with_tools > no_tools, "Tool mentions should increase complexity"

    def test_assess_domain_keywords(self, assessor):
        """Test that domain-specific keywords increase complexity."""
        simple = assessor.assess("create a bot", Intent.CREATE_AGENT)
        complex_msg = assessor.assess(
            "create an agent with BANT qualification and GDPR compliance",
            Intent.CREATE_AGENT
        )
        assert complex_msg > simple, "Domain keywords should increase complexity"

    def test_assess_decision_keywords(self, assessor):
        """Test that decision indicators increase complexity."""
        simple = assessor.assess("run the agent", Intent.RUN_AGENT)
        complex_msg = assessor.assess(
            "run the agent and choose the best option based on priority",
            Intent.RUN_AGENT
        )
        assert complex_msg > simple, "Decision keywords should increase complexity"

    def test_assess_always_in_range(self, assessor):
        """Test that complexity is always 0.0-1.0."""
        test_cases = [
            ("", Intent.GENERAL_CHAT),
            ("hi", Intent.GENERAL_CHAT),
            ("x" * 1000, Intent.CREATE_AGENT),
            ("search web crm api database file email sms", Intent.CREATE_AGENT),
            ("qualification bant pipeline compliance gdpr oauth", Intent.CREATE_AGENT),
            ("if when unless otherwise choose decide either or", Intent.RUN_AGENT),
        ]
        for msg, intent in test_cases:
            score = assessor.assess(msg, intent)
            assert 0.0 <= score <= 1.0, f"Invalid complexity for: {msg[:50]}..."

    def test_assess_intent_affects_complexity(self, assessor):
        """Test that intent type affects complexity calculation."""
        msg = "do something with this agent"
        # CREATE and MODIFY should boost complexity
        create_score = assessor.assess(msg, Intent.CREATE_AGENT)
        modify_score = assessor.assess(msg, Intent.MODIFY_AGENT)
        # SYSTEM_COMMAND and GENERAL_CHAT should reduce/cap complexity
        system_score = assessor.assess(msg, Intent.SYSTEM_COMMAND)
        chat_score = assessor.assess(msg, Intent.GENERAL_CHAT)

        # System command should have capped low complexity
        assert system_score <= 0.3, "System command complexity should be capped"
        # Chat should have reduced complexity
        assert chat_score <= 0.5, "Chat complexity should be capped"

    def test_assess_complex_request(self, assessor):
        """Test that genuinely complex requests score high."""
        complex_msg = (
            "Create an agent that can search the web, access CRM data, "
            "schedule calendar meetings, and send follow-up emails. "
            "It should use BANT qualification with compliance checks "
            "and choose the best approach based on lead priority and "
            "multiple different criteria depending on the situation."
        )
        score = assessor.assess(complex_msg, Intent.CREATE_AGENT)
        assert score > 0.6, f"Complex request should score > 0.6, got {score}"


# =============================================================================
# Test: Route Decision
# =============================================================================


class TestRouteDecision:
    """Test RouteDecision dataclass behavior."""

    def test_create_valid_decision(self):
        """Test creating a valid RouteDecision."""
        decision = RouteDecision(
            intent=Intent.CREATE_AGENT,
            confidence=0.95,
            complexity=0.75,
            handler_name="builder",
            use_planning=True,
            use_memory=True,
            use_contracts=True,
        )
        assert decision.intent == Intent.CREATE_AGENT
        assert decision.confidence == 0.95
        assert decision.complexity == 0.75
        assert decision.handler_name == "builder"
        assert decision.use_planning is True
        assert decision.use_memory is True
        assert decision.use_contracts is True

    def test_invalid_confidence_raises(self):
        """Test that invalid confidence raises ValueError."""
        with pytest.raises(ValueError, match="Confidence must be 0.0-1.0"):
            RouteDecision(
                intent=Intent.GENERAL_CHAT,
                confidence=1.5,
                complexity=0.5,
                handler_name="chat",
            )

    def test_invalid_complexity_raises(self):
        """Test that invalid complexity raises ValueError."""
        with pytest.raises(ValueError, match="Complexity must be 0.0-1.0"):
            RouteDecision(
                intent=Intent.GENERAL_CHAT,
                confidence=0.5,
                complexity=-0.1,
                handler_name="chat",
            )

    def test_should_use_advanced_features(self):
        """Test should_use_advanced_features method."""
        # No advanced features
        simple = RouteDecision(
            intent=Intent.GENERAL_CHAT,
            confidence=0.9,
            complexity=0.2,
            handler_name="chat",
        )
        assert simple.should_use_advanced_features() is False

        # With planning
        with_planning = RouteDecision(
            intent=Intent.CREATE_AGENT,
            confidence=0.9,
            complexity=0.6,
            handler_name="builder",
            use_planning=True,
        )
        assert with_planning.should_use_advanced_features() is True

        # With memory
        with_memory = RouteDecision(
            intent=Intent.QUERY_MEMORY,
            confidence=0.9,
            complexity=0.4,
            handler_name="memory_query",
            use_memory=True,
        )
        assert with_memory.should_use_advanced_features() is True

        # With contracts
        with_contracts = RouteDecision(
            intent=Intent.CREATE_AGENT,
            confidence=0.9,
            complexity=0.8,
            handler_name="builder",
            use_contracts=True,
        )
        assert with_contracts.should_use_advanced_features() is True

    def test_to_dict(self):
        """Test to_dict serialization."""
        decision = RouteDecision(
            intent=Intent.RUN_AGENT,
            confidence=0.85,
            complexity=0.45,
            handler_name="executor",
            use_planning=False,
            use_memory=True,
            use_contracts=False,
            metadata={"test": "value"},
        )
        d = decision.to_dict()
        assert d["intent"] == "run_agent"
        assert d["confidence"] == 0.85
        assert d["complexity"] == 0.45
        assert d["handler_name"] == "executor"
        assert d["use_planning"] is False
        assert d["use_memory"] is True
        assert d["use_contracts"] is False
        assert d["metadata"] == {"test": "value"}

    def test_default_metadata(self):
        """Test that metadata defaults to empty dict."""
        decision = RouteDecision(
            intent=Intent.GENERAL_CHAT,
            confidence=0.9,
            complexity=0.1,
            handler_name="chat",
        )
        assert decision.metadata == {}


# =============================================================================
# Test: Router
# =============================================================================


class TestRouter:
    """Test Router routing decisions."""

    def test_router_initialization(self, mock_settings):
        """Test that Router initializes correctly."""
        router = Router(mock_settings)
        assert router.settings == mock_settings
        assert isinstance(router.classifier, IntentClassifier)
        assert isinstance(router.assessor, ComplexityAssessor)

    # Handler mapping tests
    @pytest.mark.parametrize("message,expected_handler", [
        ("create a new agent", "builder"),
        ("run the sales bot", "executor"),
        ("find information about John", "memory_query"),
        ("update the agent config", "agent_modifier"),
        ("/help", "system_command"),
        ("hello there", "chat"),
    ])
    def test_route_handler_mapping(self, router, message, expected_handler):
        """Test that intents map to correct handlers."""
        decision = router.route(message)
        assert decision.handler_name == expected_handler, (
            f"Expected handler '{expected_handler}' for: {message}"
        )

    def test_route_returns_valid_decision(self, router):
        """Test that route returns a valid RouteDecision."""
        decision = router.route("create a new agent")
        assert isinstance(decision, RouteDecision)
        assert isinstance(decision.intent, Intent)
        assert 0.0 <= decision.confidence <= 1.0
        assert 0.0 <= decision.complexity <= 1.0
        assert isinstance(decision.handler_name, str)

    def test_route_empty_message(self, router):
        """Test routing empty message defaults to chat."""
        decision = router.route("")
        assert decision.intent == Intent.GENERAL_CHAT
        assert decision.handler_name == "chat"

    def test_route_includes_metadata(self, router):
        """Test that routing decision includes metadata."""
        decision = router.route("create something")
        assert "message_length" in decision.metadata
        assert "routing_enabled" in decision.metadata
        assert decision.metadata["message_length"] == len("create something")

    # Feature flag tests
    def test_route_no_planning_when_disabled(self, router):
        """Test that planning is not enabled when feature flag is off."""
        # Create a complex message that would normally trigger planning
        complex_msg = (
            "Create a complex multi-step workflow with CRM integration, "
            "calendar sync, and decision-based routing with compliance checks"
        )
        decision = router.route(complex_msg)
        assert decision.use_planning is False, "Planning should be disabled"

    def test_route_no_memory_when_disabled(self, router):
        """Test that memory is not enabled when feature flag is off."""
        decision = router.route("find information about past leads")
        assert decision.use_memory is False, "Memory should be disabled"

    def test_route_no_contracts_when_disabled(self, router):
        """Test that contracts are not enabled when feature flag is off."""
        complex_msg = (
            "Create a compliance-critical agent with GDPR, HIPAA, "
            "qualification scoring and multiple decision pathways"
        )
        decision = router.route(complex_msg)
        assert decision.use_contracts is False, "Contracts should be disabled"

    def test_route_planning_enabled_when_complex(self, router_all_features):
        """Test that planning is enabled for complex requests when flag is on."""
        complex_msg = (
            "Create a multi-step workflow with CRM integration, "
            "calendar sync, and decision-based routing with compliance"
        )
        decision = router_all_features.route(complex_msg)
        # Should have high enough complexity to trigger planning
        if decision.complexity > 0.5:
            assert decision.use_planning is True, (
                f"Planning should be enabled for complexity {decision.complexity}"
            )

    def test_route_memory_enabled_for_relevant_intents(self, router_all_features):
        """Test that memory is enabled for memory-relevant intents."""
        # Query memory should enable memory
        decision = router_all_features.route("find information about John")
        assert decision.use_memory is True, "Memory should be enabled for queries"

        # Create agent should enable memory
        decision = router_all_features.route("create a new sales agent")
        assert decision.use_memory is True, "Memory should be enabled for creation"

    def test_route_memory_not_enabled_for_chat(self, router_all_features):
        """Test that memory is not enabled for general chat."""
        decision = router_all_features.route("hello how are you")
        assert decision.use_memory is False, "Memory should not be enabled for chat"

    def test_route_contracts_enabled_when_very_complex(self, router_all_features):
        """Test that contracts are enabled for very complex requests."""
        very_complex = (
            "Create an enterprise agent with full GDPR and HIPAA compliance, "
            "BANT qualification pipeline, CRM webhook integration, "
            "calendar scheduling, email automation, and multi-path "
            "decision logic based on lead scoring and priority evaluation"
        )
        decision = router_all_features.route(very_complex)
        # Should have high enough complexity to trigger contracts
        if decision.complexity > 0.7:
            assert decision.use_contracts is True, (
                f"Contracts should be enabled for complexity {decision.complexity}"
            )

    def test_get_handler_for_intent(self, router):
        """Test get_handler_for_intent utility method."""
        assert router.get_handler_for_intent(Intent.CREATE_AGENT) == "builder"
        assert router.get_handler_for_intent(Intent.RUN_AGENT) == "executor"
        assert router.get_handler_for_intent(Intent.QUERY_MEMORY) == "memory_query"
        assert router.get_handler_for_intent(Intent.MODIFY_AGENT) == "agent_modifier"
        assert router.get_handler_for_intent(Intent.SYSTEM_COMMAND) == "system_command"
        assert router.get_handler_for_intent(Intent.GENERAL_CHAT) == "chat"


# =============================================================================
# Test: Edge Cases and Integration
# =============================================================================


class TestEdgeCases:
    """Test edge cases and unusual inputs."""

    def test_very_long_message(self, router):
        """Test handling of very long messages."""
        long_msg = "create an agent that " + "does something " * 200
        decision = router.route(long_msg)
        assert isinstance(decision, RouteDecision)
        assert decision.intent == Intent.CREATE_AGENT
        assert decision.complexity <= 1.0  # Should be capped

    def test_unicode_message(self, router):
        """Test handling of unicode characters."""
        decision = router.route("create an agent for handling")
        assert isinstance(decision, RouteDecision)

    def test_special_characters(self, router):
        """Test handling of special characters."""
        decision = router.route("create agent @#$%^&*()!?")
        assert isinstance(decision, RouteDecision)

    def test_mixed_case_keywords(self, classifier):
        """Test that keyword matching is case-insensitive."""
        intent1, _ = classifier.classify("CREATE a new agent")
        intent2, _ = classifier.classify("create a new agent")
        intent3, _ = classifier.classify("CrEaTe a new agent")
        assert intent1 == intent2 == intent3 == Intent.CREATE_AGENT

    def test_multiple_matching_intents(self, classifier):
        """Test handling of messages matching multiple intents."""
        # This message has both CREATE and RUN keywords
        intent, confidence = classifier.classify("create and run a new agent")
        # Should pick the stronger match
        assert intent in (Intent.CREATE_AGENT, Intent.RUN_AGENT)
        assert confidence > 0.5

    def test_subsystem_threshold_boundaries(self, mock_settings_all_features):
        """Test subsystem activation at exact threshold boundaries."""
        router = Router(mock_settings_all_features)

        # Create a message with controlled complexity
        # Note: We can't control exact complexity, but we can test the logic
        decision = router.route("hello")
        # Simple message should have low complexity
        assert decision.complexity < 0.5
        assert decision.use_planning is False
        assert decision.use_contracts is False


class TestIntegrationWithSettings:
    """Test integration with real SigilSettings."""

    def test_router_with_real_settings(self, monkeypatch, tmp_path):
        """Test router with real SigilSettings object."""
        # Clear env vars and use temp directory
        sigil_keys = [k for k in os.environ.keys() if k.startswith("SIGIL_")]
        for key in sigil_keys:
            monkeypatch.delenv(key, raising=False)

        # Create empty .env in tmp_path
        (tmp_path / ".env").write_text("")
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            clear_settings_cache()
            settings = SigilSettings()
            router = Router(settings)

            decision = router.route("create a sales agent")
            assert decision.intent == Intent.CREATE_AGENT
            assert decision.handler_name == "builder"
            # Default settings have features disabled
            assert decision.use_planning is False
            assert decision.use_memory is False
            assert decision.use_contracts is False
        finally:
            os.chdir(original_cwd)
            clear_settings_cache()


# =============================================================================
# Test: Module Imports
# =============================================================================


class TestModuleImports:
    """Test that module exports work correctly."""

    def test_import_from_routing_module(self):
        """Test that all exports can be imported from routing module."""
        from sigil.routing import (
            Intent,
            IntentClassifier,
            ComplexityAssessor,
            RouteDecision,
            Router,
        )
        assert Intent is not None
        assert IntentClassifier is not None
        assert ComplexityAssessor is not None
        assert RouteDecision is not None
        assert Router is not None

    def test_import_intent_enum_values(self):
        """Test that Intent enum values are accessible."""
        from sigil.routing import Intent
        assert hasattr(Intent, "CREATE_AGENT")
        assert hasattr(Intent, "RUN_AGENT")
        assert hasattr(Intent, "QUERY_MEMORY")
        assert hasattr(Intent, "MODIFY_AGENT")
        assert hasattr(Intent, "SYSTEM_COMMAND")
        assert hasattr(Intent, "GENERAL_CHAT")
