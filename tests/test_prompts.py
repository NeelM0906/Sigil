"""Tests for ACTi Agent Builder prompts.

This module tests the system prompts defined in src/prompts.py:
- DEFAULT_MODEL constant
- BUILDER_SYSTEM_PROMPT
- PROMPT_ENGINEER_SYSTEM_PROMPT
- PATTERN_ANALYZER_SYSTEM_PROMPT
"""

from __future__ import annotations

import pytest

from src.prompts import (
    BUILDER_SYSTEM_PROMPT,
    DEFAULT_MODEL,
    PATTERN_ANALYZER_SYSTEM_PROMPT,
    PROMPT_ENGINEER_SYSTEM_PROMPT,
)


# -----------------------------------------------------------------------------
# DEFAULT_MODEL Tests
# -----------------------------------------------------------------------------

class TestDefaultModel:
    """Tests for the DEFAULT_MODEL constant."""

    def test_default_model_is_string(self):
        """Verify DEFAULT_MODEL is a string."""
        assert isinstance(DEFAULT_MODEL, str)

    def test_default_model_is_not_empty(self):
        """Verify DEFAULT_MODEL is not empty."""
        assert len(DEFAULT_MODEL) > 0

    def test_default_model_is_anthropic(self):
        """Verify DEFAULT_MODEL specifies Anthropic provider."""
        assert "anthropic" in DEFAULT_MODEL.lower()

    def test_default_model_is_claude(self):
        """Verify DEFAULT_MODEL specifies Claude model."""
        assert "claude" in DEFAULT_MODEL.lower()

    def test_default_model_has_provider_prefix(self):
        """Verify DEFAULT_MODEL follows provider:model format."""
        assert ":" in DEFAULT_MODEL
        parts = DEFAULT_MODEL.split(":")
        assert len(parts) == 2
        assert parts[0] == "anthropic"


# -----------------------------------------------------------------------------
# BUILDER_SYSTEM_PROMPT Tests
# -----------------------------------------------------------------------------

class TestBuilderSystemPrompt:
    """Tests for the BUILDER_SYSTEM_PROMPT constant."""

    def test_prompt_is_string(self):
        """Verify BUILDER_SYSTEM_PROMPT is a string."""
        assert isinstance(BUILDER_SYSTEM_PROMPT, str)

    def test_prompt_is_not_empty(self):
        """Verify BUILDER_SYSTEM_PROMPT is not empty."""
        assert len(BUILDER_SYSTEM_PROMPT) > 0

    def test_prompt_is_substantial(self):
        """Verify BUILDER_SYSTEM_PROMPT has substantial content."""
        # A good system prompt should be thorough
        assert len(BUILDER_SYSTEM_PROMPT) > 1000

    def test_contains_acti_methodology(self):
        """Verify prompt mentions ACTi methodology."""
        assert "ACTi" in BUILDER_SYSTEM_PROMPT

    def test_contains_five_strata(self):
        """Verify prompt describes all five ACTi strata."""
        assert "RTI" in BUILDER_SYSTEM_PROMPT
        assert "RAI" in BUILDER_SYSTEM_PROMPT
        assert "ZACS" in BUILDER_SYSTEM_PROMPT
        assert "EEI" in BUILDER_SYSTEM_PROMPT
        assert "IGE" in BUILDER_SYSTEM_PROMPT

    def test_contains_stratum_descriptions(self):
        """Verify prompt includes stratum purpose descriptions."""
        # Check for key purpose phrases
        assert "Reality" in BUILDER_SYSTEM_PROMPT or "Truth" in BUILDER_SYSTEM_PROMPT
        assert "Readiness" in BUILDER_SYSTEM_PROMPT or "Agreement" in BUILDER_SYSTEM_PROMPT
        assert "Zone" in BUILDER_SYSTEM_PROMPT or "Conversion" in BUILDER_SYSTEM_PROMPT

    def test_contains_mcp_tools_section(self):
        """Verify prompt describes available MCP tools."""
        assert "MCP" in BUILDER_SYSTEM_PROMPT
        assert "voice" in BUILDER_SYSTEM_PROMPT
        assert "websearch" in BUILDER_SYSTEM_PROMPT
        assert "calendar" in BUILDER_SYSTEM_PROMPT
        assert "communication" in BUILDER_SYSTEM_PROMPT
        assert "crm" in BUILDER_SYSTEM_PROMPT

    def test_contains_tool_providers(self):
        """Verify prompt mentions tool providers."""
        assert "ElevenLabs" in BUILDER_SYSTEM_PROMPT
        assert "Tavily" in BUILDER_SYSTEM_PROMPT
        assert "Google Calendar" in BUILDER_SYSTEM_PROMPT
        assert "Twilio" in BUILDER_SYSTEM_PROMPT
        assert "HubSpot" in BUILDER_SYSTEM_PROMPT

    def test_contains_builder_tools_section(self):
        """Verify prompt describes available builder tools."""
        assert "list_available_tools" in BUILDER_SYSTEM_PROMPT
        assert "create_agent_config" in BUILDER_SYSTEM_PROMPT
        assert "get_agent_config" in BUILDER_SYSTEM_PROMPT
        assert "list_created_agents" in BUILDER_SYSTEM_PROMPT

    def test_contains_agent_creation_process(self):
        """Verify prompt includes agent creation workflow."""
        # Should have numbered steps or workflow guidance
        assert "Step" in BUILDER_SYSTEM_PROMPT or "Process" in BUILDER_SYSTEM_PROMPT

    def test_contains_subagent_references(self):
        """Verify prompt references subagents."""
        assert "prompt-engineer" in BUILDER_SYSTEM_PROMPT
        assert "pattern-analyzer" in BUILDER_SYSTEM_PROMPT

    def test_contains_reference_patterns_section(self):
        """Verify prompt describes reference patterns."""
        assert "Bland" in BUILDER_SYSTEM_PROMPT
        assert "N8N" in BUILDER_SYSTEM_PROMPT or "n8n" in BUILDER_SYSTEM_PROMPT

    def test_contains_system_prompt_guidelines(self):
        """Verify prompt includes guidelines for creating system prompts."""
        # Should have guidance on what makes a good agent prompt
        assert "Role" in BUILDER_SYSTEM_PROMPT
        assert "system_prompt" in BUILDER_SYSTEM_PROMPT.lower()

    def test_contains_best_practices(self):
        """Verify prompt includes best practices section."""
        assert "Best Practice" in BUILDER_SYSTEM_PROMPT or "best practice" in BUILDER_SYSTEM_PROMPT.lower()

    def test_emphasizes_executable_agents(self):
        """Verify prompt emphasizes agents are executable, not just configs."""
        # Should stress that agents DO things
        assert "DO" in BUILDER_SYSTEM_PROMPT or "executable" in BUILDER_SYSTEM_PROMPT.lower()


# -----------------------------------------------------------------------------
# PROMPT_ENGINEER_SYSTEM_PROMPT Tests
# -----------------------------------------------------------------------------

class TestPromptEngineerSystemPrompt:
    """Tests for the PROMPT_ENGINEER_SYSTEM_PROMPT constant."""

    def test_prompt_is_string(self):
        """Verify PROMPT_ENGINEER_SYSTEM_PROMPT is a string."""
        assert isinstance(PROMPT_ENGINEER_SYSTEM_PROMPT, str)

    def test_prompt_is_not_empty(self):
        """Verify PROMPT_ENGINEER_SYSTEM_PROMPT is not empty."""
        assert len(PROMPT_ENGINEER_SYSTEM_PROMPT) > 0

    def test_prompt_is_substantial(self):
        """Verify PROMPT_ENGINEER_SYSTEM_PROMPT has substantial content."""
        assert len(PROMPT_ENGINEER_SYSTEM_PROMPT) > 500

    def test_defines_role_as_prompt_engineer(self):
        """Verify prompt defines role as prompt engineer."""
        assert "prompt engineer" in PROMPT_ENGINEER_SYSTEM_PROMPT.lower()

    def test_specifies_output_format(self):
        """Verify prompt specifies output should be just the prompt."""
        # Should emphasize returning ONLY the prompt text
        assert "ONLY" in PROMPT_ENGINEER_SYSTEM_PROMPT or "only" in PROMPT_ENGINEER_SYSTEM_PROMPT

    def test_contains_prompt_engineering_principles(self):
        """Verify prompt contains engineering principles."""
        assert "Role" in PROMPT_ENGINEER_SYSTEM_PROMPT
        assert "Mission" in PROMPT_ENGINEER_SYSTEM_PROMPT or "Purpose" in PROMPT_ENGINEER_SYSTEM_PROMPT

    def test_describes_stratum_alignment(self):
        """Verify prompt describes stratum-specific guidance."""
        assert "RTI" in PROMPT_ENGINEER_SYSTEM_PROMPT
        assert "RAI" in PROMPT_ENGINEER_SYSTEM_PROMPT
        assert "ZACS" in PROMPT_ENGINEER_SYSTEM_PROMPT

    def test_mentions_tool_usage_instructions(self):
        """Verify prompt mentions tool usage guidance."""
        assert "tool" in PROMPT_ENGINEER_SYSTEM_PROMPT.lower()
        assert "WHEN" in PROMPT_ENGINEER_SYSTEM_PROMPT or "when" in PROMPT_ENGINEER_SYSTEM_PROMPT.lower()
        assert "HOW" in PROMPT_ENGINEER_SYSTEM_PROMPT or "how" in PROMPT_ENGINEER_SYSTEM_PROMPT.lower()

    def test_mentions_behavioral_constraints(self):
        """Verify prompt mentions behavioral constraints."""
        assert "constraint" in PROMPT_ENGINEER_SYSTEM_PROMPT.lower() or "Constraint" in PROMPT_ENGINEER_SYSTEM_PROMPT

    def test_mentions_error_handling(self):
        """Verify prompt mentions error handling guidance."""
        assert "error" in PROMPT_ENGINEER_SYSTEM_PROMPT.lower() or "Error" in PROMPT_ENGINEER_SYSTEM_PROMPT

    def test_quality_standards_mentioned(self):
        """Verify prompt mentions quality standards."""
        assert "Quality" in PROMPT_ENGINEER_SYSTEM_PROMPT or "quality" in PROMPT_ENGINEER_SYSTEM_PROMPT.lower()


# -----------------------------------------------------------------------------
# PATTERN_ANALYZER_SYSTEM_PROMPT Tests
# -----------------------------------------------------------------------------

class TestPatternAnalyzerSystemPrompt:
    """Tests for the PATTERN_ANALYZER_SYSTEM_PROMPT constant."""

    def test_prompt_is_string(self):
        """Verify PATTERN_ANALYZER_SYSTEM_PROMPT is a string."""
        assert isinstance(PATTERN_ANALYZER_SYSTEM_PROMPT, str)

    def test_prompt_is_not_empty(self):
        """Verify PATTERN_ANALYZER_SYSTEM_PROMPT is not empty."""
        assert len(PATTERN_ANALYZER_SYSTEM_PROMPT) > 0

    def test_prompt_is_substantial(self):
        """Verify PATTERN_ANALYZER_SYSTEM_PROMPT has substantial content."""
        assert len(PATTERN_ANALYZER_SYSTEM_PROMPT) > 500

    def test_defines_role_as_pattern_analyzer(self):
        """Verify prompt defines role as pattern recognition specialist."""
        assert "pattern" in PATTERN_ANALYZER_SYSTEM_PROMPT.lower()

    def test_mentions_bland_ai(self):
        """Verify prompt mentions Bland AI configurations."""
        assert "Bland" in PATTERN_ANALYZER_SYSTEM_PROMPT

    def test_mentions_n8n(self):
        """Verify prompt mentions N8N workflows."""
        assert "N8N" in PATTERN_ANALYZER_SYSTEM_PROMPT or "n8n" in PATTERN_ANALYZER_SYSTEM_PROMPT

    def test_contains_analysis_framework(self):
        """Verify prompt contains analysis framework sections."""
        assert "Purpose" in PATTERN_ANALYZER_SYSTEM_PROMPT
        assert "Stratum" in PATTERN_ANALYZER_SYSTEM_PROMPT

    def test_mentions_prompt_patterns(self):
        """Verify prompt mentions extracting prompt patterns."""
        assert "Prompt" in PATTERN_ANALYZER_SYSTEM_PROMPT

    def test_mentions_variable_extraction(self):
        """Verify prompt mentions variable extraction."""
        assert "Variable" in PATTERN_ANALYZER_SYSTEM_PROMPT or "variable" in PATTERN_ANALYZER_SYSTEM_PROMPT.lower()

    def test_mentions_tool_node_usage(self):
        """Verify prompt mentions analyzing tool/node usage."""
        assert "Tool" in PATTERN_ANALYZER_SYSTEM_PROMPT or "Node" in PATTERN_ANALYZER_SYSTEM_PROMPT

    def test_mentions_transferable_patterns(self):
        """Verify prompt mentions extracting transferable patterns."""
        assert "Transferable" in PATTERN_ANALYZER_SYSTEM_PROMPT or "reusable" in PATTERN_ANALYZER_SYSTEM_PROMPT.lower()

    def test_mentions_acti_framework(self):
        """Verify prompt references ACTi framework."""
        assert "ACTi" in PATTERN_ANALYZER_SYSTEM_PROMPT

    def test_mentions_mcp_tool_mapping(self):
        """Verify prompt mentions MCP tool mapping."""
        assert "MCP" in PATTERN_ANALYZER_SYSTEM_PROMPT

    def test_output_format_specified(self):
        """Verify prompt specifies output format."""
        assert "Output" in PATTERN_ANALYZER_SYSTEM_PROMPT
        # Should have structured format guidance
        assert "##" in PATTERN_ANALYZER_SYSTEM_PROMPT or "Format" in PATTERN_ANALYZER_SYSTEM_PROMPT


# -----------------------------------------------------------------------------
# Module Exports Tests
# -----------------------------------------------------------------------------

class TestModuleExports:
    """Tests for module-level exports."""

    def test_all_prompts_exported(self):
        """Verify all expected prompts are importable."""
        from src.prompts import __all__

        assert "DEFAULT_MODEL" in __all__
        assert "BUILDER_SYSTEM_PROMPT" in __all__
        assert "PROMPT_ENGINEER_SYSTEM_PROMPT" in __all__
        assert "PATTERN_ANALYZER_SYSTEM_PROMPT" in __all__

    def test_exports_match_defined_constants(self):
        """Verify exported names match defined constants."""
        from src import prompts

        # All exported names should be defined
        for name in prompts.__all__:
            assert hasattr(prompts, name)


# -----------------------------------------------------------------------------
# Prompt Consistency Tests
# -----------------------------------------------------------------------------

class TestPromptConsistency:
    """Tests for consistency across all prompts."""

    def test_all_prompts_mention_acti(self):
        """Verify ACTi methodology is consistent across prompts."""
        # All prompts should reference the ACTi framework
        assert "ACTi" in BUILDER_SYSTEM_PROMPT
        # Subagent prompts should also be aware of ACTi
        assert "ACTi" in PATTERN_ANALYZER_SYSTEM_PROMPT

    def test_stratum_terminology_consistent(self):
        """Verify stratum terminology is consistent."""
        # All should use the same five strata names
        strata = ["RTI", "RAI", "ZACS", "EEI", "IGE"]

        for stratum in strata:
            assert stratum in BUILDER_SYSTEM_PROMPT

        # Pattern analyzer should know about strata too
        for stratum in strata:
            assert stratum in PATTERN_ANALYZER_SYSTEM_PROMPT

    def test_no_hardcoded_file_paths(self):
        """Verify prompts don't have hardcoded absolute paths."""
        # Should use relative paths or variables
        prompts = [
            BUILDER_SYSTEM_PROMPT,
            PROMPT_ENGINEER_SYSTEM_PROMPT,
            PATTERN_ANALYZER_SYSTEM_PROMPT,
        ]

        for prompt in prompts:
            # Should not have hardcoded user paths
            assert "/Users/" not in prompt
            assert "/home/" not in prompt
            assert "C:\\" not in prompt

    def test_prompts_are_professional(self):
        """Verify prompts maintain professional tone (no emojis)."""
        prompts = [
            BUILDER_SYSTEM_PROMPT,
            PROMPT_ENGINEER_SYSTEM_PROMPT,
            PATTERN_ANALYZER_SYSTEM_PROMPT,
        ]

        # Common emoji unicode ranges (basic check)
        emoji_indicators = [
            "\U0001F600",  # Grinning face start range
            "\U0001F44D",  # Thumbs up
            "\u2705",      # Check mark
            "\u274C",      # X mark
        ]

        for prompt in prompts:
            for emoji in emoji_indicators:
                assert emoji not in prompt, f"Found emoji in prompt: {emoji}"
