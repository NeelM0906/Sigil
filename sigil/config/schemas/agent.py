"""Agent configuration and metadata schemas for Sigil v2.

This module defines Pydantic models for agent configuration and runtime
metadata. These schemas are migrated from v1 (src/schemas.py) with v2
extensions for memory, planning, and contracts.

Classes:
    Stratum: ACTi methodology strata enum
    AgentMetadata: Runtime metadata for agents
    AgentConfig: Configuration for agent initialization

Constants:
    TOOL_CATEGORIES: Available tool categories
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from sigil.config.settings import DEFAULT_MODEL


class Stratum(str, Enum):
    """ACTi methodology strata for agent classification.

    Each stratum represents a distinct layer of intelligence in the
    Unblinded Formula framework:

    - RTI: Data gathering, fact verification, reality assessment
    - RAI: Lead qualification, rapport building, readiness evaluation
    - ZACS: Scheduling, conversions, zone-based actions
    - EEI: Analytics, optimization, economic intelligence
    - IGE: Compliance, quality control, governance

    Attributes:
        RTI: Reality & Truth Intelligence
        RAI: Readiness & Agreement Intelligence
        ZACS: Zone Action & Conversion Systems
        EEI: Economic & Ecosystem Intelligence
        IGE: Integrity & Governance Engine
    """

    RTI = "RTI"
    RAI = "RAI"
    ZACS = "ZACS"
    EEI = "EEI"
    IGE = "IGE"


# Available tool categories mapped to their capabilities
# Currently only websearch is supported via Tavily
TOOL_CATEGORIES: dict[str, str] = {
    "websearch": "Web research, fact-finding (Tavily)",
    "memory": "Memory storage and retrieval (builtin)",
    "planning": "Task decomposition and planning (builtin)",
}

# Backward compatibility alias
MCP_TOOL_CATEGORIES = TOOL_CATEGORIES


class AgentMetadata(BaseModel):
    """Metadata for tracking agent lifecycle and usage.

    This metadata is automatically managed by the persistence layer
    and provides tracking for creation, updates, execution, and versioning.

    Attributes:
        created_at: ISO timestamp when the agent was created
        updated_at: ISO timestamp of the last update
        last_executed: ISO timestamp of the last execution
        version: Version number, incremented on each update
        execution_count: Total number of times this agent has been executed
        tags: Tags for categorization and filtering

    Example:
        ```python
        metadata = AgentMetadata(
            version=2,
            tags=["sales", "qualification"]
        )
        print(metadata.created_at)  # Auto-generated timestamp
        ```
    """

    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp when the agent was created",
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp of the last update",
    )
    last_executed: Optional[str] = Field(
        default=None,
        description="ISO timestamp of the last execution",
    )
    version: int = Field(
        default=1,
        ge=1,
        description="Version number, incremented on each update",
    )
    execution_count: int = Field(
        default=0,
        ge=0,
        description="Total number of times this agent has been executed",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorization and filtering",
    )

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: list[str]) -> list[str]:
        """Validate tags are non-empty strings and normalize them.

        Args:
            v: List of tag strings to validate.

        Returns:
            List of validated, deduplicated, normalized tags.
        """
        validated = []
        for tag in v:
            tag = tag.strip().lower()
            if tag and len(tag) <= 50:
                validated.append(tag)
        return list(set(validated))  # Remove duplicates


class AgentConfig(BaseModel):
    """Configuration for an executable agent with tool capabilities.

    This is the core output of the builder - a complete agent specification
    that can be instantiated with available tools. This schema is migrated
    from v1 with additional v2 fields for memory, planning, and contracts.

    Attributes:
        name: Unique identifier for the agent (snake_case)
        description: What the agent does - clear, concise purpose statement
        system_prompt: Complete system prompt defining agent behavior
        tools: List of tool category names the agent should have access to
        model: Model identifier for the agent's LLM backbone
        stratum: ACTi stratum classification for the agent's primary function
        metadata: Agent lifecycle metadata (auto-populated on creation/update)
        memory_enabled: Whether the agent uses the 3-layer memory system
        planning_enabled: Whether the agent uses task decomposition
        contract_name: Name of the contract to enforce on outputs

    Example:
        ```python
        config = AgentConfig(
            name="lead_qualifier",
            description="Qualifies inbound leads by assessing BANT criteria",
            system_prompt="You are a professional lead qualification specialist...",
            tools=["websearch", "memory"],
            stratum=Stratum.RAI,
            memory_enabled=True
        )
        ```
    """

    name: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Unique identifier for the agent (snake_case)",
        examples=["lead_qualifier", "appointment_scheduler"],
    )
    description: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="What the agent does - clear, concise purpose statement",
        examples=["Qualifies inbound leads by assessing budget and timeline"],
    )
    system_prompt: str = Field(
        ...,
        min_length=50,
        description="Complete system prompt defining agent behavior, personality, and capabilities",
    )
    tools: list[str] = Field(
        default_factory=list,
        description="List of tool category names the agent should have access to",
        examples=[["websearch", "memory"], ["websearch", "planning"]],
    )
    model: str = Field(
        default=DEFAULT_MODEL,
        description="Model identifier for the agent's LLM backbone",
    )
    stratum: Optional[Stratum] = Field(
        default=None,
        description="ACTi stratum classification for the agent's primary function",
    )
    metadata: Optional[AgentMetadata] = Field(
        default=None,
        description="Agent lifecycle metadata (auto-populated on creation/update)",
    )
    # v2 fields
    memory_enabled: bool = Field(
        default=False,
        description="Whether the agent uses the 3-layer memory system",
    )
    planning_enabled: bool = Field(
        default=False,
        description="Whether the agent uses task decomposition and planning",
    )
    contract_name: Optional[str] = Field(
        default=None,
        description="Name of the contract to enforce on agent outputs",
    )

    @field_validator("name")
    @classmethod
    def validate_snake_case(cls, v: str) -> str:
        """Ensure name follows snake_case convention.

        Args:
            v: Name to validate.

        Returns:
            Validated name.

        Raises:
            ValueError: If name is not valid snake_case.
        """
        if not re.match(r"^[a-z][a-z0-9_]*$", v):
            raise ValueError(
                "name must be snake_case: lowercase letters, numbers, "
                "and underscores only, starting with a letter"
            )
        return v

    @field_validator("tools")
    @classmethod
    def validate_tools(cls, v: list[str]) -> list[str]:
        """Validate that all tools are recognized tool categories.

        Args:
            v: List of tool names to validate.

        Returns:
            Validated list of tool names.

        Raises:
            ValueError: If any tool name is not recognized.
        """
        invalid_tools = [t for t in v if t not in TOOL_CATEGORIES]
        if invalid_tools:
            valid_tools = ", ".join(TOOL_CATEGORIES.keys())
            raise ValueError(
                f"Invalid tool(s): {invalid_tools}. "
                f"Valid options: {valid_tools}"
            )
        return v

    @field_validator("model")
    @classmethod
    def validate_model_format(cls, v: str) -> str:
        """Validate model follows provider:model-name pattern.

        Args:
            v: Model identifier to validate.

        Returns:
            Validated model identifier.

        Raises:
            ValueError: If model format is invalid.
        """
        if ":" not in v:
            raise ValueError(
                "Model must follow 'provider:model-name' format "
                "(e.g., 'anthropic:claude-opus-4-5-20251101')"
            )
        provider, model_name = v.split(":", 1)
        if not provider or not model_name:
            raise ValueError("Model must have non-empty provider and model name")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "lead_qualifier",
                    "description": "Qualifies inbound leads by assessing budget, authority, need, and timeline",
                    "system_prompt": "You are a professional lead qualification specialist...",
                    "tools": ["communication", "crm"],
                    "model": "anthropic:claude-opus-4-5-20251101",
                    "stratum": "RAI",
                    "memory_enabled": True,
                    "planning_enabled": False,
                    "contract_name": "lead_qualification",
                }
            ]
        }
    }


__all__ = [
    "Stratum",
    "TOOL_CATEGORIES",
    "MCP_TOOL_CATEGORIES",  # Backward compatibility alias
    "AgentMetadata",
    "AgentConfig",
]
