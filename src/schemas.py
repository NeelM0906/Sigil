"""Pydantic schemas for ACTi Agent Builder.

Defines data models for agent configuration, API requests/responses,
and ACTi methodology classifications.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Stratum(str, Enum):
    """ACTi methodology strata for agent classification.

    Each stratum represents a distinct layer of intelligence in the
    Unblinded Formula framework:

    - RTI: Data gathering, fact verification, reality assessment
    - RAI: Lead qualification, rapport building, readiness evaluation
    - ZACS: Scheduling, conversions, zone-based actions
    - EEI: Analytics, optimization, economic intelligence
    - IGE: Compliance, quality control, governance
    """
    RTI = "RTI"    # Reality & Truth Intelligence
    RAI = "RAI"    # Readiness & Agreement Intelligence
    ZACS = "ZACS"  # Zone Action & Conversion Systems
    EEI = "EEI"    # Economic & Ecosystem Intelligence
    IGE = "IGE"    # Integrity & Governance Engine


# Available MCP tool categories mapped to their capabilities
MCP_TOOL_CATEGORIES = {
    "voice": "Text-to-speech, voice synthesis (ElevenLabs)",
    "websearch": "Web research, fact-finding (Tavily)",
    "calendar": "Scheduling, availability checks (Google Calendar)",
    "communication": "SMS, calls (Twilio)",
    "crm": "Contact management, deals (HubSpot)",
}


class AgentConfig(BaseModel):
    """Configuration for an executable agent with real MCP tool capabilities.

    This is the core output of the builder - a complete agent specification
    that can be instantiated with real tools via MCP protocol.
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
        description="List of MCP tool category names the agent should have access to",
        examples=[["voice", "calendar"], ["websearch", "crm"]],
    )
    model: str = Field(
        default="anthropic:claude-opus-4-5-20251101",
        description="Model identifier for the agent's LLM backbone",
    )
    stratum: Optional[Stratum] = Field(
        default=None,
        description="ACTi stratum classification for the agent's primary function",
    )

    @field_validator("name")
    @classmethod
    def validate_snake_case(cls, v: str) -> str:
        """Ensure name follows snake_case convention."""
        if not re.match(r"^[a-z][a-z0-9_]*$", v):
            raise ValueError(
                "name must be snake_case: lowercase letters, numbers, "
                "and underscores only, starting with a letter"
            )
        return v

    @field_validator("tools")
    @classmethod
    def validate_tools(cls, v: list[str]) -> list[str]:
        """Validate that all tools are recognized MCP tool categories."""
        invalid_tools = [t for t in v if t not in MCP_TOOL_CATEGORIES]
        if invalid_tools:
            valid_tools = ", ".join(MCP_TOOL_CATEGORIES.keys())
            raise ValueError(
                f"Invalid tool(s): {invalid_tools}. "
                f"Valid options: {valid_tools}"
            )
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
                }
            ]
        }
    }


class CreateAgentRequest(BaseModel):
    """Request to create a new agent via natural language description."""
    prompt: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Natural language description of the desired agent",
        examples=[
            "Create an agent that qualifies leads for a B2B SaaS company",
            "I need an agent that can schedule appointments and send reminders",
        ],
    )
    preferred_stratum: Optional[Stratum] = Field(
        default=None,
        description="Optional hint for which ACTi stratum the agent should target",
    )


class RunAgentRequest(BaseModel):
    """Request to execute an agent with a user message."""
    message: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="User message to send to the agent",
    )
    context: Optional[dict] = Field(
        default=None,
        description="Optional context data to pass to the agent (e.g., user info, session state)",
    )


class AgentResponse(BaseModel):
    """Response from an agent execution."""
    message: str = Field(
        ...,
        description="Agent's response message",
    )
    tool_calls: list[dict] = Field(
        default_factory=list,
        description="List of tool calls made during execution",
    )
    metadata: Optional[dict] = Field(
        default=None,
        description="Additional metadata from the execution",
    )


class CreateAgentResponse(BaseModel):
    """Response after creating a new agent."""
    agent_config: AgentConfig = Field(
        ...,
        description="The generated agent configuration",
    )
    file_path: Optional[str] = Field(
        default=None,
        description="Path where the agent config was saved (if persisted)",
    )


class ToolInfo(BaseModel):
    """Information about an available MCP tool."""
    name: str = Field(..., description="Tool category name")
    description: str = Field(..., description="What the tool does")
    capabilities: list[str] = Field(
        default_factory=list,
        description="Specific capabilities provided by this tool",
    )


class ToolListResponse(BaseModel):
    """Response listing all available MCP tools."""
    tools: list[ToolInfo] = Field(
        ...,
        description="List of available tool categories",
    )
