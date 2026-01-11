"""Schema definitions for Sigil v2 framework.

This module contains Pydantic models that define the data structures used
throughout the Sigil framework. These schemas provide:
- Type validation for all data structures
- Serialization/deserialization support
- JSON Schema generation for external tools
- Documentation of expected data formats

Schema Categories:
    - agent: Agent configuration and metadata schemas
    - memory: Memory item and category schemas
    - plan: Planning and task decomposition schemas
    - contract: Contract and deliverable schemas
    - events: Event and message schemas

TODO: Export all schema classes once implemented
"""

__all__ = []  # Will export: AgentConfig, MemoryItem, Plan, Contract, Event, etc.
