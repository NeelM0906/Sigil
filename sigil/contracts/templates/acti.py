"""ACTi contract templates for Sigil v2 framework.

This module provides pre-built contract templates for common ACTi
(Actualized Collective Transformational Intelligence) use cases.

ACTi Strata Templates:
- RTI (Reality & Truth Intelligence): research_report_contract
- RAI (Readiness & Agreement Intelligence): lead_qualification_contract
- ZACS (Zone Action & Conversion Systems): appointment_booking_contract
- EEI (Economic & Ecosystem Intelligence): market_analysis_contract

Each template function returns a fully configured Contract instance
with appropriate deliverables, constraints, and failure strategies.

Example:
    >>> from sigil.contracts.templates.acti import lead_qualification_contract
    >>> contract = lead_qualification_contract()
    >>> print(contract.name)
    'lead_qualification'
"""

from __future__ import annotations

from typing import Optional

from sigil.contracts.schema import (
    Contract,
    ContractConstraints,
    Deliverable,
    FailureStrategy,
)


def lead_qualification_contract(
    max_tokens: int = 5000,
    max_tool_calls: int = 5,
    timeout_seconds: int = 60,
    max_retries: int = 2,
    failure_strategy: FailureStrategy = FailureStrategy.RETRY,
) -> Contract:
    """Create a lead qualification contract for RAI stratum.

    This contract ensures lead qualification produces:
    - A qualification score (0-100)
    - BANT assessment dictionary
    - Recommended action string

    Args:
        max_tokens: Maximum total tokens for execution.
        max_tool_calls: Maximum tool invocations allowed.
        timeout_seconds: Maximum execution time in seconds.
        max_retries: Maximum retry attempts on failure.
        failure_strategy: How to handle verification failures.

    Returns:
        Contract configured for lead qualification.

    Example:
        >>> contract = lead_qualification_contract()
        >>> contract.get_deliverable_names()
        ['score', 'bant_assessment', 'recommended_action']
    """
    return Contract(
        name="lead_qualification",
        description=(
            "Ensures lead qualification produces valid scores, BANT assessments, "
            "and actionable recommendations for sales pipeline management."
        ),
        deliverables=[
            Deliverable(
                name="score",
                type="int",
                description="Lead qualification score from 0 to 100",
                required=True,
                validation_rules=["0 <= value <= 100"],
                example=75,
            ),
            Deliverable(
                name="bant_assessment",
                type="dict",
                description=(
                    "BANT assessment dictionary with keys: budget, authority, need, timeline"
                ),
                required=True,
                validation_rules=[
                    "isinstance(value, dict)",
                    "'budget' in value",
                    "'authority' in value",
                    "'need' in value",
                    "'timeline' in value",
                ],
                example={
                    "budget": "confirmed",
                    "authority": "decision_maker",
                    "need": "high",
                    "timeline": "Q1 2026",
                },
            ),
            Deliverable(
                name="recommended_action",
                type="str",
                description=(
                    "Recommended next action (e.g., 'schedule_demo', 'nurture', 'disqualify')"
                ),
                required=True,
                validation_rules=["len(value) > 0"],
                example="schedule_demo",
            ),
        ],
        constraints=ContractConstraints(
            max_total_tokens=max_tokens,
            max_tool_calls=max_tool_calls,
            timeout_seconds=timeout_seconds,
            warn_threshold=0.8,
        ),
        failure_strategy=failure_strategy,
        max_retries=max_retries,
        version="1.0.0",
        metadata={
            "stratum": "RAI",
            "stratum_name": "Readiness & Agreement Intelligence",
            "use_case": "sales_pipeline",
        },
    )


def research_report_contract(
    max_tokens: int = 8000,
    max_tool_calls: int = 10,
    timeout_seconds: int = 120,
    max_retries: int = 2,
    failure_strategy: FailureStrategy = FailureStrategy.FALLBACK,
) -> Contract:
    """Create a research report contract for RTI stratum.

    This contract ensures research reports contain:
    - A summary string
    - List of sources
    - List of key findings

    Args:
        max_tokens: Maximum total tokens for execution.
        max_tool_calls: Maximum tool invocations allowed.
        timeout_seconds: Maximum execution time in seconds.
        max_retries: Maximum retry attempts on failure.
        failure_strategy: How to handle verification failures.

    Returns:
        Contract configured for research reports.

    Example:
        >>> contract = research_report_contract()
        >>> contract.get_deliverable_names()
        ['summary', 'sources', 'key_findings']
    """
    return Contract(
        name="research_report",
        description=(
            "Ensures research reports contain comprehensive summaries, "
            "verified sources, and actionable key findings."
        ),
        deliverables=[
            Deliverable(
                name="summary",
                type="str",
                description="Executive summary of research findings (100-500 words)",
                required=True,
                validation_rules=["len(value) >= 50"],
                example=(
                    "The target market shows strong growth potential with "
                    "key players focusing on AI integration..."
                ),
            ),
            Deliverable(
                name="sources",
                type="list",
                description="List of source URLs or references used in research",
                required=True,
                validation_rules=["len(value) >= 1"],
                example=[
                    "https://example.com/market-report-2025",
                    "https://example.com/industry-analysis",
                ],
            ),
            Deliverable(
                name="key_findings",
                type="list",
                description="List of key findings from the research",
                required=True,
                validation_rules=["len(value) >= 1"],
                example=[
                    "Market projected to grow 25% YoY",
                    "AI adoption is primary driver",
                    "Top 3 competitors hold 60% market share",
                ],
            ),
        ],
        constraints=ContractConstraints(
            max_total_tokens=max_tokens,
            max_tool_calls=max_tool_calls,
            timeout_seconds=timeout_seconds,
            warn_threshold=0.8,
        ),
        failure_strategy=failure_strategy,
        max_retries=max_retries,
        version="1.0.0",
        metadata={
            "stratum": "RTI",
            "stratum_name": "Reality & Truth Intelligence",
            "use_case": "competitive_intelligence",
        },
    )


def appointment_booking_contract(
    max_tokens: int = 3000,
    max_tool_calls: int = 3,
    timeout_seconds: int = 45,
    max_retries: int = 3,
    failure_strategy: FailureStrategy = FailureStrategy.RETRY,
) -> Contract:
    """Create an appointment booking contract for ZACS stratum.

    This contract ensures appointment bookings produce:
    - Booking confirmation status
    - Datetime string (ISO format)
    - Meeting link

    Args:
        max_tokens: Maximum total tokens for execution.
        max_tool_calls: Maximum tool invocations allowed.
        timeout_seconds: Maximum execution time in seconds.
        max_retries: Maximum retry attempts on failure.
        failure_strategy: How to handle verification failures.

    Returns:
        Contract configured for appointment booking.

    Example:
        >>> contract = appointment_booking_contract()
        >>> contract.get_deliverable_names()
        ['booking_confirmed', 'datetime', 'meeting_link']
    """
    return Contract(
        name="appointment_booking",
        description=(
            "Ensures appointment bookings are confirmed with valid datetime "
            "and meeting link information."
        ),
        deliverables=[
            Deliverable(
                name="booking_confirmed",
                type="bool",
                description="Whether the booking was successfully confirmed",
                required=True,
                validation_rules=["isinstance(value, bool)"],
                example=True,
            ),
            Deliverable(
                name="datetime",
                type="str",
                description="Appointment datetime in ISO format (YYYY-MM-DDTHH:MM:SS)",
                required=True,
                validation_rules=["len(value) >= 10"],
                example="2026-01-15T14:00:00",
            ),
            Deliverable(
                name="meeting_link",
                type="str",
                description="Meeting link URL (Zoom, Google Meet, etc.)",
                required=False,
                validation_rules=["len(value) == 0 or value.startswith('http')"],
                example="https://zoom.us/j/123456789",
            ),
        ],
        constraints=ContractConstraints(
            max_total_tokens=max_tokens,
            max_tool_calls=max_tool_calls,
            timeout_seconds=timeout_seconds,
            warn_threshold=0.8,
        ),
        failure_strategy=failure_strategy,
        max_retries=max_retries,
        version="1.0.0",
        metadata={
            "stratum": "ZACS",
            "stratum_name": "Zone Action & Conversion Systems",
            "use_case": "calendar_scheduling",
        },
    )


def market_analysis_contract(
    max_tokens: int = 10000,
    max_tool_calls: int = 15,
    timeout_seconds: int = 180,
    max_retries: int = 2,
    failure_strategy: FailureStrategy = FailureStrategy.FALLBACK,
) -> Contract:
    """Create a market analysis contract for EEI stratum.

    This contract ensures market analysis produces:
    - Market size string
    - List of competitors
    - List of opportunities

    Args:
        max_tokens: Maximum total tokens for execution.
        max_tool_calls: Maximum tool invocations allowed.
        timeout_seconds: Maximum execution time in seconds.
        max_retries: Maximum retry attempts on failure.
        failure_strategy: How to handle verification failures.

    Returns:
        Contract configured for market analysis.

    Example:
        >>> contract = market_analysis_contract()
        >>> contract.get_deliverable_names()
        ['market_size', 'competitors', 'opportunities']
    """
    return Contract(
        name="market_analysis",
        description=(
            "Ensures market analysis provides comprehensive market sizing, "
            "competitor mapping, and opportunity identification."
        ),
        deliverables=[
            Deliverable(
                name="market_size",
                type="str",
                description=(
                    "Market size estimation with TAM/SAM/SOM breakdown "
                    "(e.g., '$50B TAM, $5B SAM, $500M SOM')"
                ),
                required=True,
                validation_rules=["len(value) >= 10"],
                example="$50B TAM, $5B SAM, $500M SOM by 2028",
            ),
            Deliverable(
                name="competitors",
                type="list",
                description=(
                    "List of competitor dictionaries with name, market_share, and strengths"
                ),
                required=True,
                validation_rules=["len(value) >= 1"],
                example=[
                    {
                        "name": "Competitor A",
                        "market_share": "35%",
                        "strengths": ["Brand recognition", "Enterprise sales"],
                    },
                    {
                        "name": "Competitor B",
                        "market_share": "25%",
                        "strengths": ["Pricing", "SMB focus"],
                    },
                ],
            ),
            Deliverable(
                name="opportunities",
                type="list",
                description="List of market opportunities identified",
                required=True,
                validation_rules=["len(value) >= 1"],
                example=[
                    "Underserved mid-market segment",
                    "AI-native solution gap",
                    "Geographic expansion to APAC",
                ],
            ),
        ],
        constraints=ContractConstraints(
            max_total_tokens=max_tokens,
            max_tool_calls=max_tool_calls,
            timeout_seconds=timeout_seconds,
            warn_threshold=0.8,
        ),
        failure_strategy=failure_strategy,
        max_retries=max_retries,
        version="1.0.0",
        metadata={
            "stratum": "EEI",
            "stratum_name": "Economic & Ecosystem Intelligence",
            "use_case": "strategic_planning",
        },
    )


def compliance_check_contract(
    max_tokens: int = 6000,
    max_tool_calls: int = 8,
    timeout_seconds: int = 90,
    max_retries: int = 1,
    failure_strategy: FailureStrategy = FailureStrategy.FAIL,
) -> Contract:
    """Create a compliance check contract for IGE stratum.

    This contract ensures compliance checks produce:
    - Compliance status boolean
    - List of violations found
    - List of recommendations

    Args:
        max_tokens: Maximum total tokens for execution.
        max_tool_calls: Maximum tool invocations allowed.
        timeout_seconds: Maximum execution time in seconds.
        max_retries: Maximum retry attempts on failure.
        failure_strategy: How to handle verification failures.

    Returns:
        Contract configured for compliance checking.

    Example:
        >>> contract = compliance_check_contract()
        >>> contract.get_deliverable_names()
        ['is_compliant', 'violations', 'recommendations']
    """
    return Contract(
        name="compliance_check",
        description=(
            "Ensures compliance checks provide clear pass/fail status, "
            "detailed violation listing, and remediation recommendations."
        ),
        deliverables=[
            Deliverable(
                name="is_compliant",
                type="bool",
                description="Whether the subject passes compliance check",
                required=True,
                validation_rules=["isinstance(value, bool)"],
                example=False,
            ),
            Deliverable(
                name="violations",
                type="list",
                description="List of compliance violations found (empty if compliant)",
                required=True,
                validation_rules=["isinstance(value, list)"],
                example=[
                    "Missing privacy policy disclosure",
                    "Data retention period exceeds 90 days",
                ],
            ),
            Deliverable(
                name="recommendations",
                type="list",
                description="List of remediation recommendations",
                required=True,
                validation_rules=["isinstance(value, list)"],
                example=[
                    "Add privacy policy link to footer",
                    "Implement automated data purge after 90 days",
                ],
            ),
        ],
        constraints=ContractConstraints(
            max_total_tokens=max_tokens,
            max_tool_calls=max_tool_calls,
            timeout_seconds=timeout_seconds,
            warn_threshold=0.8,
        ),
        failure_strategy=failure_strategy,
        max_retries=max_retries,
        version="1.0.0",
        metadata={
            "stratum": "IGE",
            "stratum_name": "Integrity & Governance Engine",
            "use_case": "regulatory_compliance",
        },
    )


# Template registry for dynamic lookup
CONTRACT_TEMPLATES = {
    "lead_qualification": lead_qualification_contract,
    "research_report": research_report_contract,
    "appointment_booking": appointment_booking_contract,
    "market_analysis": market_analysis_contract,
    "compliance_check": compliance_check_contract,
}


def get_template(name: str, **kwargs) -> Optional[Contract]:
    """Get a contract template by name.

    Args:
        name: Template name (e.g., 'lead_qualification').
        **kwargs: Override arguments for the template function.

    Returns:
        Contract instance if template exists, None otherwise.

    Example:
        >>> contract = get_template('lead_qualification', max_tokens=10000)
        >>> contract.constraints.max_total_tokens
        10000
    """
    template_fn = CONTRACT_TEMPLATES.get(name)
    if template_fn is None:
        return None
    return template_fn(**kwargs)


def list_templates() -> list[str]:
    """List all available template names.

    Returns:
        List of template names.

    Example:
        >>> list_templates()
        ['lead_qualification', 'research_report', 'appointment_booking', ...]
    """
    return list(CONTRACT_TEMPLATES.keys())


__all__ = [
    # Template functions
    "lead_qualification_contract",
    "research_report_contract",
    "appointment_booking_contract",
    "market_analysis_contract",
    "compliance_check_contract",
    # Registry utilities
    "CONTRACT_TEMPLATES",
    "get_template",
    "list_templates",
]
