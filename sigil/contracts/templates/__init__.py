"""Contract templates for Sigil v2.

This module contains pre-built contract templates for common use cases:

ACTi Strata Templates:
    - lead_qualification_contract: RAI stratum - sales lead qualification
    - research_report_contract: RTI stratum - research report generation
    - appointment_booking_contract: ZACS stratum - calendar scheduling
    - market_analysis_contract: EEI stratum - market analysis
    - compliance_check_contract: IGE stratum - regulatory compliance

Each template provides:
    - Pre-configured deliverables
    - Sensible default constraints
    - Customization options via parameters

Usage:
    >>> from sigil.contracts.templates.acti import lead_qualification_contract
    >>> contract = lead_qualification_contract(max_tokens=10000)
"""

from sigil.contracts.templates.acti import (
    CONTRACT_TEMPLATES,
    appointment_booking_contract,
    compliance_check_contract,
    get_template,
    lead_qualification_contract,
    list_templates,
    market_analysis_contract,
    research_report_contract,
)

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
