"""ACTi-specific category templates for Sigil v2 memory system.

This module defines predefined category templates tailored for the ACTi
(Actualized Collective Transformational Intelligence) methodology, which
focuses on sales, lead qualification, and customer engagement.

ACTi Strata:
    - RTI: Reality & Truth Intelligence
    - RAI: Readiness & Agreement Intelligence
    - ZACS: Zone Action & Conversion Systems
    - EEI: Economic & Ecosystem Intelligence
    - IGE: Integrity & Governance Engine

Each category template includes:
    - description: What the category contains
    - template: Initial markdown structure
"""

from typing import Dict, Any

# =============================================================================
# ACTi Category Templates
# =============================================================================

ACTI_CATEGORY_TEMPLATES: Dict[str, Dict[str, str]] = {
    # =========================================================================
    # Lead & Customer Intelligence
    # =========================================================================
    "lead_preferences": {
        "description": "Customer preferences, communication styles, and requirements",
        "template": """# Lead Preferences

## Communication Preferences
<!-- How leads prefer to be contacted -->
- Preferred contact method
- Best times to reach
- Communication style preferences

## Product/Service Preferences
<!-- What features and capabilities matter most -->
- Priority features
- Must-have requirements
- Nice-to-have features

## Budget & Timeline
<!-- Financial and timing constraints -->
- Budget range
- Decision timeline
- Urgency indicators

## Decision-Making Style
<!-- How leads make decisions -->
- Key decision factors
- Stakeholder involvement
- Evaluation criteria

## Additional Insights
<!-- Other relevant preferences -->
""",
    },
    "objection_patterns": {
        "description": "Common objections encountered and effective response strategies",
        "template": """# Objection Patterns

## Pricing Objections
<!-- "Too expensive", "Can't afford it", "Need discount" -->
### Common Patterns
### Effective Responses

## Timing Objections
<!-- "Not the right time", "Need to think about it" -->
### Common Patterns
### Effective Responses

## Authority Objections
<!-- "Need to check with...", "Not my decision" -->
### Common Patterns
### Effective Responses

## Competitor Comparisons
<!-- "X offers...", "Why not use Y?" -->
### Common Patterns
### Effective Responses

## Need/Fit Objections
<!-- "Not sure we need this", "Doesn't fit our process" -->
### Common Patterns
### Effective Responses

## Trust/Credibility Objections
<!-- "Never heard of you", "Need references" -->
### Common Patterns
### Effective Responses
""",
    },
    "buying_signals": {
        "description": "Indicators of purchase intent and engagement",
        "template": """# Buying Signals

## Strong Signals
<!-- High likelihood indicators -->
- Asking about pricing/terms
- Requesting proposals/contracts
- Discussing implementation timeline
- Introducing other stakeholders

## Moderate Signals
<!-- Medium likelihood indicators -->
- Asking detailed technical questions
- Comparing to current solution
- Requesting demos/trials
- Following up proactively

## Early Signals
<!-- Beginning of interest -->
- Engaging with content
- Responding to outreach
- Asking general questions

## Disengagement Signals
<!-- Warning signs -->
- Delayed responses
- Vague commitments
- Avoiding specific discussions
""",
    },
    # =========================================================================
    # Conversation & Approach Intelligence
    # =========================================================================
    "conversation_insights": {
        "description": "Patterns and learnings from customer conversations",
        "template": """# Conversation Insights

## Discovery Patterns
<!-- What works in discovery calls -->
- Effective opening questions
- Information gathering techniques
- Building rapport approaches

## Pain Point Patterns
<!-- Common challenges customers face -->
- Operational pain points
- Financial pain points
- Strategic pain points

## Decision Factors
<!-- What influences buying decisions -->
- Primary decision drivers
- Secondary considerations
- Deal breakers

## Communication Effectiveness
<!-- What resonates vs. what doesn't -->
- Messaging that works
- Approaches to avoid
- Tone and style notes
""",
    },
    "successful_approaches": {
        "description": "Strategies and techniques that have proven effective",
        "template": """# Successful Approaches

## Discovery Techniques
<!-- Effective ways to understand needs -->
- Question frameworks
- Active listening techniques
- Needs assessment methods

## Presentation Strategies
<!-- What resonates in demos/presentations -->
- Value proposition framing
- Feature prioritization
- ROI communication

## Objection Handling
<!-- Approaches that overcome resistance -->
- Reframing techniques
- Evidence-based responses
- Empathy-first approaches

## Closing Techniques
<!-- What helps close deals -->
- Creating urgency (ethically)
- Addressing final concerns
- Next step clarity

## Relationship Building
<!-- Trust and rapport methods -->
- Personal connection points
- Follow-up best practices
- Value-add touchpoints
""",
    },
    # =========================================================================
    # Product & Market Intelligence
    # =========================================================================
    "product_knowledge": {
        "description": "Product features, capabilities, and technical details",
        "template": """# Product Knowledge

## Core Features
<!-- Primary capabilities -->
- Feature descriptions
- Use cases
- Benefits

## Technical Specifications
<!-- Technical details -->
- Requirements
- Integrations
- Performance metrics

## Differentiators
<!-- Competitive advantages -->
- Unique features
- Superior capabilities
- Innovation highlights

## Pricing & Packaging
<!-- Commercial information -->
- Pricing tiers
- Package contents
- Add-ons/upgrades

## Implementation
<!-- Getting started -->
- Setup process
- Timeline expectations
- Support resources

## Common Questions
<!-- FAQ-style knowledge -->
""",
    },
    "competitor_intelligence": {
        "description": "Information about competitors and market positioning",
        "template": """# Competitor Intelligence

## Direct Competitors
<!-- Primary competitors -->
### Competitor A
- Strengths
- Weaknesses
- Positioning against

### Competitor B
- Strengths
- Weaknesses
- Positioning against

## Indirect Competitors
<!-- Alternative solutions -->
- DIY/manual approaches
- Adjacent solutions
- Legacy systems

## Competitive Positioning
<!-- How we differentiate -->
- Key differentiators
- Battle cards
- Win themes

## Market Trends
<!-- Industry movements -->
- Emerging competitors
- Market shifts
- Customer expectations
""",
    },
    "market_insights": {
        "description": "Industry trends, market dynamics, and economic factors",
        "template": """# Market Insights

## Industry Trends
<!-- Current market movements -->
- Technology trends
- Regulatory changes
- Buyer behavior shifts

## Economic Factors
<!-- Financial considerations -->
- Budget climate
- Spending patterns
- ROI expectations

## Customer Segments
<!-- Target market insights -->
- Segment characteristics
- Buying patterns
- Success indicators

## Opportunity Areas
<!-- Growth potential -->
- Underserved needs
- Emerging use cases
- Expansion opportunities
""",
    },
    # =========================================================================
    # Process & Operations Intelligence
    # =========================================================================
    "qualification_criteria": {
        "description": "BANT and other qualification frameworks",
        "template": """# Qualification Criteria

## BANT Framework
### Budget
- Budget indicators
- Qualification questions
- Red flags

### Authority
- Decision-maker identification
- Influence mapping
- Qualification questions

### Need
- Pain point indicators
- Use case validation
- Qualification questions

### Timeline
- Urgency indicators
- Project timelines
- Qualification questions

## Scoring Model
<!-- How to score leads -->
- Score ranges
- Weighting factors
- Action thresholds

## Disqualification Criteria
<!-- When to disqualify -->
- Hard disqualifiers
- Soft disqualifiers
- Referral opportunities
""",
    },
    "follow_up_patterns": {
        "description": "Effective follow-up sequences and timing",
        "template": """# Follow-up Patterns

## Initial Outreach
<!-- First contact sequences -->
- Timing recommendations
- Channel preferences
- Message templates

## Post-Meeting Follow-up
<!-- After conversations -->
- Same-day actions
- Next-day actions
- Week-after actions

## Nurture Sequences
<!-- Long-term engagement -->
- Content cadence
- Value-add touchpoints
- Re-engagement triggers

## Re-engagement
<!-- Reviving cold leads -->
- Trigger events
- Re-approach strategies
- Win-back messaging
""",
    },
    # =========================================================================
    # Feedback & Learning
    # =========================================================================
    "customer_feedback": {
        "description": "Customer feedback, testimonials, and satisfaction data",
        "template": """# Customer Feedback

## Positive Feedback
<!-- What customers love -->
- Product praise
- Service compliments
- Success stories

## Improvement Areas
<!-- Constructive feedback -->
- Feature requests
- Pain points
- Suggestions

## Testimonials
<!-- Quotable feedback -->
- Customer quotes
- Use case successes
- Recommendation statements

## Satisfaction Metrics
<!-- Quantitative feedback -->
- NPS indicators
- Satisfaction trends
- Retention signals
""",
    },
    "lessons_learned": {
        "description": "Post-mortem insights from won and lost deals",
        "template": """# Lessons Learned

## Won Deals
### Success Factors
- What worked
- Key moments
- Replicable patterns

### Customer Profiles
- Ideal customer traits
- Decision journey
- Champions identified

## Lost Deals
### Loss Reasons
- Common objections
- Competitor wins
- Internal factors

### Prevention Strategies
- Early warning signs
- Intervention points
- Process improvements

## No-Decision Outcomes
<!-- Deals that stalled -->
- Stall patterns
- Revival attempts
- Prevention strategies
""",
    },
}


# =============================================================================
# Helper Functions
# =============================================================================


def get_acti_template(category_name: str) -> dict[str, str] | None:
    """Get a specific ACTi category template.

    Args:
        category_name: Name of the category.

    Returns:
        Template dict or None if not found.
    """
    normalized = category_name.strip().lower().replace(" ", "_").replace("-", "_")
    return ACTI_CATEGORY_TEMPLATES.get(normalized)


def list_acti_categories() -> list[str]:
    """List all available ACTi category names.

    Returns:
        List of category names.
    """
    return sorted(ACTI_CATEGORY_TEMPLATES.keys())


def get_acti_category_descriptions() -> dict[str, str]:
    """Get descriptions for all ACTi categories.

    Returns:
        Dict mapping category names to descriptions.
    """
    return {
        name: template["description"]
        for name, template in ACTI_CATEGORY_TEMPLATES.items()
    }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ACTI_CATEGORY_TEMPLATES",
    "get_acti_template",
    "list_acti_categories",
    "get_acti_category_descriptions",
]
