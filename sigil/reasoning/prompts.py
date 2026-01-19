"""Prompt templates for reasoning strategies in Sigil v2.

This module provides prompt templates and builders for each reasoning strategy.
Templates are configurable and support dynamic context injection.

Classes:
    PromptTemplate: Base class for prompt templates.
    DirectPrompts: Prompts for direct strategy.
    ChainOfThoughtPrompts: Prompts for CoT strategy.
    TreeOfThoughtsPrompts: Prompts for ToT strategy.
    ReActPrompts: Prompts for ReAct strategy.
    MCTSPrompts: Prompts for MCTS strategy.

Example:
    >>> from sigil.reasoning.prompts import ChainOfThoughtPrompts
    >>>
    >>> prompts = ChainOfThoughtPrompts()
    >>> prompt = prompts.build_reasoning_prompt(
    ...     task="Calculate 15% tip on $80",
    ...     context={"format": "step by step"},
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


# =============================================================================
# Base Template
# =============================================================================


@dataclass
class PromptTemplate:
    """Base class for prompt templates.

    Attributes:
        template: The prompt template string with {placeholders}.
        description: Description of what this template does.
    """

    template: str
    description: str = ""

    def format(self, **kwargs: Any) -> str:
        """Format the template with provided values.

        Args:
            **kwargs: Values to substitute into template.

        Returns:
            Formatted prompt string.
        """
        return self.template.format(**kwargs)


# =============================================================================
# Direct Strategy Prompts
# =============================================================================


class DirectPrompts:
    """Prompt templates for DirectStrategy.

    Provides simple, direct prompts for low-complexity tasks.
    """

    SYSTEM = PromptTemplate(
        template="""You are a helpful AI assistant. Provide clear, direct answers to questions.""",
        description="System prompt for direct reasoning",
    )

    TASK = PromptTemplate(
        template="""Answer the following task directly and concisely.
{context_section}
Task: {task}

Provide a clear, direct answer:""",
        description="Main task prompt for direct reasoning",
    )

    TASK_WITH_FORMAT = PromptTemplate(
        template="""Answer the following task directly and concisely.
{context_section}
Task: {task}

Format: {format}

Answer:""",
        description="Task prompt with output format specification",
    )

    @classmethod
    def build_task_prompt(
        cls,
        task: str,
        context: Optional[dict[str, Any]] = None,
        output_format: Optional[str] = None,
    ) -> str:
        """Build the task prompt.

        Args:
            task: The task to execute.
            context: Optional context dictionary.
            output_format: Optional output format specification.

        Returns:
            Formatted prompt string.
        """
        context_section = ""
        if context:
            context_section = "\nContext:\n"
            for key, value in context.items():
                context_section += f"- {key}: {value}\n"

        if output_format:
            return cls.TASK_WITH_FORMAT.format(
                context_section=context_section,
                task=task,
                format=output_format,
            )
        else:
            return cls.TASK.format(
                context_section=context_section,
                task=task,
            )


# =============================================================================
# Chain of Thought Prompts
# =============================================================================


class ChainOfThoughtPrompts:
    """Prompt templates for ChainOfThoughtStrategy.

    Provides step-by-step reasoning prompts.
    """

    SYSTEM = PromptTemplate(
        template="""You are a careful analytical thinker who reasons through problems step by step.
Always show your work and explain your reasoning clearly.""",
        description="System prompt for chain of thought reasoning",
    )

    REASONING = PromptTemplate(
        template="""You are a careful analytical thinker. Solve the following task by thinking through it step by step.
{context_section}
Task: {task}

Let's think step by step:

1.""",
        description="Main reasoning prompt with step-by-step instruction",
    )

    REASONING_VERBOSE = PromptTemplate(
        template="""You are a careful analytical thinker. Solve the following task by thinking through it step by step.
{context_section}
Task: {task}

Instructions:
- Break down the problem into clear steps
- Show your reasoning for each step
- Verify your work before concluding
- Provide a clear final answer

Let's think step by step:

Step 1:""",
        description="Verbose reasoning prompt with detailed instructions",
    )

    SUMMARIZE = PromptTemplate(
        template="""Given the following reasoning steps:
{steps}

Provide a clear, concise summary of the final answer:""",
        description="Summarization prompt for reasoning output",
    )

    @classmethod
    def build_reasoning_prompt(
        cls,
        task: str,
        context: Optional[dict[str, Any]] = None,
        verbose: bool = False,
    ) -> str:
        """Build the reasoning prompt.

        Args:
            task: The task to reason about.
            context: Optional context dictionary.
            verbose: Whether to use verbose prompt.

        Returns:
            Formatted prompt string.
        """
        context_section = ""
        if context:
            context_section = "\nContext:\n"
            for key, value in context.items():
                context_section += f"- {key}: {value}\n"

        template = cls.REASONING_VERBOSE if verbose else cls.REASONING
        return template.format(
            context_section=context_section,
            task=task,
        )


# =============================================================================
# Tree of Thoughts Prompts
# =============================================================================


class TreeOfThoughtsPrompts:
    """Prompt templates for TreeOfThoughtsStrategy.

    Provides multi-path exploration prompts.
    """

    SYSTEM = PromptTemplate(
        template="""You are a strategic thinker who explores multiple approaches to problems.
Generate diverse solutions and evaluate them objectively.""",
        description="System prompt for tree of thoughts reasoning",
    )

    GENERATE_APPROACHES = PromptTemplate(
        template="""You are a strategic thinker. Generate {num_approaches} different approaches to solve the following task.

Each approach should be distinct and explore a different angle or method.
{context_section}
Task: {task}

Generate exactly {num_approaches} approaches in the following format:

Approach 1: [Title]
Description: [Brief description of this approach]

Approach 2: [Title]
Description: [Brief description of this approach]

... and so on.

Generate {num_approaches} distinct approaches:""",
        description="Prompt for generating multiple approaches",
    )

    EVALUATE_APPROACHES = PromptTemplate(
        template="""Evaluate the following approaches for solving this task:

Task: {task}

Approaches:{approaches_list}

For each approach, provide a score from 0 to 10 based on:
- Feasibility: How practical is this approach?
- Effectiveness: How likely is it to achieve the goal?
- Efficiency: How resource-efficient is it?

Format your response as:
Approach 1: [score]/10 - [brief justification]
Approach 2: [score]/10 - [brief justification]
... and so on.

Then state which approach is best and why.

Evaluation:""",
        description="Prompt for evaluating approaches",
    )

    REFINE_APPROACH = PromptTemplate(
        template="""You selected the following approach as the best for the task:

Task: {task}

Selected Approach: {approach}
{context_section}
Now, elaborate on this approach in detail. Provide:
1. A step-by-step implementation plan
2. Key considerations and potential challenges
3. Success criteria

Detailed solution:""",
        description="Prompt for refining the best approach",
    )

    @classmethod
    def build_generation_prompt(
        cls,
        task: str,
        num_approaches: int,
        context: Optional[dict[str, Any]] = None,
    ) -> str:
        """Build the approach generation prompt.

        Args:
            task: The task to generate approaches for.
            num_approaches: Number of approaches to generate.
            context: Optional context dictionary.

        Returns:
            Formatted prompt string.
        """
        context_section = ""
        if context:
            context_section = "\nContext:\n"
            for key, value in context.items():
                context_section += f"- {key}: {value}\n"

        return cls.GENERATE_APPROACHES.format(
            context_section=context_section,
            task=task,
            num_approaches=num_approaches,
        )

    @classmethod
    def build_evaluation_prompt(
        cls,
        task: str,
        approaches: list[str],
    ) -> str:
        """Build the evaluation prompt.

        Args:
            task: The original task.
            approaches: List of approach descriptions.

        Returns:
            Formatted prompt string.
        """
        approaches_list = ""
        for i, approach in enumerate(approaches, 1):
            approaches_list += f"\nApproach {i}: {approach}\n"

        return cls.EVALUATE_APPROACHES.format(
            task=task,
            approaches_list=approaches_list,
        )


# =============================================================================
# ReAct Prompts
# =============================================================================


class ReActPrompts:
    """Prompt templates for ReActStrategy.

    Provides Thought-Action-Observation loop prompts.
    """

    SYSTEM = PromptTemplate(
        template="""You are a problem-solving agent using the ReAct framework.
You interleave thinking with actions to solve problems step by step.
Always think before you act, and learn from observations.""",
        description="System prompt for ReAct reasoning",
    )

    INITIAL = PromptTemplate(
        template="""You are a problem-solving agent using the ReAct framework.
You have access to the following tools: {tools}

Solve the task by interleaving Thought, Action, and Observation steps.
{context_section}
Task: {task}

Instructions:
1. First, think about what you need to do (Thought)
2. Then, take an action using a tool (Action: tool_name[args])
3. You will receive an observation (Observation: result)
4. Continue until you can provide a final answer

When you have the final answer, respond with:
Final Answer: [your answer]

Begin:

Thought 1:""",
        description="Initial ReAct prompt",
    )

    CONTINUATION = PromptTemplate(
        template="""{history}

Thought {iteration}:""",
        description="Continuation prompt with history",
    )

    ACTION_TEMPLATE = PromptTemplate(
        template="""Action {iteration}: {tool_name}[{args}]""",
        description="Template for action formatting",
    )

    OBSERVATION_TEMPLATE = PromptTemplate(
        template="""Observation {iteration}: {result}""",
        description="Template for observation formatting",
    )

    @classmethod
    def build_initial_prompt(
        cls,
        task: str,
        tools: list[str],
        context: Optional[dict[str, Any]] = None,
    ) -> str:
        """Build the initial ReAct prompt.

        Args:
            task: The task to execute.
            tools: List of available tool names.
            context: Optional context dictionary.

        Returns:
            Formatted prompt string.
        """
        tools_str = ", ".join(tools) if tools else "No specific tools available"

        context_section = ""
        if context:
            other_context = {k: v for k, v in context.items() if k != "tools"}
            if other_context:
                context_section = "\nContext:\n"
                for key, value in other_context.items():
                    context_section += f"- {key}: {value}\n"

        return cls.INITIAL.format(
            tools=tools_str,
            context_section=context_section,
            task=task,
        )

    @classmethod
    def build_continuation_prompt(
        cls,
        history: str,
        iteration: int,
    ) -> str:
        """Build continuation prompt with history.

        Args:
            history: Previous Thought-Action-Observation history.
            iteration: Current iteration number.

        Returns:
            Formatted prompt string.
        """
        return cls.CONTINUATION.format(
            history=history,
            iteration=iteration,
        )


# =============================================================================
# MCTS Prompts
# =============================================================================


class MCTSPrompts:
    """Prompt templates for MCTSStrategy.

    Provides tree search and simulation prompts.
    """

    SYSTEM = PromptTemplate(
        template="""You are a strategic decision-making agent using tree search.
You explore decision trees by generating options, simulating outcomes, and selecting optimal paths.""",
        description="System prompt for MCTS reasoning",
    )

    EXPAND = PromptTemplate(
        template="""You are analyzing a decision problem using tree search.

Task: {task}
{context_section}
Current state: {state}
Path from root: {path}

Generate 3-5 possible next actions or decisions from this state.
Each action should be distinct and lead to a different outcome.

Format your response as:
Action 1: [brief description]
Action 2: [brief description]
Action 3: [brief description]
... and so on.

Generate possible actions:""",
        description="Prompt for expanding a node",
    )

    SIMULATE = PromptTemplate(
        template="""You are simulating the outcome of a decision path.

Task: {task}
{context_section}
Decision path: {path}
Current decision: {state}

Simulate the outcome of following this decision path. Consider:
1. Likely results and consequences
2. Potential risks and benefits
3. Overall quality of this path

Rate the outcome on a scale of 0 to 100, where:
- 0-30: Poor outcome
- 31-60: Moderate outcome
- 61-80: Good outcome
- 81-100: Excellent outcome

Format your response as:
Score: [0-100]
Reasoning: [brief explanation]

Evaluate:""",
        description="Prompt for simulating outcome",
    )

    SOLUTION = PromptTemplate(
        template="""Based on tree search analysis, the optimal decision path is:

Task: {task}
{context_section}
Optimal path:
{path_formatted}

Provide a comprehensive solution following this path. Include:
1. Summary of the recommended approach
2. Key steps for implementation
3. Expected outcomes and benefits

Solution:""",
        description="Prompt for generating final solution",
    )

    @classmethod
    def build_expansion_prompt(
        cls,
        task: str,
        state: str,
        path: list[str],
        context: Optional[dict[str, Any]] = None,
    ) -> str:
        """Build node expansion prompt.

        Args:
            task: The original task.
            state: Current state description.
            path: Path from root to current state.
            context: Optional context dictionary.

        Returns:
            Formatted prompt string.
        """
        context_section = ""
        if context:
            context_section = "\nContext:\n"
            for key, value in context.items():
                context_section += f"- {key}: {value}\n"

        path_str = " -> ".join(path) if path else "root"

        return cls.EXPAND.format(
            task=task,
            context_section=context_section,
            state=state,
            path=path_str,
        )

    @classmethod
    def build_simulation_prompt(
        cls,
        task: str,
        state: str,
        path: list[str],
        context: Optional[dict[str, Any]] = None,
    ) -> str:
        """Build simulation prompt.

        Args:
            task: The original task.
            state: Current state description.
            path: Path to current state.
            context: Optional context dictionary.

        Returns:
            Formatted prompt string.
        """
        context_section = ""
        if context:
            context_section = "\nContext:\n"
            for key, value in context.items():
                context_section += f"- {key}: {value}\n"

        path_str = " -> ".join(path) if path else "root"

        return cls.SIMULATE.format(
            task=task,
            context_section=context_section,
            path=path_str,
            state=state,
        )

    @classmethod
    def build_solution_prompt(
        cls,
        task: str,
        best_path: list[str],
        context: Optional[dict[str, Any]] = None,
    ) -> str:
        """Build final solution prompt.

        Args:
            task: The original task.
            best_path: The optimal path found.
            context: Optional context dictionary.

        Returns:
            Formatted prompt string.
        """
        context_section = ""
        if context:
            context_section = "\nContext:\n"
            for key, value in context.items():
                context_section += f"- {key}: {value}\n"

        path_formatted = "\n".join(
            f"  {i+1}. {step}" for i, step in enumerate(best_path)
        )

        return cls.SOLUTION.format(
            task=task,
            context_section=context_section,
            path_formatted=path_formatted,
        )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "PromptTemplate",
    "DirectPrompts",
    "ChainOfThoughtPrompts",
    "TreeOfThoughtsPrompts",
    "ReActPrompts",
    "MCTSPrompts",
]
