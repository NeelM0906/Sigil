"""Sigil v2 CLI Application.

This module implements the command-line interface for Sigil v2, providing
commands for orchestrated pipeline execution and real-time log monitoring.

Commands:
    orchestrate: Run a message through the complete pipeline (Route -> Plan -> Reason -> Validate)
    log-stream: Tail the execution log file in real-time
    status: Show current orchestrator status and metrics

Usage:
    python -m sigil.interfaces.cli.app orchestrate --task "..." --session-id "..."
    python -m sigil.interfaces.cli.app log-stream
    python -m sigil.interfaces.cli.app status

Example:
    $ python -m sigil.interfaces.cli.app orchestrate --task "Qualify lead John from Acme" --session test-1
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

import click

from sigil.interfaces.cli.monitoring import (
    TokenDisplay,
    SigilLogFormatter,
    PipelineTokenTracker,
    setup_execution_logging,
    get_log_file_path,
    LOG_FILE_PATH,
)


# =============================================================================
# Module Logger
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Token budget for 256K model
TOTAL_TOKEN_BUDGET = 256_000

# ANSI color codes for terminal output
COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
}


def colorize(text: str, color: str) -> str:
    """Apply ANSI color to text."""
    if not sys.stdout.isatty():
        return text
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"


# =============================================================================
# CLI Application
# =============================================================================


@click.group()
@click.version_option(version="2.0.0", prog_name="sigil")
@click.option("--debug/--no-debug", default=False, help="Enable debug output")
@click.pass_context
def cli(ctx: click.Context, debug: bool) -> None:
    """Sigil v2 - Self-improving agent framework CLI.

    Run AI agent pipelines with memory, planning, reasoning, and contracts.
    """
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug

    # Set up basic logging
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


@cli.command()
@click.option(
    "--task", "-t",
    required=True,
    help="The task or message to process through the pipeline"
)
@click.option(
    "--session-id", "-s",
    default=None,
    help="Session identifier (auto-generated if not provided)"
)
@click.option(
    "--user-id", "-u",
    default=None,
    help="Optional user identifier for personalization"
)
@click.option(
    "--contract", "-c",
    default=None,
    help="Optional contract name to enforce on output"
)
@click.option(
    "--strategy",
    default=None,
    type=click.Choice(["direct", "chain_of_thought", "tree_of_thoughts", "react", "mcts"]),
    help="Force a specific reasoning strategy"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show detailed pipeline output"
)
@click.pass_context
def orchestrate(
    ctx: click.Context,
    task: str,
    session_id: Optional[str],
    user_id: Optional[str],
    contract: Optional[str],
    strategy: Optional[str],
    verbose: bool,
) -> None:
    """Run a task through the complete Sigil pipeline.

    Executes the full orchestration: Route -> Plan -> Context -> Execute -> Validate.
    Shows all subsystem outputs including routing decision, plan, strategy selected,
    reasoning output, and token usage.

    Example:
        sigil orchestrate --task "Qualify lead John from Acme" --session test-1
    """
    # Generate session ID if not provided
    if session_id is None:
        session_id = f"session-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    # Set up execution logging
    setup_execution_logging()

    # Create pipeline token tracker for this execution
    token_tracker = PipelineTokenTracker()

    click.echo()
    click.echo(colorize("=" * 70, "cyan"))
    click.echo(colorize("  SIGIL v2 ORCHESTRATOR", "bold"))
    click.echo(colorize("=" * 70, "cyan"))
    click.echo()

    click.echo(f"{colorize('Task:', 'bold')} {task}")
    click.echo(f"{colorize('Session:', 'bold')} {session_id}")
    if user_id:
        click.echo(f"{colorize('User:', 'bold')} {user_id}")
    if contract:
        click.echo(f"{colorize('Contract:', 'bold')} {contract}")
    if strategy:
        click.echo(f"{colorize('Strategy Override:', 'bold')} {strategy}")
    click.echo()
    click.echo(colorize("-" * 70, "dim"))
    click.echo()

    # Run the orchestration
    try:
        result = asyncio.run(
            _run_orchestration(
                task=task,
                session_id=session_id,
                user_id=user_id,
                contract_name=contract,
                force_strategy=strategy,
                verbose=verbose,
                token_tracker=token_tracker,
            )
        )

        # Display results
        _display_orchestration_result(result, token_tracker, verbose)

    except Exception as e:
        logger.exception("Orchestration failed")
        click.echo()
        click.echo(colorize(f"ERROR: {str(e)}", "red"))
        click.echo()
        raise SystemExit(1)


async def _run_orchestration(
    task: str,
    session_id: str,
    user_id: Optional[str],
    contract_name: Optional[str],
    force_strategy: Optional[str],
    verbose: bool,
    token_tracker: PipelineTokenTracker,
) -> dict[str, Any]:
    """Execute the orchestration pipeline with progress reporting."""
    from dotenv import load_dotenv
    from pathlib import Path
    from sigil.orchestrator import SigilOrchestrator, OrchestratorRequest
    from sigil.interfaces.cli.monitoring import log_pipeline_step

    # Load environment variables from .env if present
    env_file = Path(".env")
    if env_file.exists():
        load_dotenv(env_file, override=False)

    # Enable features for interactive CLI
    os.environ.setdefault("SIGIL_USE_PLANNING", "true")
    os.environ.setdefault("SIGIL_USE_ROUTING", "true")
    os.environ.setdefault("SIGIL_USE_MEMORY", "true")

    # Create orchestrator
    orchestrator = SigilOrchestrator()

    # Create request
    request = OrchestratorRequest(
        message=task,
        session_id=session_id,
        user_id=user_id,
        contract_name=contract_name,
        force_strategy=force_strategy,
    )

    click.echo(colorize("Pipeline Execution:", "bold"))
    click.echo()

    # Log start
    log_pipeline_step("ORCHESTRATOR", "Starting pipeline execution", 0)

    # Execute pipeline
    start_time = time.perf_counter()
    response = await orchestrator.process(request)
    execution_time = (time.perf_counter() - start_time) * 1000

    # Build result with all subsystem outputs
    result = {
        "status": response.status.value,
        "request_id": response.request_id,
        "execution_time_ms": execution_time,
        "route_decision": None,
        "plan": None,
        "reasoning": None,
        "validation": None,
        "output": response.output,
        "errors": response.errors,
        "warnings": response.warnings,
        "tokens": {
            "total": response.tokens_used,
            "budget": TOTAL_TOKEN_BUDGET,
            "percentage": (response.tokens_used / TOTAL_TOKEN_BUDGET) * 100,
        },
    }

    # Extract routing info
    if response.route_decision:
        rd = response.route_decision
        result["route_decision"] = {
            "intent": rd.intent.value,
            "confidence": rd.confidence,
            "complexity": rd.complexity,
            "handler": rd.handler_name,
            "use_planning": rd.use_planning,
            "use_memory": rd.use_memory,
            "use_contracts": rd.use_contracts,
        }
        token_tracker.record_routing(50)  # Estimate routing tokens
        log_pipeline_step(
            "ROUTING",
            f"Intent: {rd.intent.value}, Complexity: {rd.complexity:.2f}",
            50
        )

    # Extract plan info (if available)
    if response.plan_id:
        result["plan"] = {
            "plan_id": response.plan_id,
        }
        # Planning tokens would be tracked by the planner
        log_pipeline_step(
            "PLANNING",
            f"Plan created: {response.plan_id}",
            0
        )
    else:
        complexity = result.get("route_decision", {}).get("complexity", 0)
        log_pipeline_step(
            "PLANNING",
            f"No plan needed for complexity {complexity:.2f}",
            0
        )

    # Extract reasoning info from output
    if response.output:
        reasoning_info = {
            "result": response.output.get("result"),
            "model": response.output.get("model"),
            "confidence": response.output.get("confidence"),
        }
        if response.output.get("reasoning_trace"):
            reasoning_info["trace_length"] = len(response.output.get("reasoning_trace", []))
        result["reasoning"] = reasoning_info

        # Log reasoning step
        reasoning_tokens = response.tokens_used - token_tracker.routing_tokens - token_tracker.memory_tokens
        token_tracker.record_reasoning(max(0, reasoning_tokens))
        log_pipeline_step(
            "REASONING",
            f"Completed with confidence: {reasoning_info.get('confidence', 'N/A')}",
            reasoning_tokens
        )

    # Extract validation info
    if response.contract_result:
        cr = response.contract_result
        result["validation"] = {
            "is_valid": cr.is_valid,
            "contract_name": cr.contract_name,
            "tokens_used": cr.tokens_used,
        }
        token_tracker.record_validation(cr.tokens_used)
        log_pipeline_step(
            "VALIDATION",
            f"Contract '{cr.contract_name}': {'PASSED' if cr.is_valid else 'FAILED'}",
            cr.tokens_used
        )
    else:
        log_pipeline_step(
            "VALIDATION",
            "No contract specified",
            0
        )

    # Log completion
    log_pipeline_step(
        "COMPLETE",
        f"Total: {response.tokens_used:,} tokens / {TOTAL_TOKEN_BUDGET:,} ({result['tokens']['percentage']:.2f}%)",
        response.tokens_used
    )

    return result


def _display_orchestration_result(
    result: dict[str, Any],
    token_tracker: PipelineTokenTracker,
    verbose: bool,
) -> None:
    """Display the orchestration result in a formatted way."""
    click.echo()
    click.echo(colorize("-" * 70, "dim"))
    click.echo()

    # Status
    status = result["status"]
    status_color = "green" if status == "success" else "red" if status == "failed" else "yellow"
    click.echo(f"{colorize('Status:', 'bold')} {colorize(status.upper(), status_color)}")
    click.echo(f"{colorize('Execution Time:', 'bold')} {result['execution_time_ms']:.2f}ms")
    click.echo()

    # Routing Decision
    if result["route_decision"]:
        rd = result["route_decision"]
        click.echo(colorize("ROUTING DECISION", "cyan"))
        click.echo(f"  Intent: {rd['intent']}")
        click.echo(f"  Confidence: {rd['confidence']:.2f}")
        click.echo(f"  Complexity: {rd['complexity']:.2f}")
        click.echo(f"  Handler: {rd['handler']}")
        click.echo(f"  Planning: {'Enabled' if rd['use_planning'] else 'Disabled'}")
        click.echo(f"  Memory: {'Enabled' if rd['use_memory'] else 'Disabled'}")
        click.echo(f"  Contracts: {'Enabled' if rd['use_contracts'] else 'Disabled'}")
        click.echo()

    # Plan
    if result["plan"]:
        click.echo(colorize("PLAN", "cyan"))
        click.echo(f"  Plan ID: {result['plan']['plan_id']}")
        click.echo()

    # Reasoning Output
    if result["reasoning"]:
        click.echo(colorize("REASONING OUTPUT", "cyan"))
        if result["reasoning"].get("result"):
            # Truncate long results
            answer = str(result["reasoning"]["result"])
            if len(answer) > 500 and not verbose:
                answer = answer[:500] + "... (use --verbose for full output)"
            click.echo(f"  Result: {answer}")
        if result["reasoning"].get("model"):
            click.echo(f"  Model: {result['reasoning']['model']}")
        if result["reasoning"].get("confidence"):
            click.echo(f"  Confidence: {result['reasoning']['confidence']}")
        click.echo()

    # Validation
    if result["validation"]:
        vr = result["validation"]
        valid_color = "green" if vr["is_valid"] else "red"
        click.echo(colorize("VALIDATION", "cyan"))
        click.echo(f"  Contract: {vr['contract_name']}")
        click.echo(f"  Valid: {colorize(str(vr['is_valid']), valid_color)}")
        click.echo()

    # Errors and Warnings
    if result["errors"]:
        click.echo(colorize("ERRORS", "red"))
        for error in result["errors"]:
            click.echo(f"  - {error}")
        click.echo()

    if result["warnings"]:
        click.echo(colorize("WARNINGS", "yellow"))
        for warning in result["warnings"]:
            click.echo(f"  - {warning}")
        click.echo()

    # Token Usage Summary
    click.echo(colorize("TOKEN USAGE", "cyan"))
    tokens = result["tokens"]

    # Display per-step breakdown
    display = TokenDisplay(TOTAL_TOKEN_BUDGET)
    click.echo(display.format_pipeline_summary(token_tracker))

    # Display total
    click.echo()
    click.echo(f"  {colorize('Total:', 'bold')} {tokens['total']:,} / {tokens['budget']:,} ({tokens['percentage']:.2f}%)")

    # Budget warning
    if tokens["percentage"] > 80:
        click.echo()
        click.echo(colorize(f"  WARNING: Token usage above 80% of budget!", "yellow"))

    click.echo()
    click.echo(colorize("=" * 70, "cyan"))


@cli.command("log-stream")
@click.option(
    "--follow", "-f",
    is_flag=True,
    default=True,
    help="Follow the log file (like tail -f)"
)
@click.option(
    "--lines", "-n",
    default=20,
    type=int,
    help="Number of lines to show initially"
)
@click.pass_context
def log_stream(ctx: click.Context, follow: bool, lines: int) -> None:
    """Tail the Sigil execution log file in real-time.

    Shows DEBUG level logs with timestamps and token costs.
    Each operation is prefixed with the component name and token cost.

    Format: [TIMESTAMP] [LEVEL] [COMPONENT] [TOKENS] Message

    Example:
        sigil log-stream
        sigil log-stream --lines 50
    """
    log_file = get_log_file_path()

    click.echo()
    click.echo(colorize(f"Tailing log file: {log_file}", "cyan"))
    click.echo(colorize("Press Ctrl+C to stop", "dim"))
    click.echo(colorize("-" * 70, "dim"))
    click.echo()

    try:
        _tail_log_file(log_file, lines, follow)
    except KeyboardInterrupt:
        click.echo()
        click.echo(colorize("Stopped.", "dim"))
    except FileNotFoundError:
        click.echo(colorize(f"Log file not found: {log_file}", "yellow"))
        click.echo("Run an orchestrate command to generate logs.")


def _tail_log_file(log_file: Path, initial_lines: int, follow: bool) -> None:
    """Tail a log file, optionally following new content."""
    # Read initial lines
    if log_file.exists():
        with open(log_file, "r") as f:
            all_lines = f.readlines()
            # Show last N lines
            start_idx = max(0, len(all_lines) - initial_lines)
            for line in all_lines[start_idx:]:
                _print_log_line(line.rstrip())

    if not follow:
        return

    # Follow mode - watch for new content
    last_size = log_file.stat().st_size if log_file.exists() else 0

    while True:
        time.sleep(0.1)  # Poll interval

        if not log_file.exists():
            continue

        current_size = log_file.stat().st_size

        if current_size > last_size:
            with open(log_file, "r") as f:
                f.seek(last_size)
                new_content = f.read()
                for line in new_content.splitlines():
                    if line.strip():
                        _print_log_line(line)
            last_size = current_size
        elif current_size < last_size:
            # File was truncated, start from beginning
            last_size = 0


def _print_log_line(line: str) -> None:
    """Print a log line with color formatting."""
    # Color based on log level
    if "[DEBUG]" in line:
        click.echo(colorize(line, "dim"))
    elif "[INFO]" in line:
        click.echo(line)
    elif "[WARNING]" in line:
        click.echo(colorize(line, "yellow"))
    elif "[ERROR]" in line:
        click.echo(colorize(line, "red"))
    elif "[CRITICAL]" in line:
        click.echo(colorize(line, "red"))
    else:
        click.echo(line)


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show current orchestrator status and metrics.

    Displays:
    - Active features
    - Request statistics
    - Token usage summary
    - System health
    """
    from sigil.config import get_settings
    from sigil.orchestrator import SigilOrchestrator

    settings = get_settings()

    click.echo()
    click.echo(colorize("SIGIL v2 STATUS", "bold"))
    click.echo(colorize("=" * 50, "cyan"))
    click.echo()

    # Active features
    click.echo(colorize("Active Features:", "bold"))
    features = settings.get_active_features()
    if features:
        for feature in features:
            click.echo(f"  - {feature}")
    else:
        click.echo("  (none enabled)")
    click.echo()

    # Environment
    click.echo(colorize("Environment:", "bold"))
    click.echo(f"  Environment: {settings.environment}")
    click.echo(f"  Debug: {settings.debug}")
    click.echo(f"  Log Level: {settings.log_level}")
    click.echo()

    # Log file status
    click.echo(colorize("Log File:", "bold"))
    log_file = get_log_file_path()
    if log_file.exists():
        size_kb = log_file.stat().st_size / 1024
        click.echo(f"  Path: {log_file}")
        click.echo(f"  Size: {size_kb:.2f} KB")
    else:
        click.echo(f"  Path: {log_file}")
        click.echo("  Status: Not created yet")
    click.echo()

    # Create orchestrator to get metrics
    try:
        orchestrator = SigilOrchestrator()
        metrics = orchestrator.get_metrics()

        click.echo(colorize("Orchestrator Metrics:", "bold"))
        click.echo(f"  Total Requests: {metrics['total_requests']}")
        click.echo(f"  Successful: {metrics['successful_requests']}")
        click.echo(f"  Failed: {metrics['failed_requests']}")
        click.echo(f"  Success Rate: {metrics['success_rate']:.1%}")
        click.echo(f"  Total Tokens: {metrics['total_tokens']:,}")
        click.echo(f"  Avg Tokens/Request: {metrics['avg_tokens_per_request']:.0f}")
        click.echo()

        # Health check
        health_color = "green" if orchestrator.is_healthy else "red"
        click.echo(f"{colorize('Health:', 'bold')} {colorize('OK' if orchestrator.is_healthy else 'UNHEALTHY', health_color)}")

    except Exception as e:
        click.echo(colorize(f"Could not initialize orchestrator: {e}", "yellow"))

    click.echo()


# =============================================================================
# Interactive REPL Command
# =============================================================================


@cli.command()
@click.option("--session-id", "-s", default=None, help="Session identifier")
@click.option("--user-id", "-u", default=None, help="User identifier")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
@click.option("--no-planning", is_flag=True, help="Disable planning subsystem")
@click.option("--no-routing", is_flag=True, help="Disable routing subsystem")
@click.option("--no-memory", is_flag=True, help="Disable memory subsystem")
@click.pass_context
def interactive(
    ctx: click.Context,
    session_id: Optional[str],
    user_id: Optional[str],
    verbose: bool,
    no_planning: bool,
    no_routing: bool,
    no_memory: bool,
) -> None:
    """Start an interactive REPL session.

    Type prompts and get responses with planning, routing, memory, and token tracking.
    Type 'exit' or 'quit' to exit.

    By default, all subsystems (planning, routing, memory) are ENABLED for interactive
    sessions to allow testing the full pipeline. Use --no-planning, --no-routing, or
    --no-memory flags to disable specific subsystems.
    """
    import uuid
    from sigil.config.settings import clear_settings_cache

    # Enable feature flags for interactive session by default
    # These override .env settings to ensure full pipeline testing
    # Users can disable with --no-* flags
    if not no_planning:
        os.environ["SIGIL_USE_PLANNING"] = "true"
    if not no_routing:
        os.environ["SIGIL_USE_ROUTING"] = "true"
    if not no_memory:
        os.environ["SIGIL_USE_MEMORY"] = "true"

    # Clear cached settings so new env vars take effect
    clear_settings_cache()

    session_id = session_id or f"cli-{uuid.uuid4().hex[:8]}"

    click.echo()
    click.echo(colorize("=" * 80, "bold"))
    click.echo(colorize("  SIGIL v2 INTERACTIVE CLI", "bold"))
    click.echo(colorize("=" * 80, "bold"))
    click.echo()
    click.echo("Welcome! This is an interactive session with full pipeline support.")
    click.echo(f"Session ID: {session_id}")
    click.echo()

    # Show enabled features
    features_enabled = []
    if not no_routing:
        features_enabled.append("Routing")
    if not no_planning:
        features_enabled.append("Planning")
    if not no_memory:
        features_enabled.append("Memory")
    click.echo(f"Enabled: {colorize(', '.join(features_enabled), 'green')}")
    click.echo()

    # Check MCP server availability
    from sigil.interfaces.cli.health_check import display_mcp_status
    try:
        mcp_status = asyncio.run(display_mcp_status(echo_func=click.echo))
    except Exception as e:
        click.echo(colorize(f"Could not check MCP servers: {e}", "yellow"))
        mcp_status = {}

    click.echo()
    click.echo("Type your queries below. Type 'exit' or 'quit' to exit.")
    click.echo(colorize("-" * 80, "dim"))
    click.echo()

    token_tracker = PipelineTokenTracker()
    request_count = 0

    try:
        while True:
            try:
                # Get user input
                prompt = click.prompt(colorize(">>> ", "cyan"), type=str)

                # Check for exit commands
                if prompt.lower() in ("exit", "quit", "q"):
                    click.echo()
                    click.echo(colorize(f"Session ended. Total requests: {request_count}, Total tokens: {token_tracker.total:,}", "dim"))
                    break

                # Skip empty input
                if not prompt.strip():
                    continue

                request_count += 1
                click.echo()

                # Run orchestration
                try:
                    result = asyncio.run(
                        _run_orchestration(
                            task=prompt,
                            session_id=session_id,
                            user_id=user_id,
                            contract_name=None,
                            force_strategy=None,
                            verbose=verbose,
                            token_tracker=token_tracker,
                        )
                    )

                    # Display response
                    click.echo()
                    click.echo(colorize("RESPONSE:", "bold"))
                    click.echo(colorize("-" * 80, "dim"))

                    # Extract and show output - be more aggressive in finding the actual response
                    output = result.get("output")
                    response_text = None

                    if output is None:
                        response_text = "(No response)"
                    elif isinstance(output, dict):
                        # Try various fields where the response might be
                        response_text = (
                            output.get("result") or
                            output.get("answer") or
                            output.get("response") or
                            output.get("text") or
                            output.get("output") or
                            None
                        )
                        # If still nothing, check if there's a string representation
                        if not response_text and len(output) > 0:
                            # Print the whole dict for debugging
                            import json
                            response_text = json.dumps(output, indent=2)
                    elif isinstance(output, str):
                        response_text = output
                    else:
                        response_text = str(output)

                    if response_text:
                        click.echo(response_text)
                    else:
                        click.echo("(Empty response)")
                    click.echo()

                    # Show metadata
                    click.echo(colorize("METADATA:", "dim"))

                    # Routing info
                    if result.get("route_decision"):
                        rd = result["route_decision"]
                        click.echo(f"  Route: {rd['intent']} (confidence: {rd['confidence']:.0%}, complexity: {rd['complexity']:.2f})")
                        click.echo(f"  Features: Planning={rd['use_planning']}, Memory={rd['use_memory']}, Contracts={rd['use_contracts']}")

                    # Token info
                    tokens = result.get("tokens", {})
                    click.echo(f"  Tokens: {tokens.get('total', 0):,} / {tokens.get('budget', 256000):,} ({tokens.get('percentage', 0):.2f}%)")

                    # Execution time
                    click.echo(f"  Time: {result.get('execution_time_ms', 0):.1f}ms")

                    # Errors/warnings
                    if result.get("errors"):
                        click.echo(colorize(f"  Errors: {', '.join(result['errors'])}", "red"))
                    if result.get("warnings"):
                        click.echo(colorize(f"  Warnings: {', '.join(result['warnings'])}", "yellow"))

                    click.echo()
                    click.echo(colorize("-" * 80, "dim"))
                    click.echo()

                except Exception as e:
                    click.echo(colorize(f"ERROR: {str(e)}", "red"))
                    if verbose:
                        logger.exception("Orchestration failed")
                    click.echo()

            except KeyboardInterrupt:
                click.echo()
                click.echo(colorize("Interrupted.", "yellow"))
                break

    except EOFError:
        click.echo()
        click.echo(colorize("End of input.", "dim"))


# =============================================================================
# Entry Point
# =============================================================================


def main() -> None:
    """Main entry point for the CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
