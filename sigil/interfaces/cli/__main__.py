"""Entry point for running CLI as module.

Usage:
    python -m sigil.interfaces.cli orchestrate --task "..."
    python -m sigil.interfaces.cli log-stream
    python -m sigil.interfaces.cli status
"""

from sigil.interfaces.cli.app import main

if __name__ == "__main__":
    main()
