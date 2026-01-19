"""Entry point for running CLI as module.

Usage:
    python -m sigil.interfaces.cli orchestrate --task "..."
    python -m sigil.interfaces.cli log-stream
    python -m sigil.interfaces.cli status
"""

if __name__ == "__main__":
    # Import inside if __name__ to avoid RuntimeWarning about module already loaded
    from sigil.interfaces.cli.app import main
    main()
