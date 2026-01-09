"""ACTi Agent Builder - A meta-agent framework for creating executable AI agents with real tool capabilities."""

import logging

__version__ = "0.1.0"

# Configure default logging for the package
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Set default level for the package's loggers
logging.getLogger("src").setLevel(logging.INFO)

from .builder import create_builder

__all__ = ["create_builder", "__version__"]
