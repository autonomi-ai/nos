import os
import sys
from datetime import datetime

from .constants import NOS_LOG_DIR


# Set the loguru logger level to the same level as the nos logger
date = datetime.utcnow().strftime("%Y-%m-%d_%H%M%S")
NOS_LOGGING_PATH = os.getenv("NOS_LOGGING_PATH", os.path.join(NOS_LOG_DIR, f"nos-{date}.log"))
NOS_LOGGING_ROTATION = os.getenv("NOS_LOGGING_ROTATION", "100 MB")
NOS_LOGGING_LEVEL = os.getenv("NOS_LOGGING_LEVEL", "INFO")
LOGURU_LEVEL = os.getenv("LOGURU_LEVEL", NOS_LOGGING_LEVEL)


def build_logger(name: str = None):
    """Get a logger with the specified name"""
    from loguru import logger as _logger

    # Set loguru logger level to the same level as the nos logger
    # and automatically rotate big files
    if name:
        filename = os.path.join(NOS_LOG_DIR, f"nos-{name}-{date}.log")
    else:
        filename = NOS_LOGGING_PATH

    # Remove the default logger
    _logger.remove()
    # Add a file logging handler
    _logger.add(filename, rotation=NOS_LOGGING_ROTATION)
    # Add a stdout/stderr logging handlers
    _logger.add(sys.stderr, level="WARNING")
    _logger.add(sys.stdout, level="INFO")
    return _logger


# Default logger
logger = build_logger()
