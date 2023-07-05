import os
import sys
from datetime import datetime

from .constants import NOS_LOG_DIR


# Set the loguru logger level to the same level as the nos logger
date = datetime.utcnow().strftime("%Y-%m-%d_%H%M%S")
LOGGING_PATH = os.getenv("NOS_LOGGING_PATH", os.path.join(NOS_LOG_DIR, f"nos-{date}.log"))
LOGGING_ROTATION = os.getenv("NOS_LOGGING_ROTATION", "100 MB")
LOGGING_LEVEL = os.getenv("NOS_LOGGING_LEVEL", "INFO")


def build_logger(name: str = None, level: str = LOGGING_LEVEL):
    """Get a logger with the specified name"""
    from loguru import logger as _logger

    # Set loguru logger level to the same level as the nos logger
    # and automatically rotate big files
    if name:
        os.path.join(NOS_LOG_DIR, f"nos-{name}-{date}.log")
    else:
        pass

    # Remove the default logger
    _logger.remove()
    _logger.add(sys.stdout, level=LOGGING_LEVEL)
    return _logger


# Default logger
logger = build_logger()
