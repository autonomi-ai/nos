import contextlib
import os
import sys
from dataclasses import dataclass
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
    _logger.add(sys.stderr, level="ERROR")
    return _logger


# Default logger
logger = build_logger()


@dataclass
class StreamToLogger:
    """Virtual file-like stream object that redirects writes to a logger instance."""

    level: str = "INFO"

    def write(self, buffer):
        for line in buffer.rstrip().splitlines():
            logger.opt(depth=1).log(self.level, line.rstrip())

    def flush(self):
        pass


@contextlib.contextmanager
def redirect_to_logger(redirect_contextlib_func, level: str = "INFO"):
    """Redirect stdout/stderr to the specified logger"""
    assert redirect_contextlib_func in (contextlib.redirect_stdout, contextlib.redirect_stderr)
    assert level in ("INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL")
    stream = StreamToLogger(level=level)
    with redirect_contextlib_func(stream):
        yield


def redirect_stdout_to_logger(level: str = "INFO"):
    """Redirect stdout to the specified logger"""
    return redirect_to_logger(contextlib.redirect_stdout, level=level)


def redirect_stderr_to_logger(level: str = "INFO"):
    """Redirect stderr to the specified logger"""
    return redirect_to_logger(contextlib.redirect_stderr, level=level)
