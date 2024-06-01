"""Custom exceptions for the nos package."""
from dataclasses import dataclass

from nos.logging import logger


@dataclass
class ServerException(Exception):
    """Base exception for the nos client."""

    message: str
    """Exception message."""
    exc: Exception = None
    """Exception object."""

    def __post_init__(self) -> None:
        if self.exc is not None:
            self.message = f"{self.message}, details={self.exc}"
        logger.error(self.message)

    def __str__(self) -> str:
        return f"{self.message}"


class ModelNotFoundError(ServerException):
    """Exception raised when the model is not found."""


class OutOfDeviceMemoryError(ServerException):
    """Exception raised when the device is out of memory."""
