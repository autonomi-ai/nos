"""Custom exceptions for the nos package."""
from dataclasses import dataclass


@dataclass(frozen=True)
class NosServerException(Exception):
    """Base exception for the nos client."""

    message: str
    """Exception message."""
    exc: Exception = None
    """Exception object."""

    def __str__(self) -> str:
        return f"{self.message}"


class ModelNotFoundError(NosServerException):
    """Exception raised when the model is not found."""

    def __str__(self) -> str:
        return f"Model not found, details={self.message}"
