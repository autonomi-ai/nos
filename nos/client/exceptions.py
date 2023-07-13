"""Custom exceptions for the nos client."""
from dataclasses import dataclass


@dataclass(frozen=True)
class NosClientException(Exception):
    """Base exception for the nos client."""

    message: str
    """Exception message."""
    exc: Exception = None
    """Exception object."""

    def __str__(self) -> str:
        return f"{self.message}"
