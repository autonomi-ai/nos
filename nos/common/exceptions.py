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


class NosServerReadyException(NosClientException):
    """Exception raised when the server is not ready."""

    def __str__(self) -> str:
        return f"Server not ready. {self.message}"


class NosInputValidationException(NosClientException):
    """Exception raised when input validation fails."""

    def __str__(self) -> str:
        return f"Input validation failed. {self.message}"


class NosInferenceException(NosClientException):
    """Exception raised when inference fails."""

    def __str__(self) -> str:
        return f"Inference failed. {self.message}"
