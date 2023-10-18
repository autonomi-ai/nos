from dataclasses import dataclass
from pathlib import Path

from nos.constants import NOS_MODELS_DIR
from nos.logging import redirect_stdout_to_logger


NOS_CUSTOM_MODELS_DIR = Path(NOS_MODELS_DIR) / "custom"


@dataclass(frozen=True)
class NosHubConfig:
    """NOS Hub configuration."""

    namespace: str
    """Namespace (repository, organization)."""
    name: str
    """Model name."""


@dataclass(frozen=True)
class TorchHubConfig:
    """PyTorch Hub configuration."""

    repo: str
    """Repository name (e.g. pytorch/vision)."""
    model_name: str
    """Model name (e.g. resnet18)."""
    checkpoint: str = None
    """Checkpoint name (e.g. resnet18-5c106cde.pth)."""


@dataclass(frozen=True)
class HuggingFaceHubConfig:
    """HuggingFace Hub configuration."""

    model_name: str
    """Model name (e.g. bert-base-uncased)."""
    checkpoint: str = None
    """Checkpoint name (e.g. bert-base-uncased-pytorch_model.bin)."""


def cached_checkpoint(url: str, model_id: str) -> str:
    """Download the checkpoint and return the local path.

    Args:
        url (str): URL to the checkpoint.
        model_id (str): Model identifier.

    Returns:
        str: Local path to the checkpoint.
    """
    from torchvision.datasets.utils import download_url

    # Download the checkpoint and place it in the model directory (with the same filename)
    directory = Path(NOS_MODELS_DIR) / model_id
    # Capture stdout and stderr, and redirect to logger
    with redirect_stdout_to_logger(level="DEBUG"):
        download_url(url, str(directory))
    filename = directory / Path(url).name

    # Check that the file exists
    if not filename.exists():
        raise IOError(f"Failed to download checkpoint={url}.")
    return str(filename)
