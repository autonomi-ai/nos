from dataclasses import dataclass
from pathlib import Path

from nos.constants import NOS_MODELS_DIR


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


@dataclass(frozen=True)
class MMLabConfig:
    """OpenMMlab configuration."""

    config: str
    """Model configuration file."""

    checkpoint: str
    """Model checkpoint file."""

    def validate(self):
        """Validate the configuration."""
        if not Path(self.config).exists():
            raise IOError(f"Failed to load config={self.config}.")
        if not Path(self.checkpoint).exists():
            raise IOError(f"Failed to load checkpoint={self.checkpoint}.")
        return self

    @property
    def model_name(self) -> str:
        """Model name.

        Returns:
            str: Model name.
        """
        return Path(self.config).stem

    def cached_checkpoint(self) -> str:
        """Download the checkpoint and return the local path.

        Returns:
            str: Local path to the checkpoint.
        """
        from torchvision.datasets.utils import download_url

        # Download the checkpoint and place it in the model directory (with the same filename)
        directory = Path(NOS_MODELS_DIR) / self.model_name
        download_url(self.checkpoint, str(directory))
        filename = directory / Path(self.checkpoint).name

        # Check that the file exists
        if not filename.exists():
            raise IOError(f"Failed to download checkpoint={self.checkpoint}.")
        return str(filename)
