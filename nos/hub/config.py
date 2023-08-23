from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union

from nos.common.metaclass import SingletonMetaclass
from nos.constants import NOS_MODELS_DIR
from nos.logging import logger


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
    download_url(url, str(directory))
    filename = directory / Path(url).name

    # Check that the file exists
    if not filename.exists():
        raise IOError(f"Failed to download checkpoint={url}.")
    return str(filename)


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

    @property
    def cached_checkpoint(self) -> str:
        if Path(self.checkpoint).exists():
            return self.checkpoint
        return cached_checkpoint(self.checkpoint, self.model_name)


@dataclass(frozen=True)
class MMLabHub(metaclass=SingletonMetaclass):
    """OpenMMlab model registry."""

    work_dirs: List[Union[str, Path]] = field(
        default_factory=lambda: [NOS_CUSTOM_MODELS_DIR / "openmmlab/mmdetection"]
    )
    """List of model working directories with pre-trained models."""

    _models: Dict[str, MMLabConfig] = field(default_factory=dict)
    """Model registry."""

    def __post_init__(self):
        """Post-initialization."""
        for directory in self.work_dirs:
            logger.debug(f"Registering checkpoints from directory: {directory}")
            self._register_checkpoints(str(directory))

    def _register_checkpoints(self, directory: str, namespace: str = "custom"):
        """Register checkpoints from a directory.

        Registered models:
            - {namespace}_{model_stem}_{checkpoint_stem}: Specific checkpoint.
            - {namespace}_{model_stem}_latest: Latest checkpoint.
        """
        directory = Path(directory)
        if not directory.exists():
            logger.debug(f"Skipping directory, does not exist [dir={directory}].")
            return

        # Create one entry per model .pth file
        # {namespace}_{model_stem}_{checkpoint_stem}
        for path in directory.rglob("*.pth"):
            model_dir = path.parent
            model_stem = model_dir.stem
            config = model_dir / f"{model_stem}.py"
            checkpoint = path
            assert config.exists(), f"Failed to find config: {config}."
            assert checkpoint.exists(), f"Failed to find checkpoint: {checkpoint}."
            mm_config = MMLabConfig(config=str(config), checkpoint=str(checkpoint))
            key = f"{namespace}_{model_stem}_{path.stem}"
            self._models[key] = mm_config
            logger.debug(f"Registering model [key={key}, cfg={mm_config}]")

        # Create one entry for the latest checkpoint
        # {namespace}_{model_stem}_latest
        for path in directory.rglob("last_checkpoint"):
            model_dir = path.parent
            model_stem = model_dir.stem
            if path.exists():
                latest_filename = None
                try:
                    with open(str(path), "r") as f:
                        latest_filename = f.read().strip()
                        latest_basename = Path(latest_filename).name
                    config = checkpoint.parent / f"{model_stem}.py"
                    checkpoint = model_dir / latest_basename
                    assert checkpoint.exists(), f"Failed to find latest checkpoint: {checkpoint}."
                    mm_config = MMLabConfig(config=str(config), checkpoint=str(checkpoint))
                    key = f"{namespace}_{model_stem}_latest"
                    self._models[key] = mm_config
                    logger.debug(f"Registering latest model [key={model_stem}, cfg={mm_config}]")
                except Exception as e:
                    logger.warning(f"Failed to load latest checkpoint: {e}")

    def __contains__(self, key: str) -> bool:
        return key in self._models

    def __getitem__(self, key: str) -> MMLabConfig:
        return self._models[key]

    def __len__(self) -> int:
        return len(self._models)

    def __iter__(self):
        return iter(self._models)

    def get(self, key: str) -> MMLabConfig:
        return self.__getitem__(key)
