from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

from nos.constants import NOS_MODELS_DIR
from nos.hub.config import cached_checkpoint
from nos.logging import logger


@dataclass(frozen=True)
class OpenMMLabConfig:
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


@dataclass
class OpenMMLabHub:
    """OpenMMlab model registry."""

    namespace: str
    """Namespace (e.g openmmlab/mmdetection)."""

    work_dir: Path = field(init=False, default=None)
    """List of model working directories with pre-trained models."""

    configs: Dict[str, OpenMMLabConfig] = field(default_factory=dict)
    """Model registry."""

    _instance: Dict[str, "OpenMMLabHub"] = field(init=False, default=None)
    """Singleton instance per key/namespace."""

    def __new__(cls, namespace: str):
        """Create a singleton instance by key/namespace."""
        if cls._instance is None:
            cls._instance = {}
        if namespace not in cls._instance:
            cls._instance[namespace] = super().__new__(cls)
        return cls._instance[namespace]

    def __post_init__(self):
        """Post-initialization."""
        self.work_dir = NOS_MODELS_DIR / self.namespace
        logger.debug(f"Registering checkpoints from directory: {self.work_dir}")
        self._register_checkpoints(str(self.work_dir))

    def _register_checkpoints(self, directory: str):
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
            mm_config = OpenMMLabConfig(config=str(config), checkpoint=str(checkpoint))
            key = f"{self.namespace}/{model_stem}_{path.stem}"
            self.configs[key] = mm_config
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
                    mm_config = OpenMMLabConfig(config=str(config), checkpoint=str(checkpoint))
                    key = f"{self.namespace}/{model_stem}_latest"
                    self.configs[key] = mm_config
                    logger.debug(f"Registering latest model [key={model_stem}, cfg={mm_config}]")
                except Exception as e:
                    logger.warning(f"Failed to load latest checkpoint: {e}")

    def __contains__(self, key: str) -> bool:
        return key in self.configs

    def __getitem__(self, key: str) -> OpenMMLabConfig:
        return self.configs[key]

    def __len__(self) -> int:
        return len(self.configs)

    def __iter__(self):
        return iter(self.configs)

    def get(self, key: str) -> OpenMMLabConfig:
        return self.__getitem__(key)
