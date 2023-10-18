import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Union

import torch

from nos.hub.config import NOS_MODELS_DIR
from nos.logging import logger


@dataclass(frozen=True)
class StableDiffusionDreamboothConfig:
    pass


@dataclass(frozen=True)
class StableDiffusionDreamboothLoRAConfig:
    """Stable Diffusion LoRA model configuration."""

    model_name: str
    """Model name (e.g `stabilityai/stable-diffusion-2-1`)."""
    attn_procs: str
    """Attention processors path."""
    resolution: int = 512
    """Image resoltion (width/height)."""
    dtype: torch.dtype = torch.float32
    """Data type (e.g. `torch.float32`)."""


StableDiffusionDreamboothConfigType = Union[StableDiffusionDreamboothConfig, StableDiffusionDreamboothLoRAConfig]
StableDiffusionDreamboothConfigs = {
    "stable-diffusion-dreambooth": StableDiffusionDreamboothConfig,
    "stable-diffusion-dreambooth-lora": StableDiffusionDreamboothLoRAConfig,
}


@dataclass
class StableDiffusionDreamboothHub:
    """StableDiffusion model registry."""

    namespace: str
    """Namespace (e.g diffusers/dreambooth)."""

    configs: Dict[str, StableDiffusionDreamboothConfigType] = field(init=False, default_factory=dict)
    """Model registry."""

    work_dir: Path = field(init=False, default=None)
    """List of model working directories with pre-trained models."""

    _instance: Dict[str, "StableDiffusionDreamboothHub"] = field(init=False, default=None)
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
        if self.work_dir is None:
            self.work_dir = NOS_MODELS_DIR / self.namespace
        logger.debug(f"Registering checkpoints from directory: {self.work_dir}")
        self.update()

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

        # Create one entry for the latest checkpoint
        # <model_dir>/weights/pytorch_lora_weights.safetensors
        # <model_dir>/weights/checkpoint-<number>/pytorch_lora_weights.safetensors
        for path in directory.rglob("*_metadata.json"):
            try:
                metadata = None
                with open(str(path), "r") as f:
                    metadata = json.load(f)
                model_name = metadata["model_name"]
                method = metadata["method"]
                job_metadata = metadata["job_config"]
                uuid = job_metadata["uuid"]
                weights_directory = Path(job_metadata["output_directory"])
                assert weights_directory.exists(), f"Failed to find weights directory: {weights_directory}."

                # Find the latest checkpoint
                checkpoint = Path(weights_directory) / "pytorch_lora_weights.safetensors"
                assert checkpoint.exists(), f"Failed to find latest checkpoint: {checkpoint}."

                # Register models based on the method
                if method == "stable-diffusion-dreambooth-lora":
                    sd_config = StableDiffusionDreamboothLoRAConfig(
                        model_name=model_name, attn_procs=str(weights_directory)
                    )
                else:
                    raise NotImplementedError(f"Method not implemented yet [method={method}].")

                # Register the model
                key = f"{self.namespace}/{uuid}"
                self.configs[key] = sd_config
                logger.debug(f"Registering model [key={key}, model={model_name}, cfg={sd_config}]")
            except Exception as e:
                logger.warning(f"Failed to load latest checkpoint: {e}")

    def __contains__(self, key: str) -> bool:
        return key in self.configs

    def __getitem__(self, key: str) -> StableDiffusionDreamboothConfigType:
        return self.configs[key]

    def __len__(self) -> int:
        return len(self.configs)

    def __iter__(self):
        return iter(self.configs)

    def get(self, key: str) -> StableDiffusionDreamboothConfigType:
        return self.__getitem__(key)

    def update(self) -> None:
        """Update the registry."""
        self._register_checkpoints(str(self.work_dir))
