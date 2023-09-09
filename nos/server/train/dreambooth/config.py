import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from nos.common.git import cached_repo
from nos.common.runtime_env import RuntimeEnvironmentsHub
from nos.common.spec import RuntimeEnv
from nos.constants import NOS_HOME, NOS_MODELS_DIR
from nos.logging import logger
from nos.server.train.config import TrainingJobConfig


# Register the runtime environment for fine-tuning LoRA models
GIT_TAG = "v0.20.1"
RUNTIME_ENV_NAME = "huggingface/diffusers-latest"
RuntimeEnvironmentsHub.register(
    RUNTIME_ENV_NAME,
    RuntimeEnv.from_packages(
        [f"https://github.com/huggingface/diffusers/archive/refs/tags/{GIT_TAG}.zip", "accelerate>=0.22.0", "bitsandbytes>=0.40.0", "xformers>=0.0.20"]
    ),
)

NOS_VOLUME_DIR = NOS_HOME / "volumes"
NOS_CUSTOM_MODELS_DIR = NOS_MODELS_DIR / "custom"


@dataclass
class StableDiffusionDreamboothTrainingJobConfig(TrainingJobConfig):
    """Configuration for a stable-diffusion training job."""

    model_name: str = "stabilityai/stable-diffusion-2-1"
    """Model name (e.g `stabilityai/stable-diffusion-2-1`)."""

    instance_directory: str = None
    """Image instance directory (e.g. dog)."""
    """This should be relative to the NOS_VOLUME_DIR."""

    instance_prompt: str = None
    """Image instance prompt (e.g. A photo of sks dog in a bucket)."""

    max_train_steps: int = 500
    """Maximum number of training steps."""

    resolution: int = 512
    """Image resolution."""

    seed: int = 0
    """Random seed."""

    runtime_env: RuntimeEnv = field(init=False, default_factory=lambda: RuntimeEnvironmentsHub.get(RUNTIME_ENV_NAME))
    """The runtime environment for the training job."""

    repo_directory: str = field(
        init=False,
        default=cached_repo(
            f"https://github.com/huggingface/diffusers/archive/refs/tags/{GIT_TAG}.zip",
            repo_name="diffusers",
            subdirectory="examples/dreambooth",
        ),
    )
    """The repository to use for the training job."""

    def __post_init__(self):
        if self.instance_directory is None:
            raise ValueError("instance_directory must be specified.")
        if self.instance_prompt is None:
            raise ValueError("instance_prompt must be specified.")

        logger.debug(
            f"{self.__class__.__name__} [uuid={self.uuid}, working_dir={self.working_directory}, instance_dir={self.instance_directory}]"
        )

        # Copy the instance directory to the working directory
        instance_volume_directory = NOS_VOLUME_DIR / self.instance_directory
        logger.debug(f"Instance volume directory [dir={instance_volume_directory}]")
        if not Path(instance_volume_directory).exists():
            raise IOError(f"Failed to load instance_directory={instance_volume_directory}.")
        instance_directory = Path(self.working_directory) / "instances"
        shutil.copytree(instance_volume_directory, str(instance_directory))
        nfiles = len(os.listdir(instance_directory))
        logger.debug(f"Copied instance directory from {instance_directory} [nfiles={nfiles}]")

        # Set the instance and output directories
        self.instance_directory = str(instance_directory)

    @property
    def entrypoint(self):
        """The entrypoint to run for the training job."""
        return (
            f"""accelerate launch train_dreambooth_lora.py """
            f"""--pretrained_model_name_or_path={self.model_name} """
            f"""--instance_data_dir={self.instance_directory} """
            f"""--output_dir={self.weights_directory} """
            f"""--instance_prompt="{self.instance_prompt}" """
            f"""--resolution={self.resolution} """
            f"""--train_batch_size=1 """
            f"""--gradient_accumulation_steps=1 """
            f"""--checkpointing_steps={self.max_train_steps // 5} """
            f"""--learning_rate=1e-4"""
            f"""--lr_scheduler="constant" """
            f""" --lr_warmup_steps=0 """
            f"""--max_train_steps={self.max_train_steps} """
            f"""--seed="{self.seed}" """
        )

    def job_configuration(self) -> Dict[str, Any]:
        """The job configuration for the Ray training job."""
        return {
            **super().job_configuration(),
            "entrypoint_num_gpus": 1,
        }

class StableDiffusionXLDreamboothTrainingJobConfig(StableDiffusionDreamboothTrainingJobConfig):
    
    resolution: int = 1024
    """Image resolution."""

    @property
    def entrypoint(self):
        """The entrypoint to run for the training job."""
        return (
            f"""accelerate launch train_dreambooth_lora_sdxl.py"""
            f"""--pretrained_model_name_or_path={self.model_name} """
            f"""--instance_data_dir={self.instance_directory} """
            f"""--output_dir={self.weights_directory} """
            f"""--instance_prompt="{self.instance_prompt}" """
            f"""--resolution={self.resolution} """
            f"""--train_batch_size=1 """
            f"""--checkpointing_steps={self.max_train_steps // 5} """
            f"""--learning_rate=1e-5"""
            f"""--lr_scheduler="constant" """
            f""" --lr_warmup_steps=0 """
            f"""--max_train_steps={self.max_train_steps} """
            f"""--seed="{self.seed}" """
            f"""--gradient_accumulation_steps=4 """
            f"""--enable_xformers_memory_efficient_attention """
            f"""--gradient_checkpointing """
            f"""--use_8bit_adam """
            f"""--mixed_precision="fp16" """
        )
            
