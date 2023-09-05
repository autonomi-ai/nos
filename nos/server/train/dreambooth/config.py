import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict

from nos.common.git import cached_repo
from nos.common.spec import RuntimeEnv
from nos.constants import NOS_HOME
from nos.logging import logger
from nos.models.dreambooth.dreambooth import StableDiffusionDreamboothConfigs
from nos.server.train.config import TrainingJobConfig


GIT_TAG = "v0.20.1"

NOS_VOLUME_DIR = NOS_HOME / "volumes"
RUNTIME_ENVS = {
    "diffusers-latest": {
        "working_dir": "./nos/experimental/",
        "runtime_env": RuntimeEnv.from_packages(
            [f"https://github.com/huggingface/diffusers/archive/refs/tags/{GIT_TAG}.zip", "accelerate>=0.22.0"]
        ),
    },
}


@dataclass
class StableDiffusionTrainingJobConfig:
    """Configuration for a stable-diffusion training job."""

    model_name: str
    """Model name (e.g `stabilityai/stable-diffusion-2-1`)."""

    method: str
    """Stable diffusion training method (choice of `stable-diffusion-dreambooth-lora`)."""

    instance_directory: str
    """Image instance directory (e.g. dog)."""

    instance_prompt: str
    """Image instance prompt (e.g. A photo of sks dog in a bucket)."""

    max_train_steps: int = 500
    """Maximum number of training steps."""

    resolution: int = 512
    """Image resolution."""

    seed: int = 0
    """Random seed."""

    job_config: TrainingJobConfig = field(init=False, default=None)
    """The job configuration for the training job."""

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
        if self.method not in StableDiffusionDreamboothConfigs:
            raise ValueError(
                f"Invalid method: {self.method}, available methods: [{','.join(k for k in StableDiffusionDreamboothConfigs)}]"
            )

        # Setup the working directories (output, repo)
        runtime_env = RUNTIME_ENVS["diffusers-latest"].copy()
        runtime_env["working_dir"] = self.repo_directory

        # Create a new short unique name using method and uuid (with 8 characters)
        model_id = f"{self.method}_{uuid.uuid4().hex[:8]}"
        self.job_config = TrainingJobConfig(uuid=model_id, runtime_env=runtime_env)
        job_id = self.job_config.uuid
        working_directory = Path(self.job_config.working_directory)

        # Copy the instance directory to the working directory
        self.instance_directory = NOS_VOLUME_DIR / self.instance_directory
        logger.debug(f"Instance directory [dir={self.instance_directory}]")
        if not Path(self.instance_directory).exists():
            raise IOError(f"Failed to load instance_directory={self.instance_directory}.")
        instance_directory = working_directory / "instances"
        shutil.copytree(self.instance_directory, str(instance_directory))
        nfiles = len(os.listdir(instance_directory))
        logger.debug(f"Copied instance directory to {working_directory} [nfiles={nfiles}]")

        # Set the instance and output directories
        self.instance_directory = str(instance_directory)

        # Write the metadata and model configuration files
        logger.debug(f"Writing metadata and job configuration files to {working_directory}")
        with open(str(working_directory / f"{job_id}_metadata.json"), "w") as fp:
            json.dump(asdict(self), fp, indent=2)
        logger.debug(f"Finished writing metadata and job configuration files to {working_directory}")

    @property
    def entrypoint(self):
        """The entrypoint to run for the training job."""
        return (
            f"""accelerate launch train_dreambooth_lora.py"""
            f""" --pretrained_model_name_or_path={self.model_name}"""
            f""" --instance_data_dir={self.instance_directory}"""
            f""" --output_dir={self.job_config.output_directory}"""
            f''' --instance_prompt="{self.instance_prompt}"'''
            f""" --resolution={self.resolution}"""
            f""" --train_batch_size=1"""
            f""" --gradient_accumulation_steps=1"""
            f""" --checkpointing_steps={self.max_train_steps // 5}"""
            f""" --learning_rate=1e-4"""
            f''' --lr_scheduler="constant"'''
            f""" --lr_warmup_steps=0"""
            f""" --max_train_steps={self.max_train_steps}"""
            f''' --seed="{self.seed}"'''
        )
        # f''' --validation_prompt="A photo of sks dog in a bucket"'''
        # f''' --validation_epochs=50'''

    def job_dict(self) -> Dict[str, Any]:
        """The job configuration for the Ray training job."""
        return {
            "submission_id": self.job_config.uuid,
            "runtime_env": self.job_config.runtime_env,
            "entrypoint": self.entrypoint,
            "entrypoint_num_gpus": 1,
        }
