import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict

from nos.common.git import cached_repo
from nos.constants import NOS_TMP_DIR
from nos.logging import logger


GIT_TAG = "v0.20.1"
RUNTIME_ENVS = {
    "diffusers-latest": {
        "working_dir": "./nos/experimental/train/dreambooth",
        "pip": [f"https://github.com/huggingface/diffusers/archive/refs/tags/{GIT_TAG}.zip", "accelerate>=0.22.0"],
    }
}


@dataclass
class StableDiffusionTrainingJobConfig:
    """Configuration for a training job.

    Training job contents are written to `~/.nos/tmp/{uuid}/`.
        {uuid}_metadata.json: Metadata for the training job.
        {uuid}_job_config.json: Job configuration for the training job.
    """

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

    runtime_env: Dict[str, str] = field(default_factory=lambda: RUNTIME_ENVS["diffusers-latest"])
    """The runtime environment to use for the training job."""

    _uuid: str = field(init=False, default_factory=lambda: str(uuid.uuid4()))
    """The UUID for creating a unique training job directory."""

    _output_directory: str = field(init=False)
    """The output directory for the training job."""

    _repo_directory: str = field(
        init=False,
        default=cached_repo(
            f"https://github.com/huggingface/diffusers/archive/refs/tags/{GIT_TAG}.zip",
            repo_name="diffusers",
            subdirectory="examples/dreambooth",
        ),
    )
    """The repository to use for the training job."""

    def __post_init__(self):
        if self.method not in ("stable-diffusion-dreambooth-lora"):
            raise ValueError(f"Invalid method: {self.method}, available methods: ['stable-diffusion-dreambooth-lora']")

        # Setup the instance and output directories
        logger.debug("Setting up instance and output directories")
        working_directory = Path(NOS_TMP_DIR / self._uuid)
        working_directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Finished setting up instance and output directories [working_directory={working_directory}]")

        # Copy the instance directory to the working directory
        if not Path(self.instance_directory).exists():
            raise IOError(f"Failed to load instance_directory={self.instance_directory}.")
        instance_directory = working_directory / "instances"
        shutil.copytree(self.instance_directory, str(instance_directory))
        nfiles = len(os.listdir(instance_directory))
        logger.debug(f"Copied instance directory to {working_directory} [nfiles={nfiles}]")

        # Create an output directory for weights
        output_directory = working_directory / "weights"
        output_directory.mkdir(parents=True, exist_ok=True)

        # Setup the diffusers working directory
        self.runtime_env["working_dir"] = self._repo_directory

        # Set the instance and output directories
        self.instance_directory = str(instance_directory)
        self._working_directory = str(working_directory)
        self._output_directory = str(output_directory)

        # Write the metadata and job configuration files
        logger.debug(f"Writing metadata and job configuration files to {working_directory}")
        with open(str(working_directory / f"{self._uuid}_metadata.json"), "w") as fp:
            json.dump(asdict(self), fp, indent=2)
        with open(str(working_directory / f"{self._uuid}_job_config.json"), "w") as fp:
            json.dump(asdict(self), fp, indent=2)
        logger.debug(f"Finished writing metadata and job configuration files to {working_directory}")

    @property
    def entrypoint(self):
        """The entrypoint to run for the training job."""
        return (
            f"""accelerate launch train_dreambooth_lora.py"""
            f""" --pretrained_model_name_or_path={self.model_name}"""
            f""" --instance_data_dir={self.instance_directory}"""
            f""" --output_dir={self._output_directory}"""
            f''' --instance_prompt="{self.instance_prompt}"'''
            f""" --resolution={self.resolution}"""
            f""" --train_batch_size=1"""
            f""" --gradient_accumulation_steps=1"""
            f""" --checkpointing_steps=100"""
            f""" --learning_rate=1e-4"""
            f''' --lr_scheduler="constant"'''
            f""" --lr_warmup_steps=0"""
            f""" --max_train_steps={self.max_train_steps}"""
            f''' --seed="0"'''
        )
        # f''' --validation_prompt="A photo of sks dog in a bucket"'''
        # f''' --validation_epochs=50'''

    def job_dict(self) -> Dict[str, Any]:
        """The job configuration for the training job."""
        return {
            "submission_id": self._uuid,
            "entrypoint": self.entrypoint,
            "runtime_env": self.runtime_env,
            "entrypoint_num_gpus": 1,
        }
