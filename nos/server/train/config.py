import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict

from nos.common.spec import RuntimeEnv
from nos.constants import NOS_MODELS_DIR
from nos.logging import logger


RUNTIME_ENVS = {
    "diffusers-latest": {
        "working_dir": "./nos/experimental/",
        "runtime_env": RuntimeEnv.from_packages(
            ["https://github.com/huggingface/diffusers/archive/refs/tags/v0.20.1.zip", "accelerate>=0.22.0"]
        ),
    },
    "mmdetection-latest": {
        "working_dir": "./nos/experimental/",
        "runtime_env": RuntimeEnv.from_packages(
            ["https://github.com/open-mmlab/mmdetection/archive/refs/tags/v3.1.0.zip"]
        ),
    },
}

NOS_CUSTOM_MODELS_DIR = NOS_MODELS_DIR / "custom"


@dataclass
class TrainingJobConfig:
    """Generic configuration for a training job.

    Training job contents are written to `~/.nos/tmp/{uuid}/`.
        {uuid}_metadata.json: Metadata for the training job.
        {uuid}_job_config.json: Job configuration for the training job.
    """

    runtime_env: Dict[str, str]
    """The runtime environment to use for the training job."""

    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    """The UUID for creating a unique training job directory."""

    output_directory: str = field(init=False, default=None)
    """The output directory for the training job."""

    working_directory: str = field(default=NOS_CUSTOM_MODELS_DIR)
    """The working directory for the training job."""

    def __post_init__(self):
        # Setup the instance and output directories
        logger.debug("Setting up instance and output directories")
        working_directory = Path(self.working_directory / self.uuid)
        working_directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Finished setting up instance and output directories [working_directory={working_directory}]")

        # Create an output directory for weights
        output_directory = working_directory / "weights"
        output_directory.mkdir(parents=True, exist_ok=True)

        # Set the instance and output directories
        self.working_directory = str(working_directory)
        self.output_directory = str(output_directory)

        # Write the metadata and job configuration files
        logger.debug(f"Writing metadata and job configuration files to {working_directory}")
        with open(str(working_directory / f"{self.uuid}_job_config.json"), "w") as fp:
            json.dump(asdict(self), fp, indent=2)
        logger.debug(f"Finished writing metadata and job configuration files to {working_directory}")
