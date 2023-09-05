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
from nos.server.train.config import TrainingJobConfig


GIT_TAG = "v3.1.0"

NOS_VOLUME_DIR = NOS_HOME / "volumes"
RUNTIME_ENVS = {
    "mmdetection-latest": {
        "working_dir": None,
        "runtime_env": RuntimeEnv.from_packages(
            [f"https://github.com/open-mmlab/mmdetection/archive/refs/tags/{GIT_TAG}.zip"]
        ),
    },
}


@dataclass
class MMDetectionTrainingJobConfig(TrainingJobConfig):
    """Configuration for open-mmlab/mmdetection training job."""

    config_filename: str
    """Model config filename (e.g `configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py`)."""

    repo_directory: str = field(
        init=False,
        default=cached_repo(
            f"https://github.com/open-mmlab/mmdetection/archive/refs/tags/{GIT_TAG}.zip",
            repo_name="mmdetection",
            subdirectory="tools",
        ),
    )
    """The repository to use for the training job."""

    def __post_init__(self):
        # if self.method not in MMDetectionTrainingJobConfig:
        #     raise ValueError(
        #         f"Invalid method: {self.method}, available methods: [{','.join(k for k in StableDiffusionDreamboothConfigs)}]"
        #     )

        # Setup the working directories (output, repo)
        runtime_env = RUNTIME_ENVS["mmdetection-latest"].copy()
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
        return f"""python tools/train.py {self.config_filename}"""

    def job_dict(self) -> Dict[str, Any]:
        """The job configuration for the Ray training job."""
        return {
            "submission_id": self.job_config.uuid,
            "runtime_env": self.job_config.runtime_env,
            "entrypoint": self.entrypoint,
            "entrypoint_num_gpus": 1,
        }
