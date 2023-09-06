from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from nos.common.git import cached_repo
from nos.common.runtime_env import RuntimeEnv, RuntimeEnvironmentsHub
from nos.constants import NOS_HOME
from nos.logging import logger
from nos.server.train.config import TrainingJobConfig


# Register the runtime environment for fine-tuning open-mmlab/mmdetection models
GIT_TAG = "v3.1.0"
RUNTIME_ENV_NAME = "open-mmlab/mmdetection-latest"
RuntimeEnvironmentsHub.register(
    RUNTIME_ENV_NAME,
    RuntimeEnv(
        conda="mmdet-cu118",
        working_dir=cached_repo(
            f"https://github.com/open-mmlab/mmdetection/archive/refs/tags/{GIT_TAG}.zip", repo_name="mmdetection"
        ),
    ),
)
NOS_VOLUME_DIR = NOS_HOME / "volumes"


@dataclass
class MMDetectionTrainingJobConfig(TrainingJobConfig):
    """Configuration for open-mmlab/mmdetection training job."""

    config_filename: str = None
    """Model config filename (e.g `configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py`)."""

    runtime_env: RuntimeEnv = field(init=False, default_factory=lambda: RuntimeEnvironmentsHub.get(RUNTIME_ENV_NAME))
    """The runtime environment for the training job."""

    def __post_init__(self):
        if not Path(self.config_filename).exists():
            raise IOError(f"Failed to load config [filename={self.config_filename}].")
        logger.debug(f"{self.__class__.__name__} [uuid={self.uuid}, working_dir={self.working_directory}]")

    @property
    def entrypoint(self):
        """The entrypoint to run for the training job."""
        return f"""python tools/train.py {self.config_filename}"""

    def job_configuration(self) -> Dict[str, Any]:
        """The job configuration for the Ray training job."""
        return {
            **super().job_configuration(),
            "entrypoint_num_gpus": 0.5,
        }
