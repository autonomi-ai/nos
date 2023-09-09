from dataclasses import dataclass, field
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

    config_overrides: Dict[str, Any] = field(default_factory=dict)
    """Model config overrides as a dictionary."""

    runtime_env: RuntimeEnv = field(init=False, default_factory=lambda: RuntimeEnvironmentsHub.get(RUNTIME_ENV_NAME))
    """The runtime environment for the training job."""

    def __post_init__(self):
        # config_filename = Path(__file__).parent / Path(self.config_filename)
        # if not Path(self.config_filename).exists():
        #     raise IOError(f"Failed to load config [filename={self.config_filename}].")
        # logger.debug(f"{self.__class__.__name__} [uuid={self.uuid}, working_dir={self.working_directory}]")

        from mmengine.config import Config

        # Override the configuration with defaults for fine-tuning
        frozen_config = {
            "val_dataloader": None,
            "val_evaluator": None,
            "test_dataloader": None,
            "test_evaluator": None,
            "optim_wrapper": {
                "optimizer": {"type": "SGD", "lr": 0.01, "momentum": 0.9, "weight_decay": 0.0001},
            },
            "max_epochs": 2,
        }
        for k, v in self.config_overrides.items():
            if k in frozen_config:
                raise ValueError(f"Cannot override frozen key [key={k}, value={v}].")
            logger.debug(f"Overriding model config [key={k}, value={v}].")

        # Load the config file and merge with overrides
        cfg = Config.fromfile(self.config_filename)
        cfg.merge({**frozen_config, **self.config_overrides})
        logger.debug(f"Loaded config [filename={self.config_filename}, cfg={cfg}].")

        # Move the config file from the volume directory to the repo's configs directory
        # config_filename = NOS_VOLUME_DIR / self.config_filename
        # if not config_filename.exists():
        #     raise IOError(f"Failed to load config from volume [filename={config_filename}, volume={NOS_VOLUME_DIR}].")
        # config_dest = self.working_directory / Path(self.config_filename).name
        # shutil.copy(config_filename, config_dest)
        # self.config_filename = config_dest
        # logger.debug(f"Copied config file to working directory [filename={self.config_filename}].")

    @property
    def entrypoint(self):
        """The entrypoint to run for the training job."""
        entrypoint = """ls"""  # python tools/train.py {self.config_filename}"""
        logger.debug(
            f"Running training job [uuid={self.uuid}, working_dir={self.working_directory}, entrypoint={entrypoint}]."
        )
        return entrypoint

    def job_configuration(self) -> Dict[str, Any]:
        """The job configuration for the Ray training job."""
        return {
            **super().job_configuration(),
            "entrypoint_num_gpus": 0.5,
        }
