from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from nos.common.runtime_env import RuntimeEnv, RuntimeEnvironmentsHub
from nos.constants import NOS_HOME
from nos.logging import logger
from nos.server.train.config import TrainingJobConfig


# Register the runtime environment for fine-tuning open-mmlab/mmdetection models
RUNTIME_ENV_NAME = "mmdet-gpu"
WORKING_DIR = "/app/mmdetection"

RuntimeEnvironmentsHub.register(
    RUNTIME_ENV_NAME,
    RuntimeEnv(
        runtime=RUNTIME_ENV_NAME,
        working_dir=WORKING_DIR,
    ),
)

NOS_VOLUME_DIR = NOS_HOME / "volumes"
NOS_VOLUME_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class MMDetectionTrainingJobConfig(TrainingJobConfig):
    """Configuration for open-mmlab/mmdetection training job."""

    config_filename: str = None
    """Model config filename (e.g `configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py`)."""

    config_overrides: Dict[str, Any] = field(default_factory=dict)  # noqa: E128
    """Model config overrides as a dictionary."""

    runtime_env: str = field(default_factory=lambda: RuntimeEnvironmentsHub.get(RUNTIME_ENV_NAME))
    """The runtime environment to use for the training job."""

    def __post_init__(self):
        from mmengine.config import Config

        # Load the config file
        config_filename = Path(WORKING_DIR) / self.config_filename
        if not Path(config_filename).exists():
            raise IOError(f"Failed to load config [filename={config_filename}].")
        logger.debug(f"{self.__class__.__name__} [uuid={self.uuid}, output_dir={self.output_directory}]")

        # Override the configuration with defaults for fine-tuning
        # turn off black formatting for this section
        # fmt: off
        frozen_config = {
            "val_dataloader": None,
            "val_evaluator": None,
            "val_cfg": None,
            "test_cfg": None,
            "test_dataloader": None,
            "test_evaluator": None,
            "optim_wrapper": {
                "optimizer": {"type": "SGD", "lr": 0.01, "momentum": 0.9, "weight_decay": 0.0001},
            },
            "max_epochs": 2,
        }
        # fmt: on
        for k, v in self.config_overrides.items():
            if k in frozen_config:
                raise ValueError(f"Cannot override frozen key [key={k}, value={v}].")
            logger.debug(f"Overriding model config [key={k}, value={v}].")

        # Merge with overrides and save the updated config file
        cfg = Config.fromfile(config_filename)
        cfg.merge({**frozen_config, **self.config_overrides})
        logger.debug(f"Loaded config [filename={config_filename}, cfg={cfg}].")

        # Save the updated config file to the working directory
        config_dest = config_filename.parent / f"{config_filename.name}_ft.py"
        logger.debug(f"Writing updated config [filename={config_dest}].")
        cfg.dump(str(config_dest))
        self.config_filename = config_dest
        logger.debug(f"Saved updated config [filename={self.config_filename}].")

    @property
    def entrypoint(self):
        """The entrypoint to run for the training job."""
        entrypoint = f"""cd {WORKING_DIR} && python tools/train.py --help"""
        logger.debug(
            f"Running training job [uuid={self.uuid}, output_dir={self.output_directory}, entrypoint={entrypoint}]."
        )
        return entrypoint

    def job_configuration(self) -> Dict[str, Any]:
        """The job configuration for the Ray training job."""
        return {
            **super().job_configuration(),
            "entrypoint_num_gpus": 0.5,
        }
