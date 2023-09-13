import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict

from nos.common.spec import RuntimeEnv
from nos.constants import NOS_CACHE_DIR
from nos.logging import logger


NOS_JOBS_DIR = NOS_CACHE_DIR / "jobs"
NOS_WORKING_DIR = NOS_CACHE_DIR / "working_dirs"

NOS_JOBS_DIR.mkdir(parents=True, exist_ok=True)
NOS_WORKING_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainingJobConfig:
    """Generic configuration for a training job.

    Training job contents are written to `~/.nos/cache/jobs/<uuid>/`.
        cache/jobs/<uuid>:
            weights/: Output directory for weights.
            weights/<epoch>.pth: Weights for each epoch.
            <uuid>_config.json: Configuration for the training job.
    """

    method: str = field(default="nos/custom")
    """Training method (e.g. `diffusers/stable-diffusion-dreambooth-lora` etc)."""

    runtime_env: RuntimeEnv = field(init=False, default_factory=lambda: RuntimeEnv())
    """The runtime environment to use for the training job."""

    uuid: str = field(default_factory=lambda: str(uuid.uuid4().hex[:8]))
    """The UUID for creating a unique training job directory."""
    """Note, this is typically overriden by the subclass."""

    working_directory: str = field(default=NOS_JOBS_DIR)
    """The working directory for the training job."""

    metadata: Dict[str, Any] = field(default=None)
    """Metadata for the training job."""

    def __post_init__(self):
        logger.debug("Set up working directories")
        working_directory = Path(self.working_directory) / f"{self.method}_{self.uuid}"
        working_directory.mkdir(parents=True, exist_ok=True)
        self.working_directory = str(working_directory)
        logger.debug(f"Finished setting up working directories [working_dir={working_directory}]")

    def save(self):
        """Save the training job configuration."""
        logger.debug(f"Writing configuration files [working_dir={self.working_directory}]")
        with open(str(Path(self.working_directory) / f"{self.uuid}_config.json"), "w") as fp:
            json.dump(asdict(self), fp, indent=2)
        logger.debug(f"Finished writing configuration files [working_dir={self.working_directory}]")

    @property
    def entrypoint(self):
        """The entrypoint to run for the training job."""
        raise NotImplementedError()

    @property
    def weights_directory(self) -> str:
        """The weights / output directory for the training job."""
        weights_directory = Path(self.working_directory) / "weights"
        weights_directory.mkdir(parents=True, exist_ok=True)
        return str(weights_directory)

    def job_configuration(self) -> Dict[str, Any]:
        """The job configuration for the Ray training job.

        See ray.job_submission.JobSubmissionClient.submit_job() for more details:
            https://docs.ray.io/en/latest/cluster/running-applications/job-submission/doc/ray.job_submission.JobSubmissionClient.submit_job.html#ray.job_submission.JobSubmissionClient.submit_job
        """
        return {
            "entrypoint": self.entrypoint,
            "submission_id": self.uuid,
            "runtime_env": self.runtime_env.asdict(),
        }


@dataclass
class NoOpTrainingJobConfig(TrainingJobConfig):
    """Configuration for a no-op training job."""

    config: Dict[str, Any] = field(default_factory=dict)
    """No-op configuration."""

    def __post_init__(self):
        super().__post_init__()
        logger.debug(
            f"{self.__class__.__name__} [uuid={self.uuid}, config={self.config}, working_dir={self.working_directory}]"
        )

    @property
    def entrypoint(self):
        """The entrypoint to run for the training job."""
        return f"cd {self.working_directory} && echo 'No-op training job'"
