from typing import Any, Dict

from nos.exceptions import ModelNotFoundError
from nos.executors.ray import RayExecutor, RayJobExecutor
from nos.experimental.train.dreambooth.config import StableDiffusionTrainingJobConfig
from nos.logging import logger
from nos.protoc import import_module


nos_service_pb2 = import_module("nos_service_pb2")
nos_service_pb2_grpc = import_module("nos_service_pb2_grpc")


class TrainingService:
    """Ray-executor based training service."""

    config_cls = {
        "stable-diffusion-dreambooth-lora": StableDiffusionTrainingJobConfig,
    }

    def __init__(self):
        self.executor = RayExecutor.get()
        try:
            self.executor.init()
        except Exception as e:
            err_msg = f"Failed to initialize executor [e={e}]"
            logger.info(err_msg)
            raise RuntimeError(err_msg)

    def train(self, method: str, training_inputs: Dict[str, Any], metadata: Dict[str, Any] = None) -> str:
        """Train / Fine-tune a model by submitting a job to the RayJobExecutor.

        Args:
            method (str): Training method (e.g. `stable-diffusion-dreambooth-lora`).
            training_inputs (Dict[str, Any]): Training inputs.
        Returns:
            str: Job ID.
        """
        try:
            config_cls = self.config_cls[method]
        except KeyError:
            raise ModelNotFoundError(f"Training not supported for method [method={method}]")

        # Check if the training inputs are correctly specified
        config = config_cls(method=method, **training_inputs)
        try:
            pass
        except Exception as e:
            raise ValueError(f"Invalid training inputs [training_inputs={training_inputs}, e={e}]")

        # Submit the training job as a Ray job
        configd = config.job_dict()
        if metadata is not None:
            configd["metadata"] = metadata
        logger.debug("Submitting training job")
        logger.debug(f"config\n{configd}]")
        job_id = self.executor.jobs.submit(**configd)
        logger.debug(f"Submitted training job [job_id={job_id}, config={configd}]")
        return job_id

    @property
    def jobs(self) -> RayJobExecutor:
        return self.executor.jobs
