import json
from typing import Any, Dict

from nos.exceptions import ModelNotFoundError
from nos.executors.ray import RayExecutor, RayJobExecutor
from nos.logging import logger
from nos.protoc import import_module

from .config import NoOpTrainingJobConfig
from .dreambooth.config import StableDiffusionDreamboothTrainingJobConfig, StableDiffusionXLDreamboothTrainingJobConfig
from .openmmlab.mmdetection.config import MMDetectionTrainingJobConfig


nos_service_pb2 = import_module("nos_service_pb2")
nos_service_pb2_grpc = import_module("nos_service_pb2_grpc")


def register_model(model_name: str, *args, **kwargs):
    import torch
    from PIL import Image

    from nos import hub
    from nos.common import Batch, ImageSpec, ImageT, TaskType
    from nos.models.dreambooth.dreambooth import StableDiffusionDreamboothHub, StableDiffusionLoRA

    # Update the registry with newer configs
    sd_hub = StableDiffusionDreamboothHub(namespace="custom")
    psize = len(sd_hub)
    sd_hub.update()
    logger.debug(f"Updated registry with newer configs [namespace=custom, size={len(sd_hub)}, prev_size={psize}]")

    # Register the model
    model_id = f"custom/{model_name}"
    logger.debug(f"Registering new model [model={model_id}]")
    hub.register(
        model_id,
        TaskType.IMAGE_GENERATION,
        StableDiffusionLoRA,
        init_args=(model_id,),
        init_kwargs={"dtype": torch.float16},
        method_name="__call__",
        inputs={"prompts": Batch[str, 1], "num_images": int, "height": int, "width": int},
        outputs={"images": Batch[ImageT[Image.Image, ImageSpec(shape=(None, None, 3), dtype="uint8")]]},
    )
    logger.debug(f"Registering new model [model={model_id}]")


class TrainingService:
    """Ray-executor based training service."""

    config_cls = {
        "nos/noop-train": NoOpTrainingJobConfig,
        "diffusers/stable-diffusion-dreambooth-lora": StableDiffusionDreamboothTrainingJobConfig,
        "diffusers/stable-diffusion-xl-dreambooth-lora": StableDiffusionXLDreamboothTrainingJobConfig,
        "open-mmlab/mmdetection": MMDetectionTrainingJobConfig,
    }

    def __init__(self):
        """Initialize the training service."""
        self.executor = RayExecutor.get()
        if not self.executor.is_initialized():
            raise RuntimeError("Ray executor is not initialized")

    def train(self, method: str, inputs: Dict[str, Any], metadata: Dict[str, Any] = None) -> str:
        """Train / Fine-tune a model by submitting a job to the RayJobExecutor.

            See https://docs.ray.io/en/latest/cluster/running-applications/job-submission/doc/ray.job_submission.JobSubmissionClient.submit_job.html#ray.job_submission.JobSubmissionClient.submit_job
            for more details on the job configuration.

        Args:
            method (str): Training method (e.g. `stable-diffusion-dreambooth-lora`).
            inputs (Dict[str, Any]): Training inputs.
            metadata (Dict[str, Any], optional): Metadata to attach to the training job. Defaults to None.
        Returns:
            str: Job ID.
        """
        # Check if the training method and inputs are correctly specified
        try:
            config_cls = self.config_cls[method]
            logger.debug(f"Using training config class [method={method}, config_cls={config_cls}]")
            config = config_cls(method=method, **inputs)
            logger.debug(f"Created training config [config={config}]")
        except KeyError:
            logger.error(f"Training not supported for method [method={method}]")
            raise ModelNotFoundError(f"Training not supported for method [method={method}]")
        except Exception as e:
            logger.error(f"Invalid training inputs [inputs={inputs}, e={e}]")
            raise ValueError(f"Invalid training inputs [inputs={inputs}, e={e}]")

        # Submit the training job as a Ray job
        configd = config.job_configuration()
        logger.debug("Submitting training job with configuration ... 👇")
        logger.debug(f"\n{json.dumps(configd, indent=2)}")
        job_id = self.executor.jobs.submit(**configd, metadata=metadata)
        logger.debug(f"Submitted training job [job_id={job_id}]")

        # hooks = {"on_completed": (register_model, (job_id,), {})}

        # # Spawn a thread to monitor the job
        # def monitor_job_hook(job_id: str, timeout: int = 600, retry_interval: int = 5):
        #     """Hook for monitoring the job status and running callbacks on completion."""
        #     st = time.time()
        #     while time.time() - st < timeout:
        #         status = self.executor.jobs.status(job_id)
        #         if str(status) == "SUCCEEDED":
        #             logger.debug(f"Training job completed [job_id={job_id}, status={status}]")
        #             cb, args, kwargs = hooks["on_completed"]
        #             logger.debug(f"Running callback [cb={cb}, args={args}, kwargs={kwargs}]")
        #             cb(*args, **kwargs)
        #             logger.debug(f"Callback completed [cb={cb}, args={args}, kwargs={kwargs}]")
        #             break
        #         else:
        #             logger.debug(f"Training job not completed yet [job_id={job_id}, status={status}]")
        #             time.sleep(retry_interval)

        # threading.Thread(target=monitor_job_hook, args=(job_id,), daemon=True).start()
        return job_id

    @property
    def jobs(self) -> RayJobExecutor:
        return self.executor.jobs
