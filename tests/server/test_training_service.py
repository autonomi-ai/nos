import shutil
import tempfile
from pathlib import Path

import pytest

from nos.executors.ray import RayExecutor
from nos.logging import logger
from nos.server.train import TrainingService
from nos.test.conftest import ray_executor  # noqa: F401
from nos.test.utils import NOS_TEST_IMAGE

pytestmark = pytest.mark.server


def submit_dreambooth_lora_job(svc: TrainingService, method: str) -> str:  # noqa: F811
    """Submit a dreambooth LoRA job."""
    # Copy test image to temporary directory and test training service
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_image = Path(tmp_dir) / "test_image.jpg"
        shutil.copy(NOS_TEST_IMAGE, tmp_image)

        job_id = svc.train(
            method=method,
            inputs={
                "model_name": "stabilityai/stable-diffusion-2-1",
                "instance_directory": tmp_dir,
                "instance_prompt": "A photo of a bench on the moon",
                "resolution": 512,
                "max_train_steps": 10,
                "seed": 0,
            },
            overrides={},  # type: ignore
            metadata={
                "name": "sdv21-dreambooth-lora-test-bench",
            },
        )
        assert job_id is not None
        logger.debug(f"Submitted job with id: {job_id}")
    return job_id


def submit_mmdetection_job(svc: TrainingService, method: str) -> str:  # noqa: F811
    """Submit a mmdetection job."""
    from nos.server.train.openmmlab.mmdetection import config

    config_dir = Path(config.__file__).parent
    logger.debug(f"Config dir: {config_dir}")
    job_id = svc.train(
        method=method,
        inputs={
            "config_filename": "configs/yolox/yolox_s_8xb8-300e_coco.py",
        },
        overrides={
            "data_root": "data/coco/",
            "dataset_type": "HuggingfaceDataset",
        },  # type: ignore
        metadata={
            "name": "yolox-s-8xb8-300e-coco-ft",
        },
    )
    assert job_id is not None
    logger.debug(f"Submitted job with id: {job_id}")
    return job_id


# TRAINING_METHODS = list(TrainingService.config_cls.keys())
# TRAINING_METHODS = ["open-mmlab/mmdetection"]
TRAINING_METHODS = ["diffusers/stable-diffusion-dreambooth-lora"]

@pytest.mark.parametrize("method", TRAINING_METHODS)
def test_training_service(ray_executor: RayExecutor, method):  # noqa: F811
    """Test training service for dreambooth LoRA."""

    # Test training service
    svc = TrainingService()

    # Submit a job
    if method == "diffusers/stable-diffusion-dreambooth-lora":
        job_id = submit_dreambooth_lora_job(svc, method)
    elif method == "open-mmlab/mmdetection":
        job_id = submit_mmdetection_job(svc, method)
    else:
        raise ValueError(f"Invalid method: {method}")

    # Get logs for the job
    logs = svc.jobs.logs(job_id)
    assert logs is not None
    logger.debug(f"Logs for job {job_id}: {logs}")

    # Get info for the job
    info = svc.jobs.info(job_id)
    assert info is not None
    logger.debug(f"Info for job {job_id}: {info}")

    # Get status for the job
    status = svc.jobs.status(job_id)
    assert status is not None
    logger.debug(f"Status for job {job_id}: {status}")

    # Wait for the job to complete
    status = svc.jobs.wait(job_id, timeout=600, retry_interval=5)
    assert status is not None
    logger.debug(f"Status for job {job_id}: {status}")

    logs = svc.jobs.logs(job_id)
    assert status == "SUCCEEDED", f"Training job failed [job_id={job_id}, status={status}]\n{'-' * 80}\n{logs}"
