import shutil
import tempfile
from pathlib import Path

import pytest

from nos.executors.ray import RayExecutor
from nos.logging import logger
from nos.test.conftest import ray_executor  # noqa: F401
from nos.test.utils import NOS_TEST_IMAGE


pytestmark = pytest.mark.server


def test_training_service(ray_executor: RayExecutor):  # noqa: F811
    """Test training service."""
    from nos.server.train import TrainingService

    # Test training service
    svc = TrainingService()

    # Copy test image to temporary directory and test training service
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_image = Path(tmp_dir) / "test_image.jpg"
        shutil.copy(NOS_TEST_IMAGE, tmp_image)

        job_id = svc.train(
            method="stable-diffusion-dreambooth-lora",
            training_inputs={
                "model_name": "stabilityai/stable-diffusion-2-1",
                "instance_directory": tmp_dir,
                "instance_prompt": "A photo of a bench on the moon",
                "resolution": 512,
                "max_train_steps": 10,
                "seed": 0,
            },
            metadata={
                "name": "sdv21-dreambooth-lora-test-bench",
            },
        )
        assert job_id is not None
        logger.debug(f"Submitted job with id: {job_id}")

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


def test_inference_service_with_trained_model(ray_executor: RayExecutor):  # noqa: F811
    """Test inference service."""
    from nos.server._service import InferenceService

    # Test training service
    InferenceService()
    # job_id = svc.execute(
    #     method="stable-diffusion-dreambooth-lora",
    #     inference_inputs={
    #         "model_name": "stabilityai/stable-diffusion-2-1",
    #         "instance_directory": tmp_dir,
    #         "instance_prompt": "A photo of a bench on the moon",
    #         "resolution": 512,
    #         "max_train_steps": 100,
    #         "seed": 0,
    #     },
    #     metadata={
    #         "name": "sdv21-dreambooth-lora-test-bench",
    #     },
    # )
    # assert job_id is not None
    # logger.debug(f"Submitted job with id: {job_id}")
