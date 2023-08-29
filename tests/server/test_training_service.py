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
    from nos.server._service import TrainingService

    # Copy test image to temporary directory and test training service
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_image = Path(tmp_dir) / "test_image.jpg"
        shutil.copy(NOS_TEST_IMAGE, tmp_image)

        # Test training service
        svc = TrainingService()
        job_id = svc.train(
            method="stable-diffusion-dreambooth-lora",
            training_inputs={
                "model_name": "stabilityai/stable-diffusion-2-1",
                "instance_directory": tmp_dir,
                "instance_prompt": "A photo of sks dog in a bucket",
                "resolution": 512,
            },
            metadata={
                "name": "sdv21-dreambooth-lora-test-bench",
            },
        )
        assert job_id is not None
        logger.debug(f"Submitted job with id: {job_id}")
