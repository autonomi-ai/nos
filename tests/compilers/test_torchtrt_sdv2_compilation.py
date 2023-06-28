"""Test and benchmarking SDv2 compilation with TorchTRT.

On NVIDIA 4090:
 - ["stabilityai/stable-diffusion-2-1"] Eager: 200it [00:-, 10.3it/s]
 - ["stabilityai/stable-diffusion-2-1"]   TRT: 200it [00:-, xx.xit/s]
"""
import os

import pytest

from nos.logging import logger
from nos.test.utils import PyTestGroup


env = os.environ.get("NOS_ENV", os.getenv("CONDA_DEFAULT_ENV", "base_gpu"))
logger.info(f"Using env: {env}")
pytestmark = pytest.mark.skipif(
    env not in ("nos_trt_dev", "nos_trt_runtime"),
    reason=f"Requires nos env [nos_trt_dev, nos_trt_runtime], but using {env}",
)


@pytest.mark.benchmark(group=PyTestGroup.MODEL_COMPILATION)
def test_sdv2_torchtrt_compilation():
    """Test and benchmark compilation of SDv2 with TorchTRT."""
    from PIL import Image

    from nos.models.stable_diffusion import StableDiffusionTensorRT

    sd = StableDiffusionTensorRT(model_name="stabilityai/stable-diffusion-2-1", scheduler="ddim")
    images = sd(prompts=["fox jumped over dog"], num_inference_steps=10, num_images=1)  # noqa: B018
    assert len(images) == 1
    assert isinstance(images[0], Image.Image)
    assert images[0].size == (512, 512)
