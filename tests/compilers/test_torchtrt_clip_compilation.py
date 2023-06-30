"""Test and benchmarking CLIP compilation with TorchTRT.

On NVIDIA 4090 (fp32):
 - torch-eager | openai/clip-vit-base-patch32: 2000it [00:20,  99 it/s]
 -   torch-trt | openai/clip-vit-base-patch32: 3260it [00:20, 162 it/s]
"""
import os

import numpy as np
import pytest

from nos.common import tqdm
from nos.logging import logger
from nos.test.utils import NOS_TEST_IMAGE, PyTestGroup


env = os.environ.get("NOS_ENV", os.getenv("CONDA_DEFAULT_ENV", "base_gpu"))
logger.info(f"Using env: {env}")
pytestmark = pytest.mark.skipif(
    env not in ("nos_trt_dev", "nos_trt_runtime"),
    reason=f"Requires nos env [nos_trt_dev, nos_trt_runtime], but using {env}",
)


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("model_name", ["openai/clip-vit-base-patch32"])
@pytest.mark.benchmark(group=PyTestGroup.MODEL_COMPILATION)
@pytest.mark.benchmark(group=PyTestGroup.MODEL_BENCHMARK)
def test_clip_torchtrt_compilation(model_name, batch_size):
    """Test and benchmark compilation of CLIP with TorchTRT."""
    from PIL import Image

    from nos.models.clip import CLIP, CLIPTensorRT

    im = Image.open(NOS_TEST_IMAGE)
    im = im.resize((224, 224))

    # Load the accelerated model
    model = CLIPTensorRT(model_name=model_name)

    # First run will trigger a compilation (if not already cached)
    embed_im = model.encode_image([im for _ in range(batch_size)])
    assert embed_im.shape == (batch_size, model.cfg.D)

    # Get the reference embedding
    model_ref = CLIP(model_name=model_name)
    embed_im_ref = model_ref.encode_image(im)
    del model_ref

    # Compare outputs (dot product should be close to 1)
    embed_im /= np.linalg.norm(embed_im, axis=-1, keepdims=True)
    embed_im_ref /= np.linalg.norm(embed_im_ref, axis=-1, keepdims=True)
    aligned = (embed_im * embed_im_ref).sum()
    assert aligned > 0.99

    # Attempt running with a different image size
    # Note (spillai): This will resize the image to the original compiled size (224, 224)
    _ = model.encode_image([im.resize((320, 240))])

    # Subsequent runs will use the cached compilation
    logger.debug(f"Benchmarking {model_name}")
    logger.debug("Warming up (5s) ...")
    for _ in tqdm(duration=5.0, disable=True):
        model.encode_image([im])
    logger.debug("Running benchmark (20s) ...")
    for _ in tqdm(duration=20.0, desc=f"torch-trt | {model_name}"):
        model.encode_image([im])
