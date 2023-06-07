from typing import List

import pytest

from nos.test.utils import PyTestGroup, skip_if_no_torch_cuda


MODEL_NAME = "runwayml/stable-diffusion-v1-5"


@pytest.fixture(scope="module")
def model():
    from nos.models import StableDiffusion  # noqa: F401

    # TODO (spillai): @pytest.parametrize("scheduler", ["ddim", "euler-discrete"])
    yield StableDiffusion(model_name=MODEL_NAME, scheduler="ddim")


@pytest.mark.benchmark(group=PyTestGroup.HUB)
def test_stable_diffusion(model):
    """Use StableDiffusion to generate an image from a text prompt.

    Note: This test should be able to run with CPU or GPU.
    CPU: 3.6s/it
    GPU (2080 Ti): 8.5it/s
    GPU (4090): 9.8it/s
    """
    from PIL import Image

    images: List[Image.Image] = model.__call__(
        "astronaut on a horse on the moon",
        num_images=1,
        num_inference_steps=100,
        guidance_scale=7.5,
        width=512,
        height=512,
    )
    (image,) = images
    assert image is not None
    assert image.size == (512, 512)


@pytest.mark.benchmark(group=PyTestGroup.HUB)
def test_stable_diffusion_download_all():
    """Download all StableDiffusion models from HuggingFace Hub."""
    import torch
    from diffusers import StableDiffusionPipeline

    from nos.models import StableDiffusion  # noqa: F401

    for _, config in StableDiffusion.configs.items():
        StableDiffusionPipeline.from_pretrained(config.model_name, torch_dtype=torch.float16)


@skip_if_no_torch_cuda
@pytest.mark.benchmark(group=PyTestGroup.BENCHMARK_MODELS)
def test_stable_diffusion_benchmark(model):
    """Benchmark StableDiffusion model."""
    from nos.test.benchmark import run_benchmark

    steps = 10
    time_ms = run_benchmark(
        lambda: model.__call__(
            "astronaut on a horse on the moon",
            num_images=1,
            num_inference_steps=steps,
            guidance_scale=7.5,
        ),
        num_iters=5,
    )
    print(f"BENCHMARK [{MODEL_NAME}]: {time_ms / steps:.2f} ms / step")
