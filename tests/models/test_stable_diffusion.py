from typing import List

import pytest

from nos.models.stable_diffusion import StableDiffusion
from nos.test.utils import PyTestGroup, skip_if_no_torch_cuda


STABLE_DIFFUSION_MODELS = StableDiffusion.configs.keys()


@pytest.mark.benchmark(group=PyTestGroup.HUB)
@pytest.mark.parametrize("model", STABLE_DIFFUSION_MODELS)
def test_stable_diffusion_predict(model):
    """Use StableDiffusion to generate an image from a text prompt.

    Note: This test should be able to run with CPU or GPU.
    CPU: 3.6s/it
    GPU (2080 Ti): 8.5it/s
    GPU (4090): 9.8it/s
    """
    from PIL import Image

    model = StableDiffusion(model_name=model, scheduler="ddim")
    images: List[Image.Image] = model.__call__(
        ["astronaut on a horse on the moon"] * 2,
        negative_prompts=["negative"] * 2,
        num_images=1,
        num_inference_steps=100,
        guidance_scale=7.5,
        width=512,
        height=512,
    )
    image = images[0]
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
@pytest.mark.benchmark(group=PyTestGroup.MODEL_BENCHMARK)
@pytest.mark.parametrize("model", STABLE_DIFFUSION_MODELS)
def test_stable_diffusion_benchmark(model):
    """Benchmark StableDiffusion model."""
    from nos.test.benchmark import run_benchmark

    model = StableDiffusion(model_name=model, scheduler="ddim")

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
    print(f"BENCHMARK [{model}]: {time_ms / steps:.2f} ms / step")
