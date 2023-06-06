from typing import List

import pytest

from nos.models import StableDiffusion2  # noqa: F401
from nos.test.utils import PyTestGroup, skip_all_if_no_torch_cuda


pytestmark = skip_all_if_no_torch_cuda()


@pytest.fixture(scope="module")
def model():
    # TODO (spillai): @pytest.parametrize("scheduler", ["ddim", "euler-discrete"])
    MODEL_NAME = "stabilityai/stable-diffusion-2"
    yield StableDiffusion2(model_name=MODEL_NAME, scheduler="ddim")


def test_stable_diffusion(model):
    from PIL import Image

    images: List[Image.Image] = model.__call__(
        "astronaut on a horse on the moon",
        num_images=1,
        num_inference_steps=10,
        guidance_scale=7.5,
    )
    (image,) = images
    assert image is not None
    assert image.size == (768, 768)


@pytest.mark.benchmark(group=PyTestGroup.BENCHMARK_MODELS)
def test_stable_diffusion_benchmark(model):
    """Benchmark StableDiffusion2 model."""
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
    print(f"BENCHMARK [StableDiffusion2]: {time_ms / steps:.2f} ms / step")
