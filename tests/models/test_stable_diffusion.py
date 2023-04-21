from typing import List

import pytest

from nos.test.utils import PyTestGroup, benchmark, requires_torch_cuda


@pytest.fixture(scope="module")
def model():
    from nos.models import StableDiffusion2  # noqa: F401

    # TODO (spillai): @pytest.parametrize("scheduler", ["ddim", "euler-discrete"])
    yield StableDiffusion2(model_name="stabilityai/stable-diffusion-2", scheduler="ddim")


@requires_torch_cuda
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


@benchmark
@requires_torch_cuda
@pytest.mark.benchmark(group=PyTestGroup.BENCHMARK_MODELS)
def test_stable_diffusion_benchmark(model):
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
    print(f"StableDiffusion2: {time_ms / steps:.2f} ms / step")
