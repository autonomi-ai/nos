from typing import List

import pytest

from nos.test.utils import PyTestGroup, skip_all_if_no_torch_cuda


pytestmark = pytest.mark.skip("TODO: Pending fixes to ControlNet")  # skip_all_if_no_torch_cuda()


MODEL_NAME = "runwayml/stable-diffusion-v1-5"


@pytest.fixture(scope="module")
def model():
    from nos.models import StableDiffusion  # noqa: F401

    # TODO (spillai): @pytest.parametrize("scheduler", ["ddim", "euler-discrete"])
    yield StableDiffusion(model_name=MODEL_NAME, scheduler="ddim")


def test_controlnet(model):
    """Use StableDiffusion to generate an image from a text prompt, and the modify it with ControlNet."""
    import torch
    from controlnet_aux.processor import Processor
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
    from PIL import Image

    # Generate an image from a text prompt
    images: List[Image.Image] = model.__call__(
        "astronaut on a horse on the moon",
        num_images=1,
        num_inference_steps=10,
        guidance_scale=7.5,
        width=512,
        height=512,
    )
    (image,) = images
    assert image is not None
    assert image.size == (512, 512)

    # CompVis/stable-diffusion-v1-4, runwayml/stable-diffusion-v1-5, stabilityai/stable-diffusion-2-1
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        MODEL_NAME, controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    # pipe.enable_xformers_memory_efficient_attention()

    # Generate depth map from generated image
    # Note: This needs to correspond to the controlnet model used in the pipeline
    depth_proc = Processor("depth_midas")
    control_img = depth_proc(image, to_pil=True)

    # Generate new images from text prompts
    prompt = ", best quality, extremely detailed"
    prompt = [t + prompt for t in ["Sandra Oh", "beyonce", "oprah", "michelle obama"]]
    generator = [torch.Generator(device="cpu").manual_seed(2) for i in range(len(prompt))]
    output = pipe(
        prompt,
        control_img,
        negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] * 4,
        num_inference_steps=20,
        generator=generator,
        width=512,
        height=512,
    )
    assert len(output.images) == len(prompt)
    assert isinstance(output.images[0], Image.Image)
    assert output.images[0].size == (512, 512)


@pytest.mark.parametrize("task", ["depth_midas", "canny", "openpose"])
def test_controlnet_control_tasks(task):
    """Test various control tasks."""
    from controlnet_aux.processor import Processor
    from PIL import Image

    from nos.test.utils import NOS_TEST_IMAGE

    img = Image.open(NOS_TEST_IMAGE)
    img = img.resize((512, 512))

    proc = Processor(task)
    img = proc(img, to_pil=True)
    assert img is not None
    assert isinstance(img, Image.Image)


@pytest.mark.skip("TODO: Fix this test")
@pytest.mark.benchmark(group=PyTestGroup.MODEL_BENCHMARK)
def test_controlnet_benchmark(model):
    """Benchmark ControlNet model."""
    raise NotImplementedError()
