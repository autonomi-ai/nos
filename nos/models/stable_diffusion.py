from typing import List, Union

import torch
from PIL import Image

from nos import hub
from nos.common import TaskType
from nos.common.types import Batch, ImageSpec, ImageT


class StableDiffusion2:
    """StableDiffusion model for text to image generation."""

    def __init__(
        self,
        model_name: str = "stabilityai/stable-diffusion-2",
        scheduler: str = "ddim",
        dtype: torch.dtype = torch.float16,
    ):
        from diffusers import (
            DDIMScheduler,
            EulerDiscreteScheduler,
            StableDiffusionPipeline,
        )

        # Only run on CUDA if available
        if torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = dtype
            torch.backends.cuda.matmul.allow_tf32 = True
        else:
            raise RuntimeError("No CUDA device available")

        if scheduler == "ddim":
            self.scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
        elif scheduler == "euler-discrete":
            self.scheduler = EulerDiscreteScheduler.from_pretrained(model_name, subfolder="scheduler")
        else:
            raise ValueError(f"Unknown scheduler: {scheduler}, choose from: ['ddim', 'euler-discrete']")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            scheduler=self.scheduler,
            torch_dtype=self.dtype,
            revision="fp16",
        )
        self.pipe = self.pipe.to(self.device)
        self.pipe.enable_attention_slicing()

        # TODO (spillai): Pending xformers integration
        # self.pipe.enable_xformers_memory_efficient_attention()

    def __call__(
        self,
        prompts: Union[str, List[str]],
        num_images: int = 1,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        height: int = None,
        width: int = None,
    ) -> List[Image.Image]:
        """Generate images from text prompt."""
        if isinstance(prompts, str):
            prompts = [prompts]
        with torch.inference_mode():
            with torch.autocast("cuda"):
                images = self.pipe(
                    prompts * num_images,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                ).images
                return images


# Register the model with the hub
# TODO (spillai): Ideally, we should do this via a decorator
# @hub.register("stabilityai/stable-diffusion-2")
# def stable_diffusion_2_ddim_fp16():
#     return StableDiffusion2("stabilityai/stable-diffusion-2", scheduler="ddim", dtype=torch.float16)
#
hub.register(
    "stabilityai/stable-diffusion-2",
    TaskType.IMAGE_GENERATION,
    StableDiffusion2,
    init_args=("stabilityai/stable-diffusion-2",),
    init_kwargs={"scheduler": "ddim", "dtype": torch.float16},
    method_name="__call__",
    inputs={"prompts": Batch[str], "num_images": int, "height": int, "width": int},
    outputs={"images": Batch[ImageT[Image.Image, ImageSpec(shape=(None, None, 3), dtype="uint8")]]},
)
