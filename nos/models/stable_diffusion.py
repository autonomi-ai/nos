from dataclasses import dataclass
from typing import List, Union

import torch
from PIL import Image

from nos import hub
from nos.common import TaskType
from nos.common.types import Batch, ImageSpec, ImageT
from nos.hub import HuggingFaceHubConfig


@dataclass(frozen=True)
class StableDiffusionConfig(HuggingFaceHubConfig):
    pass


class StableDiffusion:
    """StableDiffusion model for text to image generation."""

    configs = {
        "CompVis/stable-diffusion-v1-4": StableDiffusionConfig(
            model_name="CompVis/stable-diffusion-v1-4",
        ),
        "runwayml/stable-diffusion-v1-5": StableDiffusionConfig(
            model_name="runwayml/stable-diffusion-v1-5",
        ),
        "stabilityai/stable-diffusion-2": StableDiffusionConfig(
            model_name="stabilityai/stable-diffusion-2",
        ),
        "stabilityai/stable-diffusion-2-1": StableDiffusionConfig(
            model_name="stabilityai/stable-diffusion-2-1",
        ),
        "stabilityai/stable-diffusion-xl-base-1-0": StableDiffusionConfig(
            model_name="stabilityai/stable-diffusion-xl-base-1.0",
        ),
    }

    def __init__(
        self,
        model_name: str = "stabilityai/stable-diffusion-2",
        scheduler: str = "ddim",
        dtype: torch.dtype = torch.float16,
    ):
        from diffusers import (
            DDIMScheduler,
            DiffusionPipeline,
            EulerDiscreteScheduler,
        )

        try:
            self.cfg = StableDiffusion.configs[model_name]
        except KeyError:
            raise ValueError(f"Invalid model_name: {model_name}, available models: {StableDiffusion.configs.keys()}")

        # Only run on CUDA if available
        if torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = dtype
            self.revision = "fp16"
            # TODO (spillai): Investigate if this has any effect on performance
            # tf32 vs fp32: https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/279
            torch.backends.cuda.matmul.allow_tf32 = True
        else:
            self.device = "cpu"
            self.dtype = torch.float32
            self.revision = None

        if scheduler == "ddim":
            self.scheduler = DDIMScheduler.from_pretrained(self.cfg.model_name, subfolder="scheduler")
        elif scheduler == "euler-discrete":
            self.scheduler = EulerDiscreteScheduler.from_pretrained(self.cfg.model_name, subfolder="scheduler")
        else:
            raise ValueError(f"Unknown scheduler: {scheduler}, choose from: ['ddim', 'euler-discrete']")

        self.pipe = DiffusionPipeline.from_pretrained(
            self.cfg.model_name,
            scheduler=self.scheduler,
            torch_dtype=self.dtype,
        )
        self.pipe = self.pipe.to(self.device)

        # TODO (spillai): Pending xformers integration
        # self.pipe.enable_attention_slicing()
        # self.pipe.enable_xformers_memory_efficient_attention()

    def __call__(
        self,
        prompts: Union[str, List[str]],
        negative_prompts: Union[str, List[str]] = None,
        num_images: int = 1,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        height: int = None,
        width: int = None,
        seed: int = -1,
    ) -> List[Image.Image]:
        """Generate images from text prompt."""
        # Input validation and defaults
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]
        if isinstance(negative_prompts, list):
            negative_prompts *= num_images

        # Generate images with the appropriate seed
        g = torch.Generator(device=self.device)
        if seed != -1:
            g.manual_seed(seed)
        else:
            g.seed()

        # TODO (spillai): Pending xformers integration
        return self.pipe(
            prompts * num_images,
            negative_prompt=negative_prompts,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=g,
        ).images


# Register the model with the hub
for model_name in StableDiffusion.configs.keys():
    hub.register(
        model_name,
        TaskType.IMAGE_GENERATION,
        StableDiffusion,
        init_args=(model_name,),
        init_kwargs={"scheduler": "ddim", "dtype": torch.float16},
        method="__call__",
        inputs={
            "prompts": Batch[str, 1],
            "negative_prompts": Batch[str, 1],
            "num_images": int,
            "num_inference_steps": int,
            "guidance_scale": float,
            "height": int,
            "width": int,
            "seed": int,
        },
        outputs={"images": Batch[ImageT[Image.Image, ImageSpec(shape=(None, None, 3), dtype="uint8")]]},
    )
