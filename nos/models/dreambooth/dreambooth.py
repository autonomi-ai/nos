from dataclasses import dataclass
from typing import List, Union

import torch
from PIL import Image


@dataclass(frozen=True)
class StableDiffusionLoRAConfig:
    """Stable Diffusion LoRA model configuration."""

    model_name: str
    """Model name (e.g `stabilityai/stable-diffusion-2-1`)."""
    attn_procs: str
    """Attention processors path."""
    width: int = 512
    """Image width."""
    height: int = 512
    """Image height."""
    dtype: torch.dtype = torch.float32
    """Data type (e.g. `torch.float32`)."""


class StableDiffusionLoRA:
    """Stable Diffusion LoRA model for DreamBooth."""

    configs = {
        StableDiffusionLoRAConfig(
            model_name="stabilityai/stable-diffusion-2-1",
            attn_procs="/home/spillai/software/huggingface/diffusers/examples/dreambooth/sdv21_dreambooth_fp32_lora_dog",
            dtype=torch.float32,
            width=768,
            height=768,
        ),
        StableDiffusionLoRAConfig(
            model_name="runwayml/stable-diffusion-v1-5",
            attn_procs="/home/spillai/software/huggingface/diffusers/examples/dreambooth/sdv15_dreambooth_fp32_lora_dog",
            dtype=torch.float32,
        ),
    }

    def __init__(self, model_name: str = "stabilityai/stable-diffusion-2-1", dtype: str = torch.dtype):
        from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

        try:
            self.cfg = StableDiffusionLoRA.configs[model_name]
        except KeyError:
            raise ValueError(
                f"Invalid model_name: {model_name}, available models: {StableDiffusionLoRA.configs.keys()}"
            )

        if torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = dtype
        else:
            self.device = "cpu"
            self.dtype = torch.float32

        self.pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=dtype)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(self.device)

        # Load attention processors one by one
        self.pipe.unet.load_attn_procs(self.cfg.attn_procs)

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
        return self.pipe(
            prompts * num_images,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height if height is not None else self.cfg.height,
            width=width if width is not None else self.cfg.width,
        ).images
