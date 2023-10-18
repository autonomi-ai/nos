from pathlib import Path
from typing import List, Union

import torch
from PIL import Image

from nos.logging import logger

from .hub import StableDiffusionDreamboothHub, StableDiffusionDreamboothLoRAConfig


class StableDiffusionLoRA:
    """Stable Diffusion LoRA model for DreamBooth."""

    configs = StableDiffusionDreamboothHub(namespace="custom").configs

    def __init__(
        self,
        model_name: str = "stabilityai/stable-diffusion-2-1",
        weights_dir=None,
        dtype: torch.dtype = torch.float16,
    ):
        from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

        if weights_dir:
            # Manually specified weights directory, create a config
            self.cfg = StableDiffusionDreamboothLoRAConfig(model_name=model_name, attn_procs=str(weights_dir))
            StableDiffusionLoRA.configs[model_name] = self.cfg
        else:
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

        self.pipe = DiffusionPipeline.from_pretrained(self.cfg.model_name, torch_dtype=self.dtype)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(self.device)

        # Update attention processors
        self.pipe.load_lora_weights(weights_dir, weight_name="pytorch_lora_weights.safetensors")

    def update_attn_procs(self, model_name: str):
        """Update attention processors."""
        try:
            cfg = StableDiffusionLoRA.configs[model_name]
        except KeyError:
            raise ValueError(
                f"Invalid model_name: {model_name}, available models: {StableDiffusionLoRA.configs.keys()}"
            )

        if self.cfg.model_name != cfg.model_name:
            raise ValueError(f"Invalid model weights, [new_model={model_name}, expected_model={self.cfg.model_name}]")

        if not Path(self.cfg.attn_procs).exists():
            raise IOError(f"Failed to find attention processors [path={self.cfg.attn_procs}].")

        logger.debug(f"Updating attention processors [path={self.cfg.attn_procs}].")
        self.pipe.unet.load_attn_procs(self.cfg.attn_procs)
        logger.debug(f"Updated attention processors [path={self.cfg.attn_procs}].")

    def __call__(
        self,
        prompts: Union[str, List[str]],
        num_images: int = 1,
        num_inference_steps: int = 30,
        height: int = None,
        width: int = None,
    ) -> List[Image.Image]:
        """Generate images from text prompt."""
        if isinstance(prompts, str):
            prompts = [prompts]
        return self.pipe(
            prompts * num_images,
            num_inference_steps=num_inference_steps,
            height=height if height is not None else self.cfg.resolution,
            width=width if width is not None else self.cfg.resolution,
        ).images
