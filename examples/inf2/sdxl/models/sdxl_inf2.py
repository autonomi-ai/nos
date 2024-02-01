"""SDXL model accelerated with AWS Neuron (using optimum-neuron)."""
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
from PIL import Image

from nos.constants import NOS_CACHE_DIR
from nos.hub import HuggingFaceHubConfig
from nos.neuron.device import NeuronDevice


@dataclass(frozen=True)
class StableDiffusionInf2Config(HuggingFaceHubConfig):
    """SDXL model configuration for Inf2."""

    batch_size: int = 1
    """Batch size for the model."""

    image_height: int = 1024
    """Height of the image."""

    image_width: int = 1024
    """Width of the image."""

    compiler_args: Dict[str, Any] = field(
        default_factory=lambda: {"auto_cast": "matmul", "auto_cast_type": "bf16"}, repr=False
    )
    """Compiler arguments for the model."""

    @property
    def id(self) -> str:
        """Model ID."""
        return f"{self.model_name}-bs-{self.batch_size}-{self.image_height}x{self.image_width}-{self.compiler_args.get('auto_cast_type', 'fp32')}"


class StableDiffusionXLInf2:
    configs = {
        "stabilityai/stable-diffusion-xl-base-1.0-inf2": StableDiffusionInf2Config(
            model_name="stabilityai/stable-diffusion-xl-base-1.0",
        ),
    }

    def __init__(self, model_name: str = "stabilityai/stable-diffusion-xl-base-1.0-inf2"):
        from nos.logging import logger

        NeuronDevice.setup_environment()
        try:
            cfg = StableDiffusionXLInf2.configs[model_name]
        except KeyError:
            raise ValueError(f"Invalid model_name: {model_name}, available models: {self.configs.keys()}")
        self.logger = logger
        self.model = None
        self.__load__(cfg)

    def __load__(self, cfg: StableDiffusionInf2Config):
        from optimum.neuron import NeuronStableDiffusionXLPipeline

        if self.model is not None:
            self.logger.debug(f"De-allocating existing model [cfg={self.cfg}, id={self.cfg.id}]")
            del self.model
            self.model = None
        self.cfg = cfg

        # Load model from cache if available, otherwise load from HF and compile
        # (cache is specific to model_name, batch_size and sequence_length)
        self.logger.debug(f"Loading model [cfg={self.cfg}, id={self.cfg.id}]")
        cache_dir = NOS_CACHE_DIR / "neuron" / self.cfg.id
        if Path(cache_dir).exists():
            self.logger.debug(f"Loading model from {cache_dir}")
            self.model = NeuronStableDiffusionXLPipeline.from_pretrained(str(cache_dir))
            self.logger.debug(f"Loaded model from {cache_dir}")
        else:
            input_shapes = {
                "batch_size": self.cfg.batch_size,
                "height": self.cfg.image_height,
                "width": self.cfg.image_width,
            }
            self.model = NeuronStableDiffusionXLPipeline.from_pretrained(
                self.cfg.model_name, export=True, **self.cfg.compiler_args, **input_shapes
            )
            self.model.save_pretrained(str(cache_dir))
            self.logger.debug(f"Saved model to {cache_dir}")
        self.logger.debug(f"Loaded neuron model [id={self.cfg.id}]")

    @torch.inference_mode()
    def __call__(
        self,
        prompts: Union[str, List[str]],
        num_images: int = 1,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
    ) -> List[Image.Image]:
        """Generate images from text prompt."""

        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(prompts, list) and len(prompts) != 1:
            raise ValueError(f"Invalid number of prompts: {len(prompts)}, expected: 1")
        if height != self.cfg.image_height or width != self.cfg.image_width:
            cfg = replace(self.cfg, image_height=height, image_width=width)
            self.logger.debug(f"Re-loading model [cfg={cfg}, id={cfg.id}, prev_id={self.cfg.id}]")
            self.__load__(cfg)
            assert self.model is not None
        return self.model(
            prompts,
            num_images_per_prompt=num_images,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images
