from typing import List, Union

import torch
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from PIL import Image


class AnimateDiff:
    def __init__(self, model_name: str = "Lykon/dreamshaper-7"):
        # Assert that CUDA is available for GPU acceleration
        assert torch.cuda.is_available()

        # Initialize the motion adapter from a pretrained model
        adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", use_safetensors=True)

        # Initialize the pipeline from a pretrained model and attach the motion adapter
        self.pipe = AnimateDiffPipeline.from_pretrained(model_name, motion_adapter=adapter, use_safetensors=True)

        # Initialize the scheduler from a pretrained model with specified parameters
        scheduler = DDIMScheduler.from_pretrained(
            model_name,
            subfolder="scheduler",
            beta_schedule="linear",
            clip_sample=False,
            timestep_spacing="linspace",
            steps_offset=1,
        )
        self.pipe.scheduler = scheduler

        # Move the pipeline to the GPU and set the data type to float16 for efficiency
        self.pipe.to(torch_device="cuda", torch_dtype=torch.float16)

        # Enable memory saving features in the pipeline
        self.pipe.enable_vae_slicing()
        self.pipe.enable_model_cpu_offload()

    def __call__(
        self,
        prompts: Union[str, List[str]],
        negative_prompts: Union[str, List[str]] = None,
        num_inference_steps: int = 25,
        num_frames: int = 16,
        height: int = None,
        width: int = None,
        guidance_scale: float = 7.5,
        seed: int = -1,
    ) -> List[Image.Image]:
        """Generate animated images from text prompt."""

        # Convert single string prompts to list of strings
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Initialize a random number generator on the GPU
        g = torch.Generator(device="cuda")

        # Set the seed for the random number generator if specified
        if seed != -1:
            g.manual_seed(seed)
        else:
            g.seed()

        # Generate the animated images
        yield from self.pipe(
            prompt=prompts,
            negative_prompt=negative_prompts,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=g,
            width=width,
            height=height,
        ).frames[0]
