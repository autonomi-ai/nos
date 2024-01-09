from typing import List, Union

import torch


class CustomModel:
    def __init__(self, model_name: str = "custom/custom-model"):
        from nos.logging import logger

        assert torch.cuda.is_available(), "CUDA not available"
        self.model_name = model_name
        logger.debug(f"Loading model {model_name}")

    @torch.inference_mode()
    def __call__(self, prompts: Union[str, List[str]]) -> List[str]:
        return [self.model_name + ": " + prompt for prompt in prompts]
