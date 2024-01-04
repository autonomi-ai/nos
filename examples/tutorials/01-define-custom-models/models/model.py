from typing import List, Union

import torch


class CustomModel:
    def __init__(self, model_name: str = "custom/custom-model"):
        from nos.logging import logger

        logger.debug(f"Loading model {model_name}")
        self.model_name = model_name

    def __call__(
        self,
        prompts: Union[str, List[str]],
    ) -> List[str]:
        with torch.inference_mode():
            return [self.model_name + ": " + prompt for prompt in prompts]

    def method1(self, arg1: str = "arg1") -> str:
        return arg1

    def method2(self, arg2: str = "arg2") -> str:
        return arg2
