from typing import Any

import torch


class CustomModel:
    def __init__(self, model_name: str = "custom/custom-model"):
        self.model_name = model_name
        assert not torch.cuda.is_available(), "CUDA should not be available here"

    def __call__(self, arg1: Any) -> Any:
        return arg1

    def method1(self, arg1: Any) -> Any:
        return arg1

    def method2(self, arg1: Any) -> Any:
        return arg1
