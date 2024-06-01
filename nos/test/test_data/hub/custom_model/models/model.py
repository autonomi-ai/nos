class CustomModel:
    def __init__(self, arg1: str, kwarg_int: int = 1, kwarg_str: str = "2"):
        pass

    def __call__(self, prompts: str) -> str:
        return prompts
