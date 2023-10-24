import os

from nos.logging import logger


def hf_login(write_permission: bool = False) -> str:
    """Login to huggingface hub."""
    from huggingface_hub import login

    token = os.getenv("HUGGINGFACE_HUB_TOKEN", None)
    if token is None:
        raise ValueError("HUGGINGFACE_HUB_TOKEN is not set")
    logger.debug(f"Logging into HF with token={token[:4]}...{token[-4:]}")
    login(
        token=token, write_permission=write_permission
    )  # Login to HF (required for private repos with write permission)
    return token
