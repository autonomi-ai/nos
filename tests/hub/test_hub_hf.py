import pytest

from nos.logging import logger


def test_hf_login():
    import os

    from nos.hub import hf_login

    if os.getenv("HUGGINGFACE_HUB_TOKEN", None) is None:
        logger.warning("HUGGINGFACE_HUB_TOKEN not set")
        pytest.skip("HUGGINGFACE_HUB_TOKEN not set")

    hf_login()
