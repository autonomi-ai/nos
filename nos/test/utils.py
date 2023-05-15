"""Various test utilities."""
from enum import Enum
from pathlib import Path

import pytest
import torch


NOS_TEST_DATA_DIR = Path(__file__).parent.parent.parent / "tests/test_data"
NOS_TEST_IMAGE = NOS_TEST_DATA_DIR / "test.jpg"
NOS_TEST_VIDEO = NOS_TEST_DATA_DIR / "test.mp4"
NOS_TEST_AUDIO = NOS_TEST_DATA_DIR / "test_speech.flac"


class PyTestGroup(Enum):
    """pytest group for grouping model tests, benchmarks etc."""

    UNIT = "unit"
    INTEGRATION = "integration"
    HUB = "hub"
    BENCHMARK = "benchmark"
    BENCHMARK_MODELS = "benchmark-models"


def skip_if_no_torch_cuda(test_case):
    """Decorator marking a test that requires torch cuda devices.

    These tests are skipped when torch.cuda.is_available() is set to False. If
    `CUDA_VISIBLE_DEVICES=""` then, the decorated test is not run.
    """
    return pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")(test_case)


def skip_all_if_no_torch_cuda():
    """Decorator sugar to mark all tests in a file that requires torch cuda devices.

    Usage:

        To mark all tests in a file that requires torch cuda devices,
        add the following:

        ```python
        pytestmark = skip_all_if_no_torch_cuda()
        ```
    """
    return pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
