"""Various test utilities."""
import os
import unittest
from enum import Enum
from pathlib import Path

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


def benchmark(test_case):
    """
    Decorator marking a test that is a benchmark (slow).

    These tests are triggered when `NOS_TEST_BENCHMARK=1`, and defaults to False.
    """
    print(f"benchmark: {bool(os.getenv('NOS_TEST_BENCHMARK', default=False))}")
    return unittest.skipUnless(
        bool(os.getenv("NOS_TEST_BENCHMARK", default=False)),
        "test requires NOS_TEST_BENCHMARK=1",
    )(test_case)


def requires_torch_cuda(test_case):
    """
    Decorator marking a test that requires torch cuda devices.

    These tests are skipped when torch.cuda.is_available() is set to False. If
    `CUDA_VISIBLE_DEVICES=""` then, the decorated test is not run.
    """
    return unittest.skipUnless(torch.cuda.is_available(), "test requires torch.cuda.is_available() to be True")(
        test_case
    )
