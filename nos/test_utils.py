#  Copyright 2022-  Autonomi AI , Inc. All rights reserved.
"""Various test utilities."""
import os
import unittest
from pathlib import Path

import torch

NOS_TEST_DATA_DIR = Path(__file__).parent.parent / "tests/test_data"
NOS_TEST_IMAGE = NOS_TEST_DATA_DIR / "test.jpg"
NOS_TEST_VIDEO = NOS_TEST_DATA_DIR / "test.mp4"
NOS_TEST_AUDIO = NOS_TEST_DATA_DIR / "test_speech.flac"


def benchmark(test_case):
    """
    Decorator marking a test that is a benchmark (slow).

    These tests are triggered when `NOS_TEST_BENCHMARKS=1`, and defaults to False.
    """
    return unittest.skipUnless(
        os.getenv("NOS_TEST_BENCHMARK", default=False), "slow test requires NOS_TEST_BENCHMARK=1"
    )(test_case)


def require_torch_cuda(test_case):
    """
    Decorator marking a test that requires torch cuda devices.

    These tests are skipped when torch.cuda.is_available() is set to False. If
    `CUDA_VISIBLE_DEVICES=""` then, the decorated test is not run.
    """
    return unittest.skipUnless(torch.cuda.is_available(), "test requires torch.cuda.is_available() to be True")(
        test_case
    )
