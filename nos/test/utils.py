"""Various test utilities."""
from enum import Enum
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from nos.common.system import has_gpu
from nos.logging import logger


NOS_TEST_DATA_DIR = Path(__file__).parent / "test_data"
NOS_TEST_IMAGE = NOS_TEST_DATA_DIR / "test.jpg"
NOS_TEST_VIDEO = NOS_TEST_DATA_DIR / "test.mp4"
NOS_TEST_AUDIO = NOS_TEST_DATA_DIR / "test_speech.flac"

AVAILABLE_RUNTIMES = ["auto", "cpu"]

if has_gpu():
    AVAILABLE_RUNTIMES += ["gpu"]


class PyTestGroup(Enum):
    """pytest group for grouping model tests, benchmarks etc."""

    UNIT = "unit"
    INTEGRATION = "integration"
    STRESS = "stress"
    HUB = "hub"
    BENCHMARK = "benchmark"
    MODEL_PROFILE = "model-profile"
    MODEL_BENCHMARK = "model-benchmark"
    MODEL_COMPILATION = "model-compilation"


def skip_if_no_torch_cuda(test_case):
    """Decorator marking a test that requires torch cuda devices.

    These tests are skipped when torch.cuda.is_available() is set to False. If
    `CUDA_VISIBLE_DEVICES=""` then, the decorated test is not run.
    """
    import torch

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
    import torch

    return pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


def download_url(url: str, filename: str = None) -> Path:
    """Download URL to local cache dir or filename for testing purposes."""
    import requests

    from nos.constants import NOS_CACHE_DIR

    if filename is None:
        test_dir = NOS_CACHE_DIR / "test_data"
        test_dir.mkdir(parents=True, exist_ok=True)
        basename = Path(url).name
        path = test_dir / basename
    else:
        path = Path(str(filename))

    if not path.exists():
        logger.debug(f"Downloading [url={url}, path={path}]")
        with NamedTemporaryFile(delete=False) as f:
            f.write(requests.get(url).content)
            f.flush()
            Path(f.name).rename(path)
        assert path.exists(), f"Failed to download {url}"
    return path


def get_benchmark_video() -> Path:
    VIDEO_URL = "https://zackakil.github.io/video-intelligence-api-visualiser/assets/test_video.mp4"
    return download_url(VIDEO_URL)


def get_benchmark_audio() -> Path:
    AUDIO_URL = "https://huggingface.co/datasets/reach-vb/random-audios/resolve/main/sam_altman_lex_podcast_367.flac"
    return download_url(AUDIO_URL)
