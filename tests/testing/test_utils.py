from pathlib import Path

import pytest

from nos.test.utils import skip_if_no_torch_cuda


@skip_if_no_torch_cuda
def test_torch_cuda():
    """This test is skipped when torch.cuda.is_available() is False."""
    import torch

    assert torch.cuda.is_available()


@pytest.mark.benchmark
def test_benchmark():
    """This test is demarkated as a benchmark (slow), and will only run when NOS_TEST_BENCHMARK=1."""
    assert True


@pytest.mark.benchmark
def test_benchmark_data():
    from nos.test.utils import get_benchmark_audio, get_benchmark_video

    filename = str(get_benchmark_audio())
    assert Path(filename).exists(), f"Failed to download {filename}"

    filename = str(get_benchmark_video())
    assert Path(filename).exists(), f"Failed to download {filename}"
