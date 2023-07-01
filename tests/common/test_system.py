from nos.common.system import (
    get_docker_info,
    get_nvidia_smi,
    get_system_info,
    get_torch_cuda_info,
    get_torch_info,
    get_torch_mps_info,
    has_docker,
    has_gpu,
    has_nvidia_docker,
    has_nvidia_docker_runtime_enabled,
)
from nos.test.utils import skip_if_no_torch_cuda


def test_system_info():
    info = get_system_info(docker=True, gpu=False)
    assert "system" in info
    assert "cpu" in info
    assert "memory" in info
    assert "docker" in info
    assert "gpu" not in info

    assert info["docker"]["version"] is not None
    assert info["docker"]["sdk_version"] is not None
    assert info["docker"]["compose_version"] is not None


@skip_if_no_torch_cuda
def test_system_info_with_gpu():
    info = get_system_info(docker=True, gpu=True)
    assert "docker" in info
    assert "gpu" in info
    assert len(info["gpu"]["devices"]) > 0


def test_system_utilities_cpu():
    assert has_docker(), "Docker not installed."
    assert get_docker_info() is not None

    assert get_torch_info() is not None, "torch unavailable."
    assert get_torch_cuda_info() is None, "No GPU detected via torch.cuda."
    assert get_torch_mps_info() is None


@skip_if_no_torch_cuda
def test_system_utilities_gpu():
    assert has_gpu(), "No GPU detected."
    assert get_nvidia_smi() is not None

    assert has_nvidia_docker(), "NVIDIA Docker not installed."
    assert has_nvidia_docker_runtime_enabled(), "No GPU detected within NVIDIA Docker."

    assert get_torch_info() is not None, "torch unavailable."
    assert get_torch_cuda_info() is not None, "No GPU detected via torch.cuda."
