import os
import platform
import subprocess
from functools import lru_cache
from io import StringIO
from typing import Any, Dict, Optional, Union

import pandas as pd
from cpuinfo import get_cpu_info
from psutil import cpu_count, cpu_freq, virtual_memory

from nos.logging import logger


@lru_cache(maxsize=1)
def cpu_info() -> Dict[str, Any]:
    """Get cached CPU information."""
    return get_cpu_info()


def sh(command: str) -> None:
    """Execute shell command, returning stdout."""
    try:
        output = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return output.stdout.strip()
    except subprocess.CalledProcessError:
        logger.debug(f"Failed to execute command: {command}")
        return None


def get_nvidia_smi(df: bool = False) -> Optional[Union[str, pd.DataFrame]]:
    """Get nvidia-smi details, if installed.

    Args:
        df: Return as pandas DataFrame.

    Returns:
        nvidia-smi details as string or DataFrame.
    """
    output = sh(
        "nvidia-smi --query-gpu=name,driver_version,pcie.link.gen.max,pcie.link.gen.current,memory.total --format=csv"
    )
    if output is None or not df:
        return output
    return pd.read_csv(StringIO(output), sep=", ")


def has_gpu() -> bool:
    """Check if GPU is available."""
    return get_nvidia_smi() is not None


def has_docker() -> bool:
    """Check if Docker is available."""
    return sh("docker --version") is not None


def is_inside_docker() -> bool:
    """Check if within Docker."""
    cgroup = "/proc/self/cgroup"
    return os.path.isfile("/.dockerenv") or os.path.isfile(cgroup) and any("docker" in line for line in open(cgroup))


def is_apple() -> bool:
    """Check if CPU is Apple."""
    return platform.system() == "Darwin"


def is_apple_silicon() -> bool:
    """Check if CPU is Apple Silicon.

    Note (spillai):
        >> arch = "arm64" if is_apple_silicon() else "x86_64"
    """
    info = cpu_info()
    brand = info["brand_raw"]
    return "apple m1" in brand.lower() or "apple m2" in brand.lower()


def has_nvidia_docker() -> bool:
    """Check if NVIDIA Docker is available."""
    return sh("nvidia-docker") is not None


def has_nvidia_docker_runtime_enabled() -> bool:
    """Check if NVIDIA Docker runtime is available."""
    return sh("docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi") is not None


def get_torch_info() -> Optional[Dict[str, Any]]:
    """Get torch info, if installed."""
    try:
        import torch

        return {
            "version": torch.__version__,
        }
    except ModuleNotFoundError:
        return None


def get_torch_cuda_info() -> Optional[Dict[str, Any]]:
    """Get torch CUDA info, if installed and available."""
    try:
        import torch

        assert torch.cuda.is_available()
        assert torch.cuda.device_count() > 0
        cuda_info = {
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count(),
            "devices": [],
        }

        # Note (spillai): NVIDIA SMI does not report the same order as torch.cuda
        # See torch.cuda and PCI_BUS_ID for more details.
        try:
            smi_df = get_nvidia_smi(df=True)
            cuda_info["driver_version"] = smi_df.driver_version[0]
        except Exception as e:
            logger.error(f"Failed to get nvidia-smi details: {e}")
            cuda_info["driver_version"] = None

        # Get GPU details via torch.cuda
        for device_id in range(torch.cuda.device_count()):
            device = torch.cuda.get_device_properties(device_id)
            device_info = {
                "device_id": device_id,
                "device_name": device.name,
                "device_capability": f"{device.major}.{device.minor}",
                "total_memory": device.total_memory,
                "total_memory_str": f"{device.total_memory / 1024 / 1024 / 1024:.2f} GB",
                "multi_processor_count": device.multi_processor_count,
            }
            cuda_info["devices"].append(device_info)
        return cuda_info
    except (ImportError, AssertionError):
        return None


def get_torch_mps_info() -> Optional[Dict[str, Any]]:
    """Get torch MPS info, if installed and available."""
    try:
        import torch

        assert torch.backends.mps.is_available()
        mps_info = {
            "is_macos13_or_newer": torch.backends.mps.is_macos13_or_newer(),
        }
        return mps_info
    except (ImportError, AssertionError):
        return None


def get_docker_info() -> Optional[Dict[str, Any]]:
    """Get docker version, if installed."""
    version = None
    try:
        import docker

        version = docker.__version__
    except (ImportError, docker.errors.APIError):
        pass

    compose_version = sh("docker compose version") or sh("docker-compose version")
    return {
        "version": sh("docker --version"),
        "sdk_version": version,
        "compose_version": compose_version,
    }


def get_system_info(docker: bool = False, gpu: bool = False) -> Dict[str, Any]:
    """Get system information (including CPU, GPU, RAM, etc.)"""
    cpu = cpu_info()
    vmem = virtual_memory()

    info = {
        "system": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "architecture": platform.architecture(),
            "processor": platform.processor(),
            "python_implementation": platform.python_implementation(),
        },
        "cpu": {
            "model": cpu["brand_raw"],
            "architecture": cpu["arch_string_raw"],
            "cores": {
                "physical": cpu_count(logical=False),
                "total": cpu_count(logical=True),
            },
            "frequency": cpu_freq().max,
            "frequency_str": f"{(cpu_freq().max / 1000):.2f} GHz",
        },
        "memory": {
            "total": vmem.total,
            "used": vmem.used,
            "available": vmem.available,
        },
        "torch": get_torch_info(),
    }
    if docker:
        info["docker"] = get_docker_info()
    if gpu:
        torch_cuda_info = get_torch_cuda_info()
        torch_mps_info = get_torch_mps_info()
        if torch_cuda_info is not None:
            info["gpu"] = torch_cuda_info
        elif torch_mps_info is not None:
            info["gpu"] = torch_mps_info
        else:
            info["gpu"] = None
    return info
