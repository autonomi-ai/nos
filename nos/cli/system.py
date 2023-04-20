import platform
import subprocess
from typing import Optional

import psutil
import torch
import typer
from rich.console import Console
from rich.panel import Panel

import docker
from docker.errors import APIError


system_cli = typer.Typer(name="system", help="NOS System CLI.", no_args_is_help=True)
console = Console()


def sh(command: str) -> None:
    """Execute shell command, returning stdout."""
    try:
        output = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return output.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def get_docker_version() -> Optional[str]:
    """Get docker version, if installed."""
    return sh("docker --version")


def get_nvidia_smi() -> Optional[str]:
    """Get nvidia-smi details, if installed."""
    return sh(
        "nvidia-smi --query-gpu=name,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,memory.total --format=csv"
    )


@system_cli.command("info")
def _system_info() -> None:
    """Get system information (including CPU, GPU, RAM, etc.)"""

    # Get system info
    sys_info = f"""
    System: {platform.system()}
    Release: {platform.release()}
    Version: {platform.version()}
    Machine: {platform.machine()}
    Architecture: {platform.architecture()}
    Processor Type: {platform.processor()}
    Python Implementation: {platform.python_implementation()}
    CPU Count: {psutil.cpu_count()}
    """
    console.print(Panel(sys_info, title="System"))

    # Get docker version
    console.print(Panel(f"{get_docker_version()}", title="Docker"))

    # Get nvidia-smi details
    console.print(Panel(f"{get_nvidia_smi()}", title="nvidia-smi"))

    # Check if GPU is available (via torch)
    torch_gpu_info = ""
    try:
        torch_gpu_info = (
            f"""GPU Count: {torch.cuda.device_count()}\n"""
            f"""CUDA Version: {torch.version.cuda}\n"""
            f"""CUDNN Version: {torch.backends.cudnn.version()}\n"""
        )
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch_gpu_info += "\n" if i > 0 else ""
                torch_gpu_info += f"""[{i}] Device Properties: ({torch.cuda.get_device_properties(i)})"""
    except ModuleNotFoundError:
        torch_gpu_info = "GPU: None"
    console.print(Panel(torch_gpu_info, title="Torch"))

    # Check if GPU is available
    CUDA_RUNTIME_IMAGE = "nvidia/cuda:11.0.3-runtime-ubuntu18.04"
    nvidia_docker_gpu_info = ""
    try:
        # Get the output of nvidia-smi running in the container
        client = docker.from_env()
        container = client.containers.run(CUDA_RUNTIME_IMAGE, "nvidia-smi", detach=True)
        for i, log in enumerate(container.logs(stream=True)):
            nvidia_docker_gpu_info += "\n" if i > 0 else ""
            nvidia_docker_gpu_info += f"{log.strip().decode()}"
        container.stop()
    except (APIError, ModuleNotFoundError):
        nvidia_docker_gpu_info = "Failed to run nvidia-smi within docker container"
    console.print(Panel(nvidia_docker_gpu_info, title="nvidia-smi (docker)"))
